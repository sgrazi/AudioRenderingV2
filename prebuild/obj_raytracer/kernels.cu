#include "./kernels.cuh"
#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void fillZeros(float *buf, int size)
{
    int thread = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread < size)
        buf[thread] = 0.f;
}

void fillWithZeroesKernel(float *buf, int size)
{
    int numThreads = 256;
    int numBlocks;
    if (size % numThreads != 0) {
        numBlocks = (size / numThreads) + 1;
    } else {
        numBlocks = size / numThreads;
    }
    fillZeros<<<numBlocks, numThreads>>>(buf, size);
}

__global__ void vectorMultiply(float* a, float* b, float* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        c[index] = a[index] * b[index];
    }
}

__global__ void convolute_toeplitz_lower_matrix(float* samples, float* IR, size_t ir_size, float* outputBuffer){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // where in the "matrix" are we
    int row, col;
    for (int i = 1; i <= ir_size; i++) {
        if (index < i) {
            row = i - 1;
            col = index;
            break;
        }
        index -= i;
    }
    
    /*
    because cell with col = row has I0
    then cell with col = row - 1 has I1 (because we moved column one to the left)
    then cell with col = row - x has Ix
    then row - col = x
    */
    int ir_index = row - col;
    atomicAdd(&outputBuffer[col], samples[row] * IR[ir_index]);
}

__global__ void convolute_toeplitz_lower_matrix_2d(float* samples, float* IR, size_t ir_size, float* outputBuffer) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= y && (x < ir_size && y < ir_size)) {
        atomicAdd(&outputBuffer[y*2], samples[x] * IR[y - x]);
        atomicAdd(&outputBuffer[(y*2) - 1], samples[x] * IR[y - x]);
    }
}

__global__ void convolute_toeplitz_vectors(float* samples, float* IR, size_t ir_size, float* outputBuffer){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int samples_offset = index / ir_size;
    int ir_index = index % ir_size;
    /*
    for this part of the convolution samples start at ir_len, thats why we sum ir_len to the samples index
    for each complete multiplication we (the whole IR vector * a subvector of samples) we need to continue with a new subvector
    this new subvector is the samples vector moved one unit forward, thats why we add a samples_offset to the samples index
    */
    atomicAdd(&outputBuffer[ir_size + samples_offset], samples[ir_size + samples_offset] * IR[ir_size - 1 - ir_index]);
}

__global__ void kerneloide(cufftComplex* IRData, float* IR, unsigned int ir_len) {
    for (int i = 0; i < ir_len; i++) { // todo, do this in kernel
        IRData[i].x = IR[i];
        IRData[i].y = 0.0f;
    }
}

__global__ void kerneloide2_0(int second, unsigned int secondsToProcess, unsigned int sampleRate, cufftComplex* sampleData, float* samples, cufftHandle samplesPlan, cufftComplex* resultData, cufftComplex* IRData) {
    for (int i = second * sampleRate; i < (second + 2) * sampleRate; i++) {
        if (i < (second + 1) * sampleRate) {
            sampleData[i].x = samples[i];
            sampleData[i].y = 0.0f;
        }
        else {
            sampleData[i].x = 0.0f;
            sampleData[i].y = 0.0f;
        }
    }
}

__global__ void kerneloide2_1(int second, unsigned int secondsToProcess, unsigned int sampleRate, cufftComplex* sampleData, float* samples, cufftHandle samplesPlan, cufftComplex* resultData, cufftComplex* IRData) {
    for (int i = second * sampleRate; i < (second + 2) * sampleRate; i++) {
        resultData[i].x += sampleData[i].x * IRData[i].x - sampleData[i].y * IRData[i].y;
        resultData[i].y += sampleData[i].x * IRData[i].y + sampleData[i].y * IRData[i].x;
    }
}


void convolute_toeplitz_in_gpu(float* samples, float* IR, int ir_len, float* outputBuffer){
    //printf("ir_len: %d", ir_len);
    const int threadsPerBlock = 256;

    // first part, lower matrix multiplication
    /*
    cantidad operaciones = (ir_len * (ir_len + 1)) / 2
    esto es porque en la multiplicacion de la matriz triangular inferior
    la primera fila tiene una operacion, la segunda dos, y la n-esima (hasta el tope que es ir_len)
    equivale a Σ(i) de i=0 a i=n la cual se resuelve con ((n * (n + 1)) / 2)
    */
    int blocksPerGrid;
    if (((ir_len * (ir_len + 1)) / 2) < threadsPerBlock)
        blocksPerGrid = 1;
    else
        blocksPerGrid = (((ir_len * (ir_len + 1)) / 2) / threadsPerBlock) + 1;
    //convolute_toeplitz_lower_matrix<<<blocksPerGrid, threadsPerBlock>>>(samples, IR, ir_len, outputBuffer);
    dim3 threadsPorBlocks(32,32);
    int aaa = (ir_len / 32) + 1;
    dim3 numBlocks(aaa, aaa);
    convolute_toeplitz_lower_matrix_2d << <numBlocks, threadsPorBlocks >> > (samples, IR, ir_len, outputBuffer);

    // second part, vector multiplication
    size_t samples_size = sizeof(samples) / sizeof(float);
    /*
    ya procesamos ir_len celdas del output
    quedan (samples_size - ir_len) celdas restantes
    cada celda tiene ir_len multiplicaciones
    */
    //blocksPerGrid = ((samples_size - ir_len) * ir_len) / threadsPerBlock;
    //convolute_toeplitz_vectors<<<blocksPerGrid, threadsPerBlock>>>(samples, IR, ir_len, outputBuffer); // todo se precisa un offset
}

void convolute_fourier_in_gpu(float* samples, float* IR, unsigned int samples_len, unsigned int ir_len, float* outputBuffer) {
    printf("entr oa fourier\n");
    printf("samples_len: %d\n", samples_len);
    printf("ir_len: %d\n", ir_len);
    const int threadsPerBlock = 256;
    const int batchSize = 1; // Number of batches
    const int sampleRate = (samples_len / 4); //since samples is 2 seconds of 2 channel audio 
    const int secondsToProcess = samples_len / sampleRate;

    // Allocate device memory for samples
    cufftComplex* sampleData;
    cudaMalloc((void**)&sampleData, sampleRate * 2 * sizeof(cufftComplex));
    // Allocate device memory for IR
    cufftComplex* IRData;
    /*float* d_IR;
    cudaMalloc(&d_IR, ir_len * sizeof(float));
    copy_to_gpu(IR, d_IR, ir_len * sizeof(float));*/
    cudaMalloc((void**)&IRData, ir_len * sizeof(cufftComplex));
    // Allocate device memory for output
    //cudaMalloc((void**)&IRData, ir_len * sizeof(cufftComplex));
    cufftComplex* resultData = new cufftComplex[samples_len];

    // Set up FFT plans
    cufftHandle samplesPlan;
    cufftPlan1d(&samplesPlan, sampleRate, CUFFT_C2C, batchSize);
    cufftHandle IRPlan;
    cufftPlan1d(&IRPlan, ir_len, CUFFT_C2C, batchSize);
    printf("init pronto\n");

    // Do FFT on IR, which will be reused a lot
    kerneloide << <1, 1 >> > (IRData, IR, ir_len);
    cufftExecC2C(IRPlan, IRData, IRData, CUFFT_FORWARD);

    // Finally convolute, second by second
    /*
    Basically we take each segment and we prolong it with 0's
    Then we do FFT and sum each segment into the total
    Then we invert the total
    Full algorithm can be found on https://www.dspguide.com/ch18/2.htm
    */
    for (int second = 0; second < secondsToProcess; second++) {
        // First second is samples, rest is 0's (this is why we do seconds + 2 as the upper limit)
        kerneloide2_0 << <1, 1 >> > (second, secondsToProcess, sampleRate, sampleData, samples, samplesPlan, resultData, IRData);
        cudaDeviceSynchronize();
        cufftExecC2C(samplesPlan, sampleData, sampleData, CUFFT_FORWARD);

        // (a + ib) (c + id) = (ac – bd) + i(ad + bc)
        kerneloide2_1 << <1, 1 >> > (second, secondsToProcess, sampleRate, sampleData, samples, samplesPlan, resultData, IRData);
        cudaDeviceSynchronize();
    }
    
    printf("convolucion hecha\n");

    // Invert result
    cufftHandle inversePlan;
    cufftPlan1d(&inversePlan, samples_len, CUFFT_C2C, batchSize);
    cufftExecC2C(inversePlan, resultData, resultData, CUFFT_INVERSE);
    printf("inversion hecha\n");

    // Move to output buffer

    for (int i = 0; i < samples_len; i++) {
        outputBuffer[i] = resultData[i].x;
        //copy_from_gpu(&resultData[i].x, &outputBuffer[i], sizeof(float));
    }
    // Clean up
    cufftDestroy(samplesPlan);
    cufftDestroy(IRPlan);
    cufftDestroy(inversePlan);
    cudaFree(sampleData);
    cudaFree(IRData);
    cudaFree(resultData);
}

void copy_from_gpu(float* device_pointer, float* host_pointer, size_t size) {
    CUDA_CHK(cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost));
};

void copy_to_gpu(float* host_pointer, float* device_pointer, size_t size) {
    CUDA_CHK(cudaMemcpy(device_pointer, host_pointer, size, cudaMemcpyHostToDevice));
};