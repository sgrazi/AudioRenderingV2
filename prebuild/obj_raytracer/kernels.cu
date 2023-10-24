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

__global__ void load_complex_vector(cufftComplex* complex_data, float* real_vector, unsigned int vector_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < vector_len){
        complex_data[idx].x = real_vector[idx];
        complex_data[idx].y = 0.0f;
    }
}

__global__ void load_sample_segment_complex_vector(int second, unsigned int sampleRate, unsigned int segment_size_in_seconds, cufftComplex* sampleData, float* samples) {
    // loads one second of samples and (segment_size_in_seconds - 1) seconds of zeros
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index = idx + second * sampleRate;
    if (idx < sampleRate * segment_size_in_seconds){
        if (idx < sampleRate) {
                sampleData[index].x = samples[index];
                sampleData[index].y = 0.0f;
            }
            else {
                sampleData[index].x = 0.0f;
                sampleData[index].y = 0.0f;
            }
    }
}

__global__ void multiply_samples_segment_and_ir(int second, unsigned int sampleRate, unsigned int segment_size_in_seconds, cufftComplex* sampleData, cufftComplex* resultData, cufftComplex* IRData) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int index = idx + second * sampleRate;
    if (idx < sampleRate * segment_size_in_seconds){
        // (a + ib) (c + id) = (ac – bd) + i(ad + bc)
        resultData[index].x += sampleData[index].x * IRData[idx].x - sampleData[index].y * IRData[idx].y;
        resultData[index].y += sampleData[index].x * IRData[idx].y + sampleData[index].y * IRData[idx].x;
    }
}

void convolute_fourier_in_gpu(float* samples, float* IR, unsigned int samples_len, unsigned int ir_len, float* outputBuffer) {
    const int threadsPerBlock = 256;
    int blocks;
    const int batchSize = 1; // Number of batches
    const int sampleRate = (samples_len / 4); //since samples is 2 seconds of 2 channel audio 
    const int secondsToProcess = samples_len / sampleRate;
    const int segment_size_in_seconds = 2;

    // Allocate device memory for samples
    cufftComplex* sampleData;
    cudaMalloc((void**)&sampleData, sampleRate * 2 * sizeof(cufftComplex));
    // Allocate device memory for IR
    cufftComplex* IRData;
    cudaMalloc((void**)&IRData, ir_len * sizeof(cufftComplex));
    // Allocate device memory for output
    cufftComplex* resultData;
    cudaMalloc((void**)&resultData, samples_len * sizeof(cufftComplex));
    // Set up FFT plans
    cufftHandle samplesPlan;
    cufftPlan1d(&samplesPlan, sampleRate, CUFFT_C2C, batchSize);
    cufftHandle IRPlan;
    cufftPlan1d(&IRPlan, ir_len, CUFFT_C2C, batchSize);
    printf("init pronto\n");

    // Do FFT on IR, which will be reused a lot
    blocks = (ir_len / threadsPerBlock) + 1;
    load_complex_vector << <threadsPerBlock, blocks >> > (IRData, IR, ir_len);
    cudaDeviceSynchronize();
    cufftExecC2C(IRPlan, IRData, IRData, CUFFT_FORWARD);

    // Finally convolute, second by second
    /*
    Basically we take each segment and we prolong it with 0's
    Then we do FFT and sum each segment into the total
    Then we invert the total
    Full algorithm can be found on https://www.dspguide.com/ch18/2.htm
    */
    blocks = (sampleRate * segment_size_in_seconds / threadsPerBlock) + 1;
    for (int second = 0; second < secondsToProcess; second++) {
        // First second is samples, rest is 0's (this is why we do seconds + 2 as the upper limit)
        load_sample_segment_complex_vector << <threadsPerBlock, blocks >> > (second, sampleRate, segment_size_in_seconds, sampleData, samples);
        cudaDeviceSynchronize();
        cufftExecC2C(samplesPlan, sampleData, sampleData, CUFFT_FORWARD);
        multiply_samples_segment_and_ir << <threadsPerBlock, blocks >> > (second, sampleRate, segment_size_in_seconds, sampleData, resultData, IRData);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    printf("convolucion hecha\n");

    // Invert result
    cufftHandle inversePlan;
    cufftPlan1d(&inversePlan, samples_len, CUFFT_C2C, batchSize);
    cufftExecC2C(inversePlan, resultData, resultData, CUFFT_INVERSE);
    printf("inversion hecha\n");
    
    // Move to output buffer

    for (int i = 0; i < samples_len; i++) {
        // outputBuffer[i] = resultData[i].x;
        copy_from_gpu(&resultData[i].x, &outputBuffer[i], sizeof(float));
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