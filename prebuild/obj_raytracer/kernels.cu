#include "./kernels.cuh"
#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void CHCK_CUFFT_RES(cufftResult_t res){
    if (res != 0)
    {
        fprintf(stderr, "CHCK_CUFFT_RES: %d\n", res);
        if (abort) exit(res);
    }
}

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

__global__ void load_sample_segment(int second, unsigned int sampleRate, unsigned int segment_size_in_seconds, float* segment, float* samples) {
    // loads one second of samples and (segment_size_in_seconds - 1) seconds of zeros
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int sampleIndex = idx + second * sampleRate;

    if (idx < sampleRate * segment_size_in_seconds){
        if (idx < sampleRate) {
            segment[idx] = samples[sampleIndex];
        }
        else {
            segment[idx] = 0.0f;
        }
    }
}

__global__ void multiply_samples_segment_and_ir(int second, unsigned int sampleRate, unsigned int segment_size_in_seconds, cufftComplex* sampleData, cufftComplex* IRData) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < sampleRate * segment_size_in_seconds){
        // (a + ib) (c + id) = (ac – bd) + i(ad + bc)
        cufftComplex temp = sampleData[idx];
        sampleData[idx].x = temp.x * IRData[idx].x - temp.y * IRData[idx].y;
        sampleData[idx].y = temp.x * IRData[idx].y + temp.y * IRData[idx].x;
    }
}

__global__ void add_segment_to_result_buffer(int second, int sampleRate, int segmentLen, float* segment, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("SAMPLE SEGMENT: %d\n", segment);
    if (index < segmentLen) {
        int outputIndex = second * sampleRate + index;
        atomicAdd(&output[outputIndex], segment[index]);
    }
}

__global__ void isAllZeros(float* arr, int size, int* result) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        if (arr[index] != 0.0f) {
            atomicExch(result, 1); // Set result to 1 if any element is not zero
        }
    }
}

// CUDA kernel to check each element if it's zero (complex numbers)
__global__ void isAllZerosComplex(cufftComplex* arr, int size, int* result) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        if (arr[index].x != 0.0f || arr[index].y != 0.0f) {
            atomicExch(result, 1); // Set result to 1 if any element is not zero
        }
    }
}

bool checkArrayZero(float* IR, unsigned int ir_len) {
    int* d_result; // Device result
    int h_result = 0; // Host result

    // Calculate grid and block sizes
    int blockSize = 256; // Number of threads per block
    int numBlocks = (ir_len + blockSize - 1) / blockSize;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    isAllZeros << <numBlocks, blockSize >> > (IR, ir_len, d_result);

    // Copy the result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_result);

    // Check the result
    if (h_result == 0) {
        printf("All elements are zero.\n");
        return true;
    }
    else {
        printf("There is at least one element that is not zero.\n");
        return false;
    }

    
}

void convolute_fourier_in_gpu(float* samples, float* IR, unsigned int samples_len, unsigned int sample_rate, unsigned int ir_len, float* outputBuffer) {
    const int threadsPerBlock = 256;
    int blocks;
    const int batchSize = 1; // Number of batches 
    const int secondsToProcess = samples_len / sample_rate;
    const int segment_size_in_seconds = 2;
    
    //int* d_result; // Device result
    //int h_result = 0; // Host result

    //// Calculate grid and block sizes
    //int blockSize = 256; // Number of threads per block
    //int numBlocks = (ir_len + blockSize - 1) / blockSize;
    //cudaMalloc(&d_result, sizeof(int));
    //cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    //// Launch the kernel
    //isAllZeros << <numBlocks, blockSize >> > (IR, ir_len, d_result);

    //// Copy the result back to host
    //cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    //// Check the result
    //if (h_result == 0) {
    //    printf("All elements are zero.\n");
    //}
    //else {
    //    printf("There is at least one element that is not zero.\n");
    //}

    //// Free GPU memory
    //cudaFree(d_result);

    // Allocate device memory for samples
    float* sampleSegment;
    int segmentSize = sample_rate * segment_size_in_seconds * sizeof(float);
    cudaMalloc((void**)&sampleSegment, segmentSize);
    cufftComplex* segmentData;
    cudaMalloc((void**)&segmentData, sample_rate * segment_size_in_seconds * sizeof(cufftComplex));
    // Allocate device memory for IR
    cufftComplex* IRData;
    cudaMalloc((void**)&IRData, ir_len * sizeof(cufftComplex));
    // Allocate device memory for output
    cufftComplex* resultData;
    cudaMalloc((void**)&resultData, samples_len * sizeof(cufftComplex));
    // Set up FFT plans
    cufftHandle segmentPlan;
    cufftPlan1d(&segmentPlan, sample_rate * segment_size_in_seconds, CUFFT_R2C, batchSize);
    cufftHandle IRPlan;
    cufftPlan1d(&IRPlan, ir_len, CUFFT_R2C, batchSize);
    // Invert result
    cufftHandle inversePlan;
    cufftPlan1d(&inversePlan, sample_rate * segment_size_in_seconds, CUFFT_C2R, batchSize);
    // Do FFT on IR, which will be reused a lot
    blocks = (ir_len / threadsPerBlock) + 1;
    cudaDeviceSynchronize();
    CHCK_CUFFT_RES(cufftExecR2C(IRPlan, IR, IRData));

    //int* d_result; // Device result
    //int h_result = 0; // Host result

    //// Allocate memory on the GPU
    //cudaMalloc(&d_result, sizeof(int));

    //// Copy the host array to the device (GPU)
    //cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    //// Calculate grid and block sizes
    //int blockSize = 256; // Number of threads per block
    //int numBlocks = (ir_len + blockSize - 1) / blockSize;

    //// Launch the kernel
    //isAllZerosComplex << <numBlocks, blockSize >> > (IRData, ir_len, d_result);

    //// Copy the result back to host
    //cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    //// Check the result
    //if (h_result == 0) {
    //    printf("COMPLEX = All elements are zero.\n");
    //}
    //else {
    //    printf("COMPLEX = There is at least one element that is not zero.\n");
    //}

    //cudaFree(d_result);


    // Finally convolute, second by second
    /*
    Basically we take each segment and we prolong it with 0's
    Then we do FFT and sum each segment into the total
    Then we invert the total
    Full algorithm can be found on https://www.dspguide.com/ch18/2.htm
    */
    blocks = ((sample_rate * segment_size_in_seconds) / threadsPerBlock) + 1;
    for (int second = 0; second < secondsToProcess; second++) {
        // First second is samples, rest is 0's (this is why we do seconds + 1 as the upper limit)
        load_sample_segment << <blocks, threadsPerBlock >> > (second, sample_rate, segment_size_in_seconds, sampleSegment, samples);
        cudaDeviceSynchronize();
        // FFT on the loaded segment, and convolute it with IR
        CHCK_CUFFT_RES(cufftExecR2C(segmentPlan, sampleSegment, segmentData));
        multiply_samples_segment_and_ir << <blocks, threadsPerBlock >> > (second, sample_rate, segment_size_in_seconds, segmentData, IRData);
        cudaDeviceSynchronize();
        // Inverse on the result and save it to buffer
        CHCK_CUFFT_RES(cufftExecC2R(inversePlan, segmentData, sampleSegment));
        blocks = (segmentSize + threadsPerBlock - 1) / threadsPerBlock;
        add_segment_to_result_buffer<<<blocks, threadsPerBlock>>>(second, sample_rate, sample_rate * segment_size_in_seconds, sampleSegment, outputBuffer);
        cudaDeviceSynchronize();
    }
    int* d_result; // Device result
    int h_result = 0; // Host result

    // Allocate memory on the GPU
    cudaMalloc(&d_result, sizeof(int));

    // Copy the host array to the device (GPU)
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
    int blockSize = 256; // Number of threads per block
    int numBlocks = (ir_len + blockSize - 1) / blockSize;

    // Launch the kernel
    isAllZeros << <numBlocks, blockSize >> > (outputBuffer, samples_len, d_result);

    // Copy the result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Check the result
    if (h_result == 0) {
        printf("output = All elements are zero.\n");
    }
    else {
        printf("output = There is at least one element that is not zero.\n");
    }

    cudaFree(d_result);

    // Clean up
    cufftDestroy(segmentPlan);
    cufftDestroy(IRPlan);
    cufftDestroy(inversePlan);
    cudaFree(sampleSegment);
    cudaFree(segmentData);
    cudaFree(IRData);
    cudaFree(resultData);
}

void copy_from_gpu(float* device_pointer, float* host_pointer, size_t size) {
    CUDA_CHK(cudaMemcpy(host_pointer, device_pointer, size, cudaMemcpyDeviceToHost));
};

void copy_to_gpu(float* host_pointer, float* device_pointer, size_t size) {
    CUDA_CHK(cudaMemcpy(device_pointer, host_pointer, size, cudaMemcpyHostToDevice));
};