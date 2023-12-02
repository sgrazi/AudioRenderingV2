#include <glm/glm.hpp>
#include <optix_function_table_definition.h>
#include "AudioRenderer.h"
#include "CUDABuffer.h"
#include "Utils.h"
#include "./kernels.cuh"

extern "C" char embedded_ptx_code[];

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    void *data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    void *data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
};

/*! constructor - performs all setup, including initializing
  optix, creates module, pipeline, programs, SBT, etc. */
AudioRenderer::AudioRenderer(const OptixModel *model, unsigned int buffer_size_in_seconds, int output_channels, int sample_rate):model(model)
{
    initOptix();

    std::cout << " creating optix context ..." << std::endl;
    createContext();

    std::cout << " setting up module ..." << std::endl;
    createModule();

    std::cout << " creating raygen programs ..." << std::endl;
    createRaygenPrograms();
    std::cout << " creating miss programs ..." << std::endl;
    createMissPrograms();
    std::cout << " creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    std::cout << " setting up pathtracer parameters ..." << std::endl;
    launchParams.size_x = 100;
    launchParams.size_y = 100;
    launchParams.size_z = 100;
    launchParams.traversable = buildAccel();
    int ir_lenght = buffer_size_in_seconds * output_channels * sample_rate;
    launchParams.ir_length = ir_lenght;
    launchParams.sample_rate = sample_rate;
    cudaMalloc(&launchParams.ir, launchParams.ir_length * sizeof(float));
    fillWithZeroesKernel(launchParams.ir, launchParams.ir_length);
    cudaDeviceSynchronize();

    std::cout << " setting up optix pipeline ..." << std::endl;
    createPipeline();

    std::cout << " building SBT ..." << std::endl;
    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));
    std::cout << " context, module, pipeline, etc, all set up ..." << std::endl;
}

OptixTraversableHandle AudioRenderer::buildAccel()
{
    vertexBuffer.resize(model->meshes.size());
    indexBuffer.resize(model->meshes.size());

    OptixTraversableHandle asHandle{0};

    std::vector<OptixBuildInput> triangleInput(model->meshes.size());
    std::vector<CUdeviceptr> d_vertices(model->meshes.size());
    std::vector<CUdeviceptr> d_indices(model->meshes.size());
    std::vector<uint32_t> triangleInputFlags(model->meshes.size());

    for (int meshID = 0; meshID < model->meshes.size(); meshID++)
    {
        // upload the model to the device: the builder
        TriangleMesh &mesh = *model->meshes[meshID];
        vertexBuffer[meshID].alloc_and_upload(mesh.vertex);
        indexBuffer[meshID].alloc_and_upload(mesh.index);

        triangleInput[meshID] = {};
        triangleInput[meshID].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // create local variables, because we need a *pointer* to the
        // device pointers
        d_vertices[meshID] = vertexBuffer[meshID].d_pointer();
        d_indices[meshID] = indexBuffer[meshID].d_pointer();

        triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
        triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertex.size();
        triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

        triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
        triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.index.size();
        triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

        triangleInputFlags[meshID] = 0;

        // in this example we have one SBT entry, and no per-primitive
        // materials:
        triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
        triangleInput[meshID].triangleArray.numSbtRecords = 1;
        triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
        triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
    }
    // ==================================================================
    // BLAS (Bottom-Level Acceleration Structure) setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                             &accelOptions,
                                             triangleInput.data(),
                                             (int)model->meshes.size(),
                                             &blasBufferSizes));

    // ==================================================================
    // prepare compaction
    // ==================================================================

    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================

    CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(optixContext,
                                0,
                                &accelOptions,
                                triangleInput.data(),
                                (int)model->meshes.size(),
                                tempBuffer.d_pointer(),
                                tempBuffer.sizeInBytes,

                                outputBuffer.d_pointer(),
                                outputBuffer.sizeInBytes,

                                &asHandle,

                                &emitDesc, 1));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    asBuffer.alloc(compactedSize);
    OPTIX_CHECK(optixAccelCompact(optixContext,
                                  /*stream:*/ 0,
                                  asHandle,
                                  asBuffer.d_pointer(),
                                  asBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // clean up
    // ==================================================================
    outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();

    return asHandle;
}

/*! helper function that initializes optix and checks for errors */
void AudioRenderer::initOptix()
{
    std::cout << " initializing optix..." << std::endl;

    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
        throw std::runtime_error(" no CUDA capable devices found!");
    std::cout << " found " << numDevices << " CUDA devices" << std::endl;

    OPTIX_CHECK(optixInit());
    std::cout << " successfully initialized optix... yay!" << std::endl;
}

static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *)
{
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

// creates and configures a optix device context (for the primary GPU device)
void AudioRenderer::createContext()
{
    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));
    CUDA_CHECK(StreamCreate(&stream));

    CUresult cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}

/*! creates the module that contains all the programs we are going
  to use. in this simple example, we use a single module from a
  single .cu file, using a single embedded ptx string */
void AudioRenderer::createModule()
{
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

    const std::string ptxCode = embedded_ptx_code;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(optixContext,
                                  &moduleCompileOptions,
                                  &pipelineCompileOptions,
                                  ptxCode.c_str(),
                                  ptxCode.size(),
                                  log, &sizeof_log,
                                  &module));
    if (sizeof_log > 1)
        printf("%s", log);
}

/*! does all setup for the raygen program(s) we are going to use */
void AudioRenderer::createRaygenPrograms()
{
    // we do a single ray gen program in this example:
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module; // Module holding single program
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &raygenPGs[0]));
    if (sizeof_log > 1)
        printf("%s", log);
}

/*! does all setup for the miss program(s) we are going to use */
void AudioRenderer::createMissPrograms()
{
    // we do a single ray gen program in this example:
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &missPGs[0]));
    if (sizeof_log > 1)
        printf("%s", log);
}

/*! does all setup for the hitgroup program(s) we are going to use */
void AudioRenderer::createHitgroupPrograms()
{
    // for this simple example, we set up a single hit group
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log, &sizeof_log,
                                        &hitgroupPGs[0]));
    if (sizeof_log > 1)
        printf("%s", log);
}

/*! assembles the full pipeline of all programs */
void AudioRenderer::createPipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    for (auto pg : raygenPGs)
        programGroups.push_back(pg);
    for (auto pg : missPGs)
        programGroups.push_back(pg);
    for (auto pg : hitgroupPGs)
        programGroups.push_back(pg);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixContext,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups.data(),
                                    (int)programGroups.size(),
                                    log, &sizeof_log,
                                    &pipeline));
    if (sizeof_log > 1)
        printf("%s", log);

    OPTIX_CHECK(optixPipelineSetStackSize(/* [in] The pipeline to configure the stack size for */
                                          pipeline,
                                          /* [in] The direct stack size requirement for direct
                                             callables invoked from IS or AH. */
                                          2 * 1024,
                                          /* [in] The direct stack size requirement for direct
                                             callables invoked from RG, MS, or CH.  */
                                          2 * 1024,
                                          /* [in] The continuation stack requirement. */
                                          2 * 1024,
                                          /* [in] The maximum depth of a traversable graph
                                             passed to trace. */
                                          1));
    if (sizeof_log > 1)
        printf("%s", log);
}

/*! constructs the shader binding table */
void AudioRenderer::buildSBT()
{
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<RaygenRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++)
    {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<MissRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++)
    {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    int numObjects = (int)model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for (int meshID = 0; meshID < numObjects; meshID++)
    {
        HitgroupRecord rec;
        // all meshes use the same code, so all same hit group
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[0], &rec));
        rec.data.color = gdt2glm(model->meshes[meshID]->diffuse);
        rec.data.mat_absorption = model->meshes[meshID]->material_absorption;
        rec.data.vertex = (glm::vec3 *)vertexBuffer[meshID].d_pointer();
        rec.data.index = (glm::ivec3 *)indexBuffer[meshID].d_pointer();
        hitgroupRecords.push_back(rec);
    }
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

/*! render one frame */
void AudioRenderer::render()
{
    vertexBuffer.clear();
    indexBuffer.clear();
    asBuffer.free();
    raygenRecordsBuffer.free();
    missRecordsBuffer.free();
    hitgroupRecordsBuffer.free();
    launchParamsBuffer.free();

    launchParams.traversable = buildAccel();
    fillWithZeroesKernel(launchParams.ir, launchParams.ir_length);
    cudaDeviceSynchronize();

    createPipeline();

    buildSBT();

    launchParamsBuffer.alloc(sizeof(launchParams));

    launchParamsBuffer.upload(&launchParams, 1);

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline, stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.size_x,
                            launchParams.size_y,
                            launchParams.size_z));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();

    /////////
    float *host = NULL;
    host = new float[launchParams.ir_length];
    copy_from_gpu(launchParams.ir, host, launchParams.ir_length * sizeof(float));
    std::ofstream outFile("output_ir.txt");

    // Check if the file is opened successfully
    if (!outFile.is_open())
    {
        std::cerr << "Error opening the file." << std::endl;
    }
    else
    {
        std::cout << "wrote ir to file" << std::endl;

        // Write each element of the float array to the file, one per line
        for (int i = 0; i < launchParams.ir_length; ++i)
        {
            outFile << host[i] << std::endl;
        }

        // Close the file
        outFile.close();
    }
}

void AudioRenderer::convolute(float *h_inputBuffer, size_t h_inputBufferSize, float *h_outputBuffer)
{
    // move inputBuffer to device
    float *d_inputBuffer;
    cudaMalloc(&d_inputBuffer, h_inputBufferSize);
    copy_to_gpu(h_inputBuffer, d_inputBuffer, h_inputBufferSize);

    // send launchParams.ir and d_inputBuffer and h_outputBuffer to kernel
    float *d_outputBuffer = NULL;
    size_t outputSize = h_inputBufferSize;
    cudaMalloc(&d_outputBuffer, h_inputBufferSize);
    // convolute_toeplitz_in_gpu(d_inputBuffer, launchParams.ir, launchParams.ir_length, d_outputBuffer);
    convolute_fourier_in_gpu(d_inputBuffer, launchParams.ir, h_inputBufferSize / sizeof(float), launchParams.sample_rate, launchParams.ir_length, d_outputBuffer);
    cudaDeviceSynchronize();
    // copy result to host
    copy_from_gpu(d_outputBuffer, h_outputBuffer, outputSize);

    // free
    cudaFree(d_inputBuffer);
    cudaFree(d_outputBuffer);
}

void AudioRenderer::setEmitterPosInOptix(glm::vec3 pos)
{
    launchParams.emitter_position = pos;
}

void AudioRenderer::setThresholds(float dist, float energy, unsigned int max_bounces)
{
    launchParams.dist_thres = dist;
    launchParams.energy_thres = energy;
    launchParams.max_bounces = max_bounces;
}

void AudioRenderer::setBasePower(float base_power)
{
    launchParams.base_power = base_power;
}

void AudioRenderer::getIROnHostMem(float *h_ir, size_t ir_size)
{
    copy_from_gpu(launchParams.ir, h_ir, ir_size);
}