#pragma once
#include <glm/glm.hpp>
#include <unordered_map>
#include <stdio.h>
#include <mutex>
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "OptixModel.h"
#include "Camera.h"
#include "kernels.cuh"

/*! a sample OptiX-7 renderer that demonstrates how to set up
    context, module, programs, pipeline, SBT, etc, and perform a
    valid launch that renders some pixel (using a simple test
    pattern, in this case */
class AudioRenderer
{
    // ------------------------------------------------------------------
    // publicly accessible interface
    // ------------------------------------------------------------------
public:
    /*! constructor - performs all setup, including initializing
      optix, creates modOptixModelpipeline, programs, SBT, etc. */
    AudioRenderer(const OptixModel *model, unsigned int ir_length_in_seconds, int sample_rate, std::vector<Material> materials, gdt::vec3f rays_per_dimension);

    /*! render one frame */
    void render();

    void convoluteLiveInput(double *h_inputBuffer, size_t h_inputBufferSize, CircularBuffer<double> *h_circularOutputBuffer);

    void convoluteAudioFile(float *h_inputBuffer, size_t h_inputBufferSize, float *h_outputBuffer_left, float *h_outputBuffer_right);

    void setEmitterPosInOptix(glm::vec3 pos);

    void setSphereCenterInOptix(glm::vec3 pos);

    void setThresholds(float energy, unsigned int max_bounces);

    void setBasePower(float base_power);

    void set_write_ir_to_file_flag(bool value);

    void set_write_output_to_file_flag(bool value);

    void normalizeAndMergeStereoOutput(double *d_outputBuffer_left, double *d_outputBuffer_right, size_t monoBufferLength, double *d_outputBuffer);

    void full_render_cycle(std::mutex *mutex, Sphere sphere, OptixModel *scene, gdt::vec3f camera_central_point, float camera_global_angle, float *audio_samples, size_t size_of_audio, float *outputBuffer_left, float *outputBuffer_right);

    void setMonoOutput(bool value);
    // void getIROnHostMem(float *h_ir, size_t ir_size);

protected:
    // ------------------------------------------------------------------
    // internal helper functions
    // ------------------------------------------------------------------

    /*! helper function that initializes optix and checks for errors */
    void initOptix();

    /*! creates and configures a optix device context (in this simple
      example, only for the primary GPU device) */
    void createContext();

    /*! creates the module that contains all the programs we are going
      to use. in this simple example, we use a single module from a
      single .cu file, using a single embedded ptx string */
    void createModule();

    /*! does all setup for the raygen program(s) we are going to use */
    void createRaygenPrograms();

    /*! does all setup for the miss program(s) we are going to use */
    void createMissPrograms();

    /*! does all setup for the hitgroup program(s) we are going to use */
    void createHitgroupPrograms();

    /*! assembles the full pipeline of all programs */
    void createPipeline();

    /*! constructs the shader binding table */
    void buildSBT();

    /*! build an acceleration structure for the given triangle mesh */
    OptixTraversableHandle buildAccel();

    /*! loads material data into map */
    std::unordered_map<int, Material> buildAbsorptionMap();

protected:
    /*! @{ CUDA device context and stream that optix pipeline will run
        on, as well as device properties for this device */
    CUcontext cudaContext;
    CUstream stream;
    cudaDeviceProp deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /*! @{ the pipeline we're building */
    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    LaunchParams launchParams;
    CUDABuffer launchParamsBuffer;
    /*! @} */

    CUDABuffer colorBuffer;

    /*! the model we are going to trace rays against */
    const OptixModel *model;
    const std::vector<Material> materials;

    /*! one buffer per input mesh */
    std::vector<CUDABuffer> vertexBuffer;
    /*! one buffer per input mesh */
    std::vector<CUDABuffer> indexBuffer;
    //! buffer that keeps the (final, compacted) accel structure
    CUDABuffer asBuffer;

    bool write_ir_to_file_flag = false;
    bool write_output_to_file_flag = false;
    bool isMono = false;
};
