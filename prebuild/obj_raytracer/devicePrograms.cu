#include <stdio.h>
#include <optix_device.h>
#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <gdt/math/vec.h>
#include <glm/glm.hpp>
#include "LaunchParams.h"
#include "PRD.h"
#include "curand_kernel.h"

#define SPEED_OF_SOUND 343 // grabbed from Cameelo/AudioRendering

/*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
extern "C" __constant__ LaunchParams optixLaunchParams;

// for this simple example, we have a single ray type
// RAY_TYPE_COUNT does not have an explicit value assigned, the compiler automatically assigns it a value one greater than the previous enumerator
enum
{
    SURFACE_RAY_TYPE = 0,
    RAY_TYPE_COUNT
};

static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void *ptr = reinterpret_cast<void *>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, uint32_t &i0, uint32_t &i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T *getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__radiance()
{
    const TriangleMeshSBTData &sbtData = *(const TriangleMeshSBTData *)optixGetSbtDataPointer();
    const float3 wrd = optixGetWorldRayDirection();
    const glm::vec3 rayDir = glm::vec3(wrd.x, wrd.y, wrd.z);
    PRD &prd = *(PRD *)getPRD<PRD>();

    const int primID = optixGetPrimitiveIndex();
    const glm::ivec3 index = sbtData.index[primID];
    const glm::vec3 P1 = sbtData.vertex[index.x];
    const glm::vec3 P2 = sbtData.vertex[index.y];
    const glm::vec3 P3 = sbtData.vertex[index.z];

    glm::vec3 U = P2 - P1;
    glm::vec3 V = P3 - P1;
    const glm::vec3 Ng = glm::normalize(glm::cross(U,V));

    const float u_barycentrics = optixGetTriangleBarycentrics().x;
    const float v_barycentrics = optixGetTriangleBarycentrics().y;
    const glm::vec3 P = (1.f - u_barycentrics - v_barycentrics) * P1 + u_barycentrics * P2 + v_barycentrics * P3;

    prd.distance += distance(P, prd.prev_position);

        // Get Ray Id
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const int iz = optixGetLaunchIndex().z;

    printf("%f", sbtData.mat_absorption);

    if (sbtData.mat_absorption < 0) {
        // we identify the receiver with a negative absorption
        printf("hit;");
        //printf("x(end+1) = %f;y(end+1) = %f;z(end+1) = %f;color(end+1) = %f;rebotes(end+1)= %d;", P.x, P.y, P.z, prd.distance, prd.recursion_depth);
        float elapsed_time = prd.distance / SPEED_OF_SOUND;
        int array_pos = round(elapsed_time * optixLaunchParams.sample_rate);
        float* ir = optixLaunchParams.ir;
        if (array_pos < optixLaunchParams.ir_length)
        {
            atomicAdd(&ir[array_pos], prd.remaining_factor);
        }
        prd.recursion_depth = -1;
    }
    else {
        prd.direction = prd.direction - 2.0f * dot(prd.direction, Ng) * Ng;
        prd.remaining_factor *= (1 - sbtData.mat_absorption);
        prd.recursion_depth++;
    }
        
    prd.prev_position = P + (1e-3f * prd.direction);
}

extern "C" __global__ void __anyhit__radiance()
{
}

extern "C" __global__ void __miss__radiance()
{
    PRD &prd = *(PRD *)getPRD<PRD>();
    prd.recursion_depth = -1;
}

extern "C" __global__ void __raygen__renderFrame()
{

    // TODO, check if dimensions are three dimensional
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const int iz = optixGetLaunchIndex().z;
    const int x_rays = optixGetLaunchDimensions().x;
    const int y_rays = optixGetLaunchDimensions().y;
    const int z_rays = optixGetLaunchDimensions().z;

    // the values we store the PRD pointer in:
    // Note: Payload Reference Data and represents the data structure used to pass information between shaders during the ray tracing process
    PRD prd;
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);
    prd.remaining_factor = (optixLaunchParams.base_power) / (x_rays * y_rays * z_rays);
    prd.distance = 0;
    prd.prev_position = optixLaunchParams.emitter_position;
    prd.recursion_depth = 0;

    int tid = iz * x_rays * y_rays + iy * x_rays + ix;

    curandState state;
    curand_init((unsigned long long)clock64(), tid, 0, &state);

    double dx = curand_uniform(&state) * 2.0f - 1.0f;
    double dy = curand_uniform(&state) * 2.0f - 1.0f;
    double dz = curand_uniform(&state) * 2.0f - 1.0f;
    // it is bound to happen that some threads have (0,0,0) as their vector
    if (dx != 0.0 || dy != 0.0 || dz != 0.0)
    {
        double length = std::sqrt(dx * dx + dy * dy + dz * dz);
        dx /= length;
        dy /= length;
        dz /= length;

        // printf("sending to %f,%f,%f...\n", dx, dy, dz);
        prd.direction = {dx, dy, dz};
        while (prd.distance < optixLaunchParams.dist_thres &&
               prd.remaining_factor > optixLaunchParams.energy_thres &&
               prd.recursion_depth >= 0 &&
               prd.recursion_depth < optixLaunchParams.max_bounces)
        {
            gdt::vec3f rayOrigin(prd.prev_position.x, prd.prev_position.y, prd.prev_position.z);
            gdt::vec3f rayDir(prd.direction.x, prd.direction.y, prd.direction.z);
            optixTrace(optixLaunchParams.traversable,
                       rayOrigin,
                       rayDir,
                       0.f,   // tmin
                       1e20f, // tmax
                       0.0f,  // rayTime
                       OptixVisibilityMask(255),
                       OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                       SURFACE_RAY_TYPE,              // SBT offset
                       RAY_TYPE_COUNT,                // SBT stride
                       SURFACE_RAY_TYPE,              // missSBTIndex
                       u0, u1);
        }
    }
}