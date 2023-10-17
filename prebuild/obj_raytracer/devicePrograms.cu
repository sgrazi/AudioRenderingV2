#include <stdio.h>
#include <optix_device.h>
#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <gdt/math/vec.h>
#include <glm/glm.hpp>
#include "LaunchParams.h"
#include "PRD.h"

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
    const glm::vec3& A = sbtData.vertex[index.x];
    const glm::vec3& B = sbtData.vertex[index.y];
    const glm::vec3& C = sbtData.vertex[index.z];
    const glm::vec3 Ng = normalize(cross(B - A, C - A));
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    glm::vec3 P = (1.f - u - v) * A + u * B + v * C;

    switch (sbtData.mat_absorption < 0) // we identify the receiver with a negative absorption
    {
    case true:
        // printf("HIT RECEIVER at: %f,%f,%f...\n", P.x, P.y, P.z);
        prd.distance += distance(P,prd.prev_position);
        float elapsed_time = prd.distance / SPEED_OF_SOUND;
        int array_pos = round(elapsed_time * optixLaunchParams.sample_rate);
        float *ir = optixLaunchParams.ir;
        if (array_pos < optixLaunchParams.ir_length) {
            ir[array_pos] += prd.remaining_factor;
        }
        break;
    case false:
        //printf("HIT MATERIAL at: %f,%f,%f...\n", P.x, P.y, P.z);
        // material
        prd.direction = prd.direction - 2.0f * (prd.direction * Ng) * Ng;
		float dist_traveled = optixGetRayTmax(); // returns the current path segment distance
        prd.distance += dist_traveled;
        prd.remaining_factor *= (1 - sbtData.mat_absorption);
        prd.recursion_depth++;
        break;
    default:
        // ERROR
    }
    prd.prev_position = P;
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
    prd.remaining_factor = (optixLaunchParams.BASE_POWER)/(x_rays*y_rays*z_rays);
    prd.distance = 0;
    prd.prev_position = optixLaunchParams.emitter_position;
    prd.recursion_depth = 0;

    double dx = (ix * (2.0 / (x_rays - 1)) - 1.0);
    double dy = (iy * (2.0 / (y_rays - 1)) - 1.0);
    double dz = (iz * (2.0 / (z_rays - 1)) - 1.0);
    // it is bound to happen that some threads have (0,0,0) as their vector
    if (dx != 0.0 || dy != 0.0 || dz != 0.0) {
        double length = std::sqrt(dx * dx + dy * dy + dz * dz);
        dx /= length;
        dy /= length;
        dz /= length;

        // printf("sending to %f,%f,%f...\n", dx, dy, dz);
        prd.direction = {dx, dy, dz};
        int i = 0;
        while (prd.distance < optixLaunchParams.dist_thres &&
               prd.remaining_factor > optixLaunchParams.energy_thres &&
               prd.recursion_depth >= 0 &&
               i < 60) // por las dudas le pongo un tope
        {
            i++;
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