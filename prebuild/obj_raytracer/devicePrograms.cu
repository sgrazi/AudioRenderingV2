#include <stdio.h>
#include <optix_device.h>
#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <gdt/math/vec.h>
#include <glm/glm.hpp>
#include <math.h>
#include "LaunchParams.h"
#include "PRD.h"
#include "curand_kernel.h"

#define SPEED_OF_SOUND 343 // grabbed from Cameelo/AudioRendering
#define CUDART_PI_F 3.141592654f
#define HRTF_HEAD_ABSORPTION 0.9f

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
    const glm::vec3 Ng = glm::normalize(glm::cross(U, V));

    const float u_barycentrics = optixGetTriangleBarycentrics().x;
    const float v_barycentrics = optixGetTriangleBarycentrics().y;
    const glm::vec3 P = (1.f - u_barycentrics - v_barycentrics) * P1 + u_barycentrics * P2 + v_barycentrics * P3;

    prd.distance += distance(P, prd.prev_position);

    // Get Ray Id
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const int iz = optixGetLaunchIndex().z;
    float *ir_right = optixLaunchParams.ir_right;
    float *ir_left = optixLaunchParams.ir_left;
    if (sbtData.mat_absorption < 0)
    {
        // Sphere radius hardcoded to 1m
        float sphere_radius = 1;
        glm::vec3 normDirection = rayDir * (1.0f / sqrt(glm::dot(rayDir, rayDir)));

        glm::vec3 oc = P - prd.sphere_center;
        float a = glm::dot(normDirection, normDirection);
        float b = 2.0 * glm::dot(oc, normDirection);
        float c = glm::dot(oc, oc) - sphere_radius * sphere_radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) {
            prd.remaining_factor = 0;
        }
        else {
            if (discriminant == 0) {
                prd.remaining_factor = 0;
            }
            else {
                float sqrtDiscriminant = sqrt(discriminant);
                float t1 = (-b - sqrtDiscriminant) / (2 * a);
                float t2 = (-b + sqrtDiscriminant) / (2 * a);

                glm::vec3 intersection1 = P + t1 * normDirection;
                glm::vec3 intersection2 = P + t2 * normDirection;

                // Two intersection points.
                prd.remaining_factor *= sqrt(glm::dot((intersection1- intersection2), (intersection1 - intersection2)));
            }
        }
    }
    
    // Average head breadth	is 15.5cm so we delay signal to the other ear and we lower its impact
    int delay = optixLaunchParams.sample_rate * 0.00044; // 0.00044 seconds for sound to travel 15.5cm
    float hrtf_absorption_rate = optixLaunchParams.hrtf_absorption_rate;

    if (sbtData.mat_absorption == -1)
    {
        // we identify the receiver with a negative absorption
        float elapsed_time = prd.distance / SPEED_OF_SOUND;
        int array_pos = round(elapsed_time * optixLaunchParams.sample_rate);
        if (array_pos < optixLaunchParams.ir_length)
        {
            atomicAdd(&ir_left[array_pos], prd.remaining_factor);
            if (!optixLaunchParams.isMono) {
                if (array_pos + delay < optixLaunchParams.ir_length)
                {
                    atomicAdd(&ir_right[array_pos + delay], prd.remaining_factor * (1- hrtf_absorption_rate));
                }
                else
                {
                    atomicAdd(&ir_right[array_pos], prd.remaining_factor * (1- hrtf_absorption_rate));
                }
            }
        }
        prd.recursion_depth = -1;
    } 
    else {
        if (sbtData.mat_absorption == -2)
        {
            // we identify the receiver with a negative absorption
            float elapsed_time = prd.distance / SPEED_OF_SOUND;
            int array_pos = round(elapsed_time * optixLaunchParams.sample_rate);
            if (array_pos < optixLaunchParams.ir_length)
            {
                atomicAdd(&ir_right[array_pos], prd.remaining_factor);
                if (!optixLaunchParams.isMono) {
                    if (array_pos + delay < optixLaunchParams.ir_length)
                    {
                        atomicAdd(&ir_left[array_pos + delay], prd.remaining_factor * (1- hrtf_absorption_rate));
                    }
                    else
                    {
                        atomicAdd(&ir_left[array_pos], prd.remaining_factor * (1- hrtf_absorption_rate));
                    }
                }
            }
            prd.recursion_depth = -1;
        }
        else
        {
            prd.direction = prd.direction - 2.0f * dot(prd.direction, Ng) * Ng;
            prd.remaining_factor *= (1 - sbtData.mat_absorption);
            prd.recursion_depth++;
        }
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

    // 4.18879020478 is the volume of the sphere
    prd.remaining_factor = (optixLaunchParams.base_power) / ((x_rays * y_rays * z_rays) * 4.18879020478);
    prd.distance = 0;
    prd.prev_position = optixLaunchParams.emitter_position;
    prd.recursion_depth = 0;
    prd.sphere_center = optixLaunchParams.sphere_center;

    int tid = iz * x_rays * y_rays + iy * x_rays + ix;

    curandState state;
    curand_init((unsigned long long)clock64(), tid, 0, &state);

    double theta = 2 * CUDART_PI_F * curand_uniform(&state);
    double phi = acos(2 * curand_uniform(&state) - 1);

    double x = sin(phi) * cos(theta);
    double y = sin(phi) * sin(theta);
    double z = cos(phi);

    // Guarantees 1 < IR_length_in_seconds < 999
    int IR_length_in_seconds = max(1, min(optixLaunchParams.ir_length / optixLaunchParams.sample_rate, 999));
    float distance_threshold = IR_length_in_seconds * SPEED_OF_SOUND + 1;
    // it is bound to happen that some threads have (0,0,0) as their vector
    if (x != 0.0 || y != 0.0 || z != 0.0)
    {
        prd.direction = {x, y, z};
        while (prd.distance < distance_threshold &&
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