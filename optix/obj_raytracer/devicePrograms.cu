#include <stdio.h>
#include <optix_device.h>
#include "LaunchParams.h"
#include "PRD.h"

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

    // compute normal:
    const int primID = optixGetPrimitiveIndex();
    const vec3i index = sbtData.index[primID];
    const vec3f &A = sbtData.vertex[index.x];
    const vec3f &B = sbtData.vertex[index.y];
    const vec3f &C = sbtData.vertex[index.z];
    const vec3f Ng = normalize(cross(B - A, C - A));

    const vec3f rayDir = optixGetWorldRayDirection();
    const float cosDN = 0.2f + .8f * fabsf(dot(rayDir, Ng));
    PRD &prd = *(PRD *)getPRD<PRD>();

    prd = cosDN * sbtData.color;

    switch (sbtData.mat)
    {
    case MAT.DEFAULT:
        // receptor
        printf("hit\n"); //para ver si lo de los mats funciona
        const vec3f dist_vec = sbtData.pos - prd.position;
        const float distance = fabs(dot(dist_vec, prd.direction));
		prd.distance += distance;
        float energy = 1;
    default;
        // material
        printf("hit mat, no deberia de llamarse\n");
        prd.energy = prd.energy * 0; // el int seria un acoustic absorption
    }
}

extern "C" __global__ void __anyhit__radiance()
{ /*! TO DO probably */
}

extern "C" __global__ void __miss__radiance()
{
    PRD &prd = *(PRD *)getPRD<PRD>();
    prd.recursion_depth = -1;
}

extern "C" __global__ void __raygen__renderFrame()
{
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const auto &camera = optixLaunchParams.camera;

    // the values we store the PRD pointer in:
    // Nota: Payload Reference Data and represents the data structure used to pass information between shaders during the ray tracing process
    PRD prd;
    uint32_t u0, u1;
    packPointer(&prd, u0, u1);
    prd.energy = 1.0f;
    prd.distance = 0;
    prd.position = optixLaunchParams.pos;
    prd.recursion_depth = 0;

    // normalized screen plane position, in [0,1]^2
    const vec2f screen(vec2f(ix + .5f, iy + .5f) / vec2f(optixLaunchParams.frame.size));

    // generate ray direction

    prd.direction = normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);
    i = 0;
    // pack data into payload
    while (prd.distance < optixLaunchParams.dist_thres &&
           prd.energy > optixLaunchParams.energy_thres &&
           prd.recursion_depth >= 0 &&
           i < 10000) //por las dudas le pongo un tope
    {
        i++;
        optixTrace(optixLaunchParams.traversable,
                   camera.position,
                   prd.direction,
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

    const int r = int(255.99f * pixelColorPRD.x);
    const int g = int(255.99f * pixelColorPRD.y);
    const int b = int(255.99f * pixelColorPRD.z);

    // convert to 32-bit rgba value (we explicitly set alpha to 0xff
    // to make stb_image_write happy ...
    const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);

    // and write to frame buffer ...
    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}