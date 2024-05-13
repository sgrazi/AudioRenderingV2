# AudioRenderingV2

Audio simulation and auralization application. Impulse response is simulated using raytracing through CUDA

## How to build

### Requirements

- Have CUDA v12.1+ installed
- Have OptiX SDK v7.7.0+
- CMake v3.26.0+

### Steps

1. Go to the root folder
2. `mkdir build`
3. `cd build && cmake.exe ..\prebuild\`
4. Open the proyect on visual studio `optix.sln`
5. In visual studio build solution `obj_raytracer`
