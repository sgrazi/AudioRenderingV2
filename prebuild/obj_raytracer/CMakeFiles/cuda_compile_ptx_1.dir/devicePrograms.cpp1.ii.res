-D__CUDA_ARCH__=520 -D__CUDA_ARCH_LIST__=520 -nologo -E -TP  -DCUDA_DOUBLE_MATH_FUNCTIONS -EHsc -D__CUDACC__ -D__NVCC__ -D__CUDACC_RDC__  -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include" -I"C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0/include" -I"../common/gdt" -I"../common/3rdParty/glfw/include" -I"../common" "-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/../include"    -D "NOMINMAX" -D "__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__=1" -D "NVCC" -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=1 -D__CUDACC_VER_BUILD__=66 -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=1 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -FI "cuda_runtime.h" "..\obj_raytracer\devicePrograms.cu" 