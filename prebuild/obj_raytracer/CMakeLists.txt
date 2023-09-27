# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)

project(obj_raytracer LANGUAGES CXX CUDA)

include_directories(${OptiX_INCLUDE})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/common/glad)

# Add rtaudio as a sub-directory so its CMakeLists.txt gets invoked.
#add_subdirectory(external/rtaudio)

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(obj_raytracer
  ${embedded_ptx_code}
  optix7.h
  CUDABuffer.h
  LaunchParams.h
  AudioRenderer.h
  AudioRenderer.cpp
  OptixModel.h
  OptixModel.cpp
  Camera.h
  Camera.cpp
  Mesh.h
  tiny_obj_loader.h
  Mesh.cpp
  OBJ_Loader.h
  RtAudio.h
  Utils.h
  main.cpp
  shaderClass.h
  shaderClass.cpp
  VAO.h
  VAO.cpp
  VBO.h
  VBO.cpp
  EBO.h
  EBO.cpp
  Utils.h
  Utils.cpp
  kernels.cu
  kernels.cuh
  tinyxml2.h
  ${CMAKE_SOURCE_DIR}/common/glad/glad.h
  ${CMAKE_SOURCE_DIR}/common/glad/glad.c
  )

target_link_libraries(obj_raytracer
  gdt
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  # glfw and opengl, for display
  glfWindow
  glfw
  ${OPENGL_gl_LIBRARY}
  rtaudio
  )