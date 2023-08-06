#pragma once
#ifndef __KERNELS__
#define __KERNELS__
/*CUDA Includes*/
#include <cuda_runtime.h>

namespace kernels {
	void fillWithZeroesKernel(float* buf);
}
#endif