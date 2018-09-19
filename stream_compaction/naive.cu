#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#define blockSize 128
namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kernCopyValues(int n, const int* idata, int* temp, bool inclusive)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
			{
				return;
			}
			if (inclusive)
			{
				temp[index] = idata[index];
			}
			else
			{
				if (index == 0)
					temp[index] = 0;
				else
					temp[index] = idata[index - 1];
			}
		}
		__global__ void kernScan(int n, int* odata, int* temp, int d)
		{
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
			{
				return;
			}

			if (index >= 1<<(d-1))
			{
				odata[index] = temp[index - (1 << (d - 1))] + temp[index];
			}
			else
			{
				odata[index] = temp[index];
			}
			__syncthreads();
			temp[index] = odata[index];
			__syncthreads();
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int* dev_idata, *dev_odata;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_odata, odata, n * sizeof(int), cudaMemcpyHostToDevice);

			int *temp;
			cudaMalloc((void**)&temp, n * sizeof(int));
			dim3 fullBlocksPerGrid = (n + blockSize - 1) / blockSize;
			kernCopyValues << <fullBlocksPerGrid, blockSize >> >(n, dev_idata, temp,false);
			cudaThreadSynchronize();
            timer().startGpuTimer();
            // TODO
			for (int d = 1; d <= ilog2ceil(n); d++)
			{
				kernScan << <fullBlocksPerGrid, blockSize >> >(n, dev_odata, temp,d);
			}
            timer().endGpuTimer();
			kernCopyValues << <fullBlocksPerGrid, blockSize >> >(n, temp, dev_odata,true);
			cudaThreadSynchronize();
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(temp);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
