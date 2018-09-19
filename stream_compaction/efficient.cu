#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#define blockSize 128
//toggle between opimization threads and not (for EC1 or part 5)
#define OPT 1

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernDataCopy(int n, int* odata, const int* idata)
		{
			int index = threadIdx.x + (blockDim.x * blockIdx.x);
			if (index >= n)
				return;
			odata[index] = idata[index];
		}

		__global__ void kernDataSet(int n, int* idata, int value)
		{
			int index = threadIdx.x + (blockDim.x * blockIdx.x);
			if (index >= n)
				return;
			idata[index] = value;
		}
		__global__ void kernRootSet(int n, int* idata)
		{
			idata[n-1] = 0;
		}
		__global__ void kernEfficientUpsweep(int n, int* idata, int d)
		{
			int index = threadIdx.x + (blockDim.x * blockIdx.x);
			if (index >= n)
			{
				return;
			}
#if OPT
			//With optimization : only launch active threads
			//Basically hack on the indices
			int targetInd = (index + 1) * (1 << (d + 1)) - 1; //for example: index = 0,1,2,3 targetInd = 1,3,5,7 (d = 0)
			idata[targetInd] += idata[targetInd - (1 << d)];
			//With optimization - end
#else
			//No optimization 
			if ((index + 1) % (1 << (d + 1)) == 0)
			{
				idata[index] = idata[index] + idata[index - (1 << d)];
			}	
			//No optimization - end
#endif
			

		}
		__global__ void kernEfficientDownsweep(int n, int* idata, int d)
		{
			int index = threadIdx.x + (blockDim.x * blockIdx.x);
			if (index >= n)
			{
				return;
			}
#if OPT
			//With optimization : only launch active threads
			int targetInd = (index + 1) * (1 << (d + 1)) - 1;
			int t = idata[targetInd - (1 << d)];
			idata[targetInd - (1 << d)] = idata[targetInd];
			idata[targetInd] += t;
			//With optimization - end
#else
			//No optimization
			if ((index + 1) % (1 << (d + 1)) == 0)
			{
				int t = idata[index - (1 << d)];
				idata[index - (1 << d)] = idata[index];
				idata[index] += t;
			}
			//No optimization - end
#endif

		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			//allocate and copy memories on CUDA device
			//need to round n to the next power of 2
			int deg = ilog2ceil(n);
			int nround2 = 1 << deg;
			int* dev_idata;
			cudaMalloc((void**)&dev_idata, sizeof(int) * nround2);
			checkCUDAError("cudamalloc failed!");
			cudaDeviceSynchronize();

			//set the extra memory to zeros
			dim3 fullBlocksPerGrid = (nround2 + blockSize - 1) / blockSize;
			kernDataSet << <fullBlocksPerGrid, blockSize >>> (nround2, dev_idata, 0);
			checkCUDAError("kerndataset failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudamemcpy failed!");
            timer().startGpuTimer();
            // TODO		
#if OPT
			//With optimization
			for (int d = 0; d <= deg - 1; d++)
			{
				int threadCount = 1 << (deg - 1 - d);
				kernEfficientUpsweep << <dim3((threadCount + blockSize - 1) / blockSize), blockSize >> >(threadCount, dev_idata, d);
			}
			checkCUDAError("kernEffcientUpSweep failed!");
			kernRootSet << <dim3(1), dim3(1) >> > (nround2, dev_idata);
			checkCUDAError("kernRootSet failed!");
			for (int d = deg - 1; d >= 0; d--)
			{
				int threadCount = 1 << (deg - 1 - d);
				kernEfficientDownsweep << <dim3((threadCount + blockSize - 1) / blockSize), blockSize >> >(threadCount, dev_idata, d);
			}
			checkCUDAError("kernEffcientDownSweep failed!");
			//With optimization - end
#else
			//No optimization: always launching same amount of threads
			for (int d = 0; d <= deg - 1; d++)
			{
				kernEfficientUpsweep << <fullBlocksPerGrid, blockSize >> >(nround2, dev_idata, d);
				
			}
			checkCUDAError("kernEffcientUpSweep failed!");
			kernRootSet << <dim3(1),dim3(1) >>> (nround2, dev_idata);
			checkCUDAError("kernRootSet failed!");
			for (int d = deg - 1; d >= 0; d--)
			{

				kernEfficientDownsweep << <fullBlocksPerGrid, blockSize >> >(nround2, dev_idata, d);
			}
			checkCUDAError("kernEffcientDownSweep failed!");
			//No optimization - end
#endif
            timer().endGpuTimer();
			cudaMemcpy(odata, dev_idata, sizeof(int)*n, cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
			int nround2 = 1 << (ilog2ceil(n));
			int deg = ilog2ceil(n);
			dim3 fullBlocksPerGrid = (nround2 + blockSize - 1) / blockSize;
			dim3 originGrid = (n + blockSize - 1) / blockSize;
			int* dev_odata, *dev_idata, *dev_bools, *dev_scanned;
			cudaMalloc((void**)&dev_idata, sizeof(int) * n);
			cudaMalloc((void**)&dev_bools, sizeof(int) * n);
			cudaMalloc((void**)&dev_scanned, sizeof(int) * nround2);
			cudaMalloc((void**)&dev_odata, sizeof(int) * n);
			cudaDeviceSynchronize();
			//set the extra memory to zeros
			
			kernDataSet << <fullBlocksPerGrid, blockSize >> > (nround2, dev_scanned, 0);
			checkCUDAError("kerndataset failed!");
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			
			//dim3 fullBlocksPerGrid2 = (n + blockSize - 1) / blockSize;
			int* dev_count;
			cudaMalloc((void**)&dev_count, sizeof(int));
			int count;
			cudaDeviceSynchronize();

            timer().startGpuTimer();
            // TODO
			//step1: map to bools
			Common::kernMapToBoolean << <originGrid, blockSize >> >(n, dev_bools, dev_idata);
			
			//kernDataCopy << <fullBlocksPerGrid, blockSize >> >(nround2, dev_scanned, dev_bools);
			cudaMemcpy(dev_scanned, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
			//step2: scan
#if OPT
			//With optimization
			for (int d = 0; d <= deg - 1; d++)
			{
				int threadCount = 1 << (deg - 1 - d);
				kernEfficientUpsweep << <dim3((threadCount + blockSize - 1) / blockSize), blockSize >> >(threadCount, dev_scanned, d);
			}
			checkCUDAError("kernEffcientUpSweep failed!");
			kernRootSet << <dim3(1), dim3(1) >> > (nround2, dev_scanned);
			checkCUDAError("kernRootSet failed!");
			for (int d = deg - 1; d >= 0; d--)
			{
				int threadCount = 1 << (deg - 1 - d);
				kernEfficientDownsweep << <dim3((threadCount + blockSize - 1) / blockSize), blockSize >> >(threadCount, dev_scanned, d);
			}
			checkCUDAError("kernEffcientDownSweep failed!");
			//With optimization - end
#else
			//No optimization: always launching same amount of threads
			for (int d = 0; d <= deg - 1; d++)
			{
				kernEfficientUpsweep << <fullBlocksPerGrid, blockSize >> >(nround2, dev_scanned, d);

			}
			checkCUDAError("kernEffcientUpSweep failed!");
			kernRootSet << <dim3(1), dim3(1) >> > (nround2, dev_scanned);
			checkCUDAError("kernRootSet failed!");
			for (int d = deg - 1; d >= 0; d--)
			{

				kernEfficientDownsweep << <fullBlocksPerGrid, blockSize >> >(nround2, dev_scanned, d);
			}
			checkCUDAError("kernEffcientDownSweep failed!");
			//No optimization - end
#endif

			//step3: scatter
			
			Common::kernScatter<<<originGrid,blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_scanned);
			
            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, sizeof(int)*n, cudaMemcpyDeviceToHost);
			
			cudaMemcpy(&count, dev_scanned + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
		
			if (idata[n - 1] != 0)
				count++;
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_scanned);
            return count;
        }
    }
}
