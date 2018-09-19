#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
			odata[0] = 0;
			for (int ind = 1; ind < n; ind++)
			{
				odata[ind] = odata[ind - 1] + idata[ind - 1];
			}
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
			int nonZeros = 0;
			for (int ind = 0; ind < n; ind++)
			{
				if (idata[ind] != 0)
				{
					odata[nonZeros] = idata[ind];
					nonZeros++;
				}
			}
	        timer().endCpuTimer();
            return nonZeros;
        }

		/*
		 * CPU scatter function
		 */
		void cpuScatter(int n, int* odata, const int* idata, int* indicator, int* scanned)
		{
			for (int ind = 0; ind < n; ind++)
			{
				if (indicator[ind] != 0)
				{
					odata[scanned[ind]] = idata[ind];
				}
			}
		}
        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			int* indicator= new int[n];
			int* scanned = new int[n];
	        timer().startCpuTimer();
	        // TODO
			for (int ind = 0; ind < n; ind++)
			{
				if (idata[ind] == 0)
					indicator[ind] = 0;
				else
					indicator[ind] = 1;
			}
			scanned[0] = 0;
			for (int ind = 1; ind < n; ind++)
			{
				scanned[ind] = scanned[ind - 1] + indicator[ind - 1];
			}
			cpuScatter(n, odata, idata, indicator, scanned);
	        timer().endCpuTimer();
			
			int num = scanned[n - 1];
			if (indicator[n - 1] != 0)
				num++;
			delete[] indicator;
			delete[] scanned;
            return num;
        }
    }
}
