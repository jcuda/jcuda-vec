/*
 * JCudaVec - Vector operations for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2013-2018 Marco Hutter - http://www.jcuda.org
 * 
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
package jcuda.vec;

import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;
import static jcuda.driver.JCudaDriver.cuModuleUnload;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

/**
 * A class summarizing the functionality of a reduction kernel
 */
class VecReduction
{
    /**
     * The maximum grid dimension that may be used for kernel launches
     */
    private static final int MAX_GRID_DIM = 512;
    
    /**
     * The CUDA context for which this instance has been created
     */
    private final CUcontext context;
    
    /**
     * The module that was created from the PTX file
     */
    private final CUmodule module;
    
    /**
     * The actual reduction function
     */
    private final CUfunction function;
    
    /**
     * The size of one element that this reduction operates on. This 
     * may be Sizeof.FLOAT or Sizeof.DOUBLE
     */
    private final int elementSize;

    /**
     * The maximum block size that may be used for kernel launches
     */
    private final int maxBlockDim; 
    
    /**
     * Temporary memory for the device output
     */
    private CUdeviceptr deviceBuffer;
    
    /**
     * Creates a new instance
     * 
     * @param context The context for which this instance is created
     * @param elementSize The element size, Sizeof.FLOAT or Sizeof.DOUBLE
     * @param dataType The type for the kernels, either "float" or "double"
     * @param reductionOperatorName The name of the reduction operator. 
     * This may be "add", "mul", "min" or "max"
     */
    VecReduction(CUcontext context,
        int elementSize, String dataType, String reductionOperatorName)
    {
        this.context = context;
        this.elementSize = elementSize;
        
        VecUtils.checkResultDriver(cuCtxSetCurrent(context));
        
        // Load the PTX file for the given data type, reduction operator
        // and target architecture
        String ptxFileNamePrefix = "/kernels/JCudaVec_kernel_reduction_"
            + dataType + "_" + reductionOperatorName;
        String ptxFileName = VecUtils.createPtxFileName(ptxFileNamePrefix);
        
        //System.out.println("Loading "+ptxFileName);
        byte ptxData[] = VecUtils.loadData(ptxFileName);
        
        // Create the CUDA module from the PTX file
        this.module = new CUmodule();
        VecUtils.checkResultDriver(
            cuModuleLoadDataEx(module, Pointer.to(ptxData), 
                0, new int[0], Pointer.to(new int[0])));
        
        // Obtain the reduction function 
        String functionName = "reduce_" + reductionOperatorName;
        function = new CUfunction();
        VecUtils.checkResultDriver(cuModuleGetFunction(
            function, module, functionName));
        
        // Allocate a chunk of temporary memory that will be used during the 
        // reduction. This must be at least numberOfBlocks * elementSize bytes
        deviceBuffer = new CUdeviceptr();
        cuMemAlloc(deviceBuffer, MAX_GRID_DIM * elementSize);
        
        maxBlockDim = VecUtils.getMaxBlockDimX();
    }
    
    /**
     * Performs a reduction on the given device memory with the given
     * number of elements.
     * 
     * @param numElements The number of elements to reduce
     * @param deviceInput The device input memory
     * @param hostResult The pointer that will store the result
     */
    void reduce(long numElements, Pointer deviceInput, Pointer hostResult)
    {
        reduce(numElements, deviceInput, maxBlockDim, MAX_GRID_DIM);
        cuMemcpyDtoH(hostResult, deviceBuffer, elementSize);     
    }
    
    
    /**
     * Performs a reduction on the given device memory with the given
     * number of elements and the specified limits for threads and
     * blocks.<br>
     * <br>
     * The result of the reduction will be stored in {@link #deviceBuffer} 
     * at index 0.
     * @param numElements The number of elements to reduce
     * @param deviceInput The device input memory
     * @param maxThreads The maximum number of threads
     * @param maxBlocks The maximum number of blocks
     */
    private void reduce(
        long numElements, Pointer deviceInput, 
        int maxThreads, int maxBlocks)
    {
        // Determine the number of threads and blocks 
        // (as done in the NVIDIA sample)
        int numBlocks = getNumBlocks(numElements, maxBlocks, maxThreads);
        int numThreads = getNumThreads(numElements, maxBlocks, maxThreads);
        
        // Call the main reduction method
        reduce(numElements, numThreads, numBlocks, 
            maxThreads, maxBlocks, deviceInput);
    }
    

    
    /**
     * Performs a reduction on the given device memory. <br>
     * <br>
     * The result of the reduction will be stored in {@link #deviceBuffer} 
     * at index 0.
     * 
     * @param numElements The number of elements for the reduction
     * @param numThreads The number of threads
     * @param numBlocks The number of blocks
     * @param maxThreads The maximum number of threads
     * @param maxBlocks The maximum number of blocks
     * @param deviceInput The input memory
     */
    private void reduce(
        long numElements, int numThreads, int numBlocks,
        int maxThreads, int maxBlocks, Pointer deviceInput)
    {
        // Perform a "tree like" reduction as in the NVIDIA sample
        reduce(numElements, numThreads, numBlocks, deviceInput, deviceBuffer);
        int s = numBlocks;
        while(s > 1) 
        {
            int threads = getNumThreads(s, maxBlocks, maxThreads);
            int blocks = getNumBlocks(s, maxBlocks, maxThreads);

            reduce(s, threads, blocks, deviceBuffer, deviceBuffer);
            s = (s + (threads * 2 - 1)) / (threads * 2);
        }
    }
    
    
    /**
     * Perform a reduction of the specified number of elements in the given 
     * device input memory, using the given number of threads and blocks, 
     * and write the results into the given output memory. 
     * 
     * @param numElements The number of elements 
     * @param threads The number of threads
     * @param blocks The number of blocks
     * @param deviceInput The device input memory
     * @param deviceOutput The device output memory. Its size must at least 
     * be numBlocks*elementSize
     */
    private void reduce(long numElements, int threads, int blocks, 
        Pointer deviceInput, Pointer deviceOutput)
    {
        // Compute the shared memory size (as done in the NIVIDA sample)
        int sharedMemSize = threads * elementSize;
        if (threads <= 32) 
        {
            sharedMemSize *= 2;
        }
        
        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new long[]{numElements}),
            Pointer.to(deviceInput),
            Pointer.to(deviceOutput)
        );

        // Call the kernel function.
        cuLaunchKernel(function,
            blocks,  1, 1,                  // Grid dimension
            threads, 1, 1,                  // Block dimension
            sharedMemSize, Vec.getStream(), // Shared memory size and stream
            kernelParameters, null          // Kernel- and extra parameters
        );
        cuCtxSynchronize();
    }
    
    /**
     * Perform a shutdown, releasing all resources that have been
     * allocated by this instance.
     */
    void shutdown()
    {
        VecUtils.checkResultDriver(cuCtxSetCurrent(context));
        VecUtils.checkResultDriver(cuModuleUnload(module));
        VecUtils.checkResultDriver(cuMemFree(deviceBuffer));
    }
    
    /**
     * Compute the number of blocks that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of blocks
     */
    private static int getNumBlocks(long n, int maxBlocks, int maxThreads)
    {
        int threads = getNumThreads(n, maxBlocks, maxThreads);
        long blocks = (n + (threads * 2 - 1)) / (threads * 2);
        return (int)Math.min(maxBlocks, blocks);
    }

    /**
     * Compute the number of threads that should be used for the
     * given input size and limits
     * 
     * @param n The input size
     * @param maxBlocks The maximum number of blocks
     * @param maxThreads The maximum number of threads
     * @return The number of threads
     */
    private static int getNumThreads(long n, int maxBlocks, int maxThreads)
    {
        if (n < maxThreads * 2) 
        {
            return (int)nextPow2((n + 1) / 2);
        }
        return maxThreads;
    }
    
    /**
     * Returns the power of 2 that is equal to or greater than x
     * 
     * @param x The input
     * @return The next power of 2
     */
    private static long nextPow2(long x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        x |= x >> 32;
        return ++x;
    }

}
