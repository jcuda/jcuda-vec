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

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.util.Random;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

/**
 * A simple benchmark comparing a the VecDouble operations to a dedicated
 * kernel that performs the same operation 
 */
public class BenchmarkVecDoubleVsOwnKernel
{
    /**
     * Entry point 
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        JCudaDriver.setExceptionsEnabled(true);
        
        final int runs = 1000;
        for (int n=1000; n<=10000000; n*=10)
        {
            runTest(n, runs);
        }
        
        System.out.println("Done");
    }
    
    /**
     * Run a benchmark with the given number of elements and runs
     * 
     * @param n The number of elements
     * @param runs The number of runs
     */
    private static void runTest(int n, int runs)
    {
        System.out.println("Bencharking " + n + " elements, " + runs + " runs");
        
        // Set up the input data
        final Random random = new Random(0);
        final double hostX[] = TestUtil.createRandomHostDataDouble(n, random);
        Pointer deviceX = TestUtil.createDevicePointerDouble(hostX);

        // Perform the computation using the vector library
        Timing timingVec = new Timing(n + " elements, " + runs + " runs, vec");
        double hostResultVec[] = runVec(n, deviceX, runs, timingVec);

        // Perform the computation using an own kernel
        Timing timingOwn = new Timing(n + " elements, " + runs + " runs, own");
        double hostResultOwn[] = runOwn(n, deviceX, runs, timingOwn);
        
        // Compare the results
        final boolean verbose = true;
        final double epsilon = 1e-14f;
        boolean passed = TestUtil.equalDouble(
            hostResultVec, hostResultOwn, epsilon, verbose);
        
        System.out.println("Passed? " + passed);
        
        System.out.println(timingVec);
        System.out.println(timingOwn);
        
        // Clean up
        TestUtil.freeDevicePointer(deviceX);
        
    }
    
    /**
     * Run the computation using the JCudaVec library
     * 
     * @param n The number of elements
     * @param deviceX The device data
     * @param runs The number of runs
     * @param timingVec The timing information
     * @return The results
     */
    private static double[] runVec(int n, Pointer deviceX,
        int runs, Timing timingVec)
    {
        Pointer deviceResultVec = TestUtil.createDevicePointerDouble(n);
        VecHandle handle = Vec.createHandle();

        timingVec.startDeviceCore();
        for (int i=0; i<runs; i++)
        {
            logistic(handle, n, deviceResultVec, deviceX);
            cuCtxSynchronize();
        }
        timingVec.endDeviceCore();
        
        double hostResultVec[] = 
            TestUtil.createHostDataDouble(deviceResultVec, n);
        TestUtil.freeDevicePointer(deviceResultVec);
        
        Vec.destroyHandle(handle);
        
        return hostResultVec;
    }
    
    /**
     * Implementation of a logistic function using the JCudaVec library
     * 
     * @param handle The {@link VecHandle}
     * @param n The number of elements
     * @param result The result pointer
     * @param x The input pointer
     */
    private static void logistic(
        VecHandle handle, long n, Pointer result, Pointer x)
    {
        // 1 / (1 + e^(-x))
        
        // result = -x
        VecDouble.negate(handle, n, result, x);
        
        // result = e^result
        VecDouble.exp(handle, n, result, result);

        // result = result + 1
        VecDouble.addScalar(handle, n, result, result, 1.0f);
        
        //result = 1 / result
        VecDouble.scalarDiv(handle, n, result, 1.0f, result);
    }
    
    /**
     * Run the computation using a dedicated, own kernel
     * 
     * @param n The number of elements
     * @param deviceX The device data
     * @param runs The number of runs
     * @param timingOwn The timing information
     * @return The results
     */
    private static double[] runOwn(int n, Pointer deviceX,
        int runs, Timing timingOwn)
    {
        Pointer deviceResultOwn = TestUtil.createDevicePointerDouble(n);

        CUmodule module = new CUmodule();
        String moduleFileName = "src/test/resources/kernels/" + 
            "LogisticFunctionKernel_double_64_cc30.ptx";
        cuModuleLoad(module, moduleFileName);
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "logistic");
        int blockSizeX = 128;
        int gridSizeX = (int)Math.ceil((double)n / blockSizeX);

        timingOwn.startDeviceCore();
        for (int i=0; i<runs; i++)
        {
            Pointer kernelParameters = Pointer.to(
                Pointer.to(new long[]{n}),
                Pointer.to(deviceResultOwn),
                Pointer.to(deviceX)
            );
            cuLaunchKernel(function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
        }
        timingOwn.endDeviceCore();
        
        double hostResultOwn[] = 
            TestUtil.createHostDataDouble(deviceResultOwn, n);
        TestUtil.freeDevicePointer(deviceResultOwn);
        return hostResultOwn;
    }
    
}
