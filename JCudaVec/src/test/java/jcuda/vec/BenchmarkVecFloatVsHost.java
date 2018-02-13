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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import jcuda.Pointer;

/**
 * A simple benchmark comparing a the VecFloat operations to host operations 
 */
public class BenchmarkVecFloatVsHost
{
    /**
     * Entry point
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        final int n = 10000000;
        
        AbstractCoreFloat simpleArithmeticCore = new AbstractCoreFloat(
            "simple arithmetic")
        {
            @Override
            protected float computeHostElement(float x, float y, float scalar)
            {
                return x*y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.mul(handle, n, result, x, y);
            }
        };
        
        AbstractCoreFloat simpleComparisonCore = new AbstractCoreFloat(
            "simple comparison")
        {
            @Override
            protected float computeHostElement(float x, float y, float scalar)
            {
                return x<y?1.0f:0.0f;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.lt(handle, n, result, x, y);
            }
        };

        AbstractCoreFloat simpleMathCore = new AbstractCoreFloat(
            "simple math")
        {
            @Override
            protected float computeHostElement(float x, float y, float scalar)
            {
                return (float)Math.pow(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.pow(handle, n, result, x, y);
            }
        };
        
        AbstractCoreFloat complexMathCore = new AbstractCoreFloat(
            "complex math")
        {
            @Override
            protected float computeHostElement(float x, float y, float scalar)
            {
                float result = 0;
                x = (float)Math.cos(x);
                y = (float)Math.pow(y, x);
                result = x + y;
                result = (float)Math.sin(result);
                return result;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                // x = cos(x);
                VecFloat.cos(handle, n, x, y);
                
                // y = pow(y, x);
                VecFloat.pow(handle, n, y, y, x);

                //result = x + y;
                VecFloat.add(handle, n, result, x, y);
                
                // result = sin(result);
                VecFloat.sin(handle, n, result, result);
            }
        };
        
        List<AbstractCoreFloat> cores = Arrays.asList(
            simpleArithmeticCore,
            simpleComparisonCore,
            simpleMathCore,
            complexMathCore
        );
        List<Timing> timings = benchmark(n, cores);
        Timing.print(timings);
    }

    /**
     * Run each of the given cores with input data of the given size, 
     * and collect the timing information
     * 
     * @param n The input size
     * @param cores The cores
     * @return The timing information
     */
    private static List<Timing> benchmark(
        int n, Iterable<? extends AbstractCoreFloat> cores)
    {
        List<Timing> timings = new ArrayList<Timing>();
        for (AbstractCoreFloat core : cores)
        {
            Timing timing = benchmark(n, core);
            timings.add(timing);
        }
        return timings;
    }
    
    /**
     * Perform a simple benchmark with the given core, and return  
     * the timing information
     * 
     * @param n The input size
     * @param core The core
     * @return The timing information
     */
    private static Timing benchmark(int n, AbstractCoreFloat core)
    {
        System.out.println("Benchmarking "+core+" with "+n+" elements");
        
        final Random random = new Random(0);
        final float hostX[] = TestUtil.createRandomHostDataFloat(n, random);
        final float hostY[] = TestUtil.createRandomHostDataFloat(n, random);
        final float hostResultReference[] = new float[n];
        final float scalar = 0.5f;
        
        Timing timing = new Timing(n+" elements, "+core.toString());

        timing.startHost();
        core.computeHost(n, hostResultReference, hostX, hostY, scalar);
        timing.endHost();
        
        VecHandle handle = Vec.createHandle();
        
        timing.startDeviceTotal();
        Pointer deviceX = TestUtil.createDevicePointerFloat(hostX);
        Pointer deviceY = TestUtil.createDevicePointerFloat(hostY);
        Pointer deviceResult = TestUtil.createDevicePointerFloat(n);
        cuCtxSynchronize();
        
        timing.startDeviceCore();
        core.computeDevice(handle, n, deviceResult, deviceX, deviceY, scalar);
        cuCtxSynchronize();
        timing.endDeviceCore();
        
        @SuppressWarnings("unused")
        float hostResult[] = TestUtil.createHostDataFloat(deviceResult, n);
        
        TestUtil.freeDevicePointer(deviceX);
        TestUtil.freeDevicePointer(deviceY);
        TestUtil.freeDevicePointer(deviceResult);
        cuCtxSynchronize();
        timing.endDeviceTotal();
        
        Vec.destroyHandle(handle);
        
        return timing;
        
    }

}
