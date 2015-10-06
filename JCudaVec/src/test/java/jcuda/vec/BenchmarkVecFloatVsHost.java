/*
 * JCudaVec - Vector operations for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2013-2015 Marco Hutter - http://www.jcuda.org
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;

/**
 * A simple benchmark comparing a the VecFloat operations to host operations 
 */
public class BenchmarkVecFloatVsHost
{
    public static void main(String[] args)
    {
        VecFloat.init();
        
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
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.mul(n, result, x, y);
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
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.lt(n, result, x, y);
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
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.pow(n, result, x, y);
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
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                // x = cos(x);
                VecFloat.cos(n, x, y);
                
                // y = pow(y, x);
                VecFloat.pow(n, y, y, x);

                //result = x + y;
                VecFloat.add(n, result, x, y);
                
                // result = sin(result);
                VecFloat.sin(n, result, result);
            }
        };
        
        List<AbstractCoreFloat> cores = Arrays.asList(
            simpleArithmeticCore,
            simpleComparisonCore,
            simpleMathCore,
            complexMathCore
        );
        List<Timing> timings = benchmark(n, cores);
        print(timings);
        
        VecFloat.shutdown();
    }

    private static void print(List<Timing> timings)
    {
        System.out.println(createString(timings));
    }
    
    private static String createString(List<Timing> timings)
    {
        int maxNameLength = -1;
        for (Timing timing : timings)
        {
            maxNameLength = Math.max(maxNameLength, timing.getName().length());
        }
        int columnWidth = 10;
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-"+maxNameLength+"s ", "Name"));
        sb.append(String.format("%"+columnWidth+"s", "Host"));
        sb.append(String.format("%"+columnWidth+"s", "Dev."));
        sb.append(String.format("%"+columnWidth+"s", "Dev.Tot."));
        sb.append("\n");
        for (Timing timing : timings)
        {
            sb.append(timing.createString(maxNameLength));
            sb.append("\n");
        }
        return sb.toString();
    }
    
    private static List<Timing> benchmark(int n, List<AbstractCoreFloat> cores)
    {
        List<Timing> timings = new ArrayList<Timing>();
        for (AbstractCoreFloat core : cores)
        {
            Timing timing = benchmark(n, core);
            timings.add(timing);
        }
        return timings;
    }
    
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
        
        timing.startDeviceTotal();
        CUdeviceptr deviceX = TestUtil.createDevicePointerFloat(hostX);
        CUdeviceptr deviceY = TestUtil.createDevicePointerFloat(hostY);
        CUdeviceptr deviceResult = TestUtil.createDevicePointerFloat(n);
        cuCtxSynchronize();
        
        timing.startDeviceCore();
        core.computeDevice(n, deviceResult, deviceX, deviceY, scalar);
        cuCtxSynchronize();
        timing.endDeviceCore();
        
        @SuppressWarnings("unused")
        float hostResult[] = TestUtil.createHostDataFloat(deviceResult, n);
        
        TestUtil.freeDevicePointer(deviceX);
        TestUtil.freeDevicePointer(deviceY);
        TestUtil.freeDevicePointer(deviceResult);
        cuCtxSynchronize();
        timing.endDeviceTotal();
        
        return timing;
        
    }

}
