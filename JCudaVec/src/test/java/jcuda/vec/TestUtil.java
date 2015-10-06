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

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxGetCurrent;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import java.util.Random;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;

/**
 * Utility for the JCudaVec tests
 */
public class TestUtil
{
    /**
     * The CUDA context that is created in {@link #init()} and 
     * from which the calling thread is detached in
     * {@link #shutdown()}
     */
    private static CUcontext context;
    
    /**
     * The default device number
     */
    private static int deviceNumber = 0;
    
    /**
     * Initializes the JCuda driver API. Then it will try to attach to the 
     * current CUDA context. If no active CUDA context exists, then it will 
     * try to create one, for the device which is specified by the current 
     * deviceNumber.
     * 
     * @throws CudaException If it is neither possible to 
     * attach to an existing context, nor to create a new
     * context.
     */
    public static void init()
    {
        checkResult(cuInit(0));

        // Try to obtain the current context
        CUcontext context = new CUcontext();
        checkResult(cuCtxGetCurrent(context));
        
        // If the context is 'null', then a new context
        // has to be created.
        CUcontext nullContext = new CUcontext(); 
        if (context.equals(nullContext))
        {
            createContext();
        }
    }
    
    /**
     * Tries to create a context for device 'deviceNumber'.
     * 
     * @throws CudaException If the device can not be 
     * accessed or the context can not be created
     */
    private static void createContext()
    {
        CUdevice device = new CUdevice();
        checkResult(cuDeviceGet(device, deviceNumber));
        CUcontext context = new CUcontext();
        checkResult(cuCtxCreate(context, 0, device));
    }
    
    /**
     * If the given result is not CUresult.CUDA_SUCCESS, then this method
     * throws a CudaException with the error message for the given result.
     * 
     * @param cuResult The result
     * @throws CudaException if the result is not CUresult.CUDA_SUCCESS
     */
    private static void checkResult(int cuResult)
    {
        if (cuResult != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(CUresult.stringFor(cuResult));
        }
    }
    
    /**
     * Shut down and destroy the current CUDA context
     */
    static void shutdown()
    {
        if (context != null)
        {
            cuCtxSynchronize();
            context = null;
        }
    }

    /**
     * Creates an array containing random values between 0 and 1.
     * 
     * @param n The size of the array
     * @param random The random number source
     * @return The array
     */
    static float[] createRandomHostDataFloat(int n, Random random)
    {
        return createRandomHostDataFloat(n, 0.0f, 1.0f, random);
    }
    
    /**
     * Creates an array containing random values between 0 and 1.
     * 
     * @param n The size of the array
     * @param random The random number source
     * @return The array
     */
    static double[] createRandomHostDataDouble(int n, Random random)
    {
        return createRandomHostDataDouble(n, 0.0, 1.0, random);
    }
    
    /**
     * Create an array containing random values in the specified range
     * 
     * @param n The size of the array
     * @param min The minimum value (inclusive)
     * @param max The maximum value (exclusive)
     * @param random The random number source
     * @return The array
     */
    static float[] createRandomHostDataFloat(
        int n, float min, float max, Random random)
    {
        float result[] = new float[n];
        for (int i=0; i<n; i++)
        {
            result[i] = min + random.nextFloat() * (max - min);
        }
        return result;
    }

    /**
     * Create an array containing random values in the specified range
     * 
     * @param n The size of the array
     * @param min The minimum value (inclusive)
     * @param max The maximum value (exclusive)
     * @param random The random number source
     * @return The array
     */
    static double[] createRandomHostDataDouble(
        int n, double min, double max, Random random)
    {
        double result[] = new double[n];
        for (int i=0; i<n; i++)
        {
            result[i] = min + random.nextDouble() * (max - min);
        }
        return result;
    }

    /**
     * Create a device pointer that points to a data block with the given
     * number of elements
     * 
     * @param n The number of elements
     * @return The new pointer
     */
    static CUdeviceptr createDevicePointerFloat(int n)
    {
        return createDevicePointerFloat(new float[n]);
    }

    /**
     * Create a device pointer that points to a data block with the given
     * number of elements
     * 
     * @param n The number of elements
     * @return The new pointer
     */
    static CUdeviceptr createDevicePointerDouble(int n)
    {
        return createDevicePointerDouble(new double[n]);
    }

    /**
     * Create a device pointer that points to a memory block with the
     * given data
     * 
     * @param hostData The host data
     * @return The device pointer
     */
    static CUdeviceptr createDevicePointerFloat(float hostData[])
    {
        CUdeviceptr devicePointer = new CUdeviceptr();
        cuMemAlloc(devicePointer, hostData.length * Sizeof.FLOAT);
        cuMemcpyHtoD(devicePointer, Pointer.to(hostData),
            hostData.length * Sizeof.FLOAT);
        return devicePointer;
    }

    /**
     * Create a device pointer that points to a memory block with the
     * given data
     * 
     * @param hostData The host data
     * @return The device pointer
     */
    static CUdeviceptr createDevicePointerDouble(double hostData[])
    {
        CUdeviceptr devicePointer = new CUdeviceptr();
        cuMemAlloc(devicePointer, hostData.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(devicePointer, Pointer.to(hostData),
            hostData.length * Sizeof.DOUBLE);
        return devicePointer;
    }

    /**
     * Copy the data from the given device pointer into an array and return it
     * 
     * @param devicePointer The device pointer
     * @param n The size of the data block, in number of elements 
     * @return The array
     */
    static float[] createHostDataFloat(CUdeviceptr devicePointer, int n)
    {
        float hostData[] = new float[n];
        cuMemcpyDtoH(Pointer.to(hostData), devicePointer, n * Sizeof.FLOAT);
        return hostData;
    }
    
    /**
     * Copy the data from the given device pointer into an array and return it
     * 
     * @param devicePointer The device pointer
     * @param n The size of the data block, in number of elements 
     * @return The array
     */
    static double[] createHostDataDouble(CUdeviceptr devicePointer, int n)
    {
        double hostData[] = new double[n];
        cuMemcpyDtoH(Pointer.to(hostData), devicePointer, n * Sizeof.DOUBLE);
        return hostData;
    }
    
    /**
     * Free the given device pointer
     * 
     * @param devicePointer The device pointer
     */
    static void freeDevicePointer(CUdeviceptr devicePointer)
    {
        cuMemFree(devicePointer);
    }

    /**
     * Returns whether the given arrays are equal within the given tolerance
     * 
     * @param result The computed result
     * @param reference The reference to compare against
     * @param epsilon The tolerance
     * @param verbose Whether the output should be verbose and continue
     * even if a mismatch is found
     * @return Whether the given arrays are equal
     */
    static boolean equalFloat(
        float result[], float reference[], float epsilon, boolean verbose)
    {
        if (result.length != reference.length)
        {
            if (verbose)
            {
                System.out.println("Arrays have different lengths");
            }
            return false;
        }
        boolean passed = true;
        for(int i = 0; i < result.length; i++)
        {
            if (Math.abs(result[i] - reference[i]) > epsilon)
            {
                passed = false;
                if (verbose)
                {
                    System.out.println(
                        "At index "+i+ " found "+result[i]+
                        " but expected "+reference[i]);
                }
                else
                {
                    break;
                }
            }
        }
        return passed;
    }

    /**
     * Returns whether the given arrays are equal within the given tolerance
     * 
     * @param result The computed result
     * @param reference The reference to compare against
     * @param epsilon The tolerance
     * @param verbose Whether the output should be verbose and continue
     * even if a mismatch is found
     * @return Whether the given arrays are equal
     */
    static boolean equalDouble(
        double result[], double reference[], double epsilon, boolean verbose)
    {
        if (result.length != reference.length)
        {
            if (verbose)
            {
                System.out.println("Arrays have different lengths");
            }
            return false;
        }
        boolean passed = true;
        for(int i = 0; i < result.length; i++)
        {
            if (Math.abs(result[i] - reference[i]) > epsilon)
            {
                passed = false;
                if (verbose)
                {
                    System.out.println(
                        "At index "+i+ " found "+result[i]+
                        " but expected "+reference[i]);
                }
                else
                {
                    break;
                }
            }
        }
        return passed;
    }
    

    /**
     * Private constructor to prevent instantiation
     */
    private TestUtil()
    {
        // Private constructor to prevent instantiation
    }
}
