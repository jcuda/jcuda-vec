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

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * Utility for the JCudaVec tests
 */
public class TestUtil
{
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
        for (int i = 0; i < n; i++)
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
        for (int i = 0; i < n; i++)
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
    static Pointer createDevicePointerFloat(int n)
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
    static Pointer createDevicePointerDouble(int n)
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
    static Pointer createDevicePointerFloat(float hostData[])
    {
        Pointer devicePointer = new Pointer();
        cudaMalloc(devicePointer, hostData.length * Sizeof.FLOAT);
        cudaMemcpy(devicePointer, Pointer.to(hostData),
            hostData.length * Sizeof.FLOAT, cudaMemcpyHostToDevice);
        return devicePointer;
    }

    /**
     * Create a device pointer that points to a memory block with the
     * given data
     * 
     * @param hostData The host data
     * @return The device pointer
     */
    static Pointer createDevicePointerDouble(double hostData[])
    {
        Pointer devicePointer = new Pointer();
        cudaMalloc(devicePointer, hostData.length * Sizeof.DOUBLE);
        cudaMemcpy(devicePointer, Pointer.to(hostData),
            hostData.length * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
        return devicePointer;
    }

    /**
     * Copy the data from the given device pointer into an array and return it
     * 
     * @param devicePointer The device pointer
     * @param n The size of the data block, in number of elements 
     * @return The array
     */
    static float[] createHostDataFloat(Pointer devicePointer, int n)
    {
        float hostData[] = new float[n];
        cudaMemcpy(Pointer.to(hostData), devicePointer,
            hostData.length * Sizeof.FLOAT, cudaMemcpyDeviceToHost);
        return hostData;
    }
    
    /**
     * Copy the data from the given device pointer into an array and return it
     * 
     * @param devicePointer The device pointer
     * @param n The size of the data block, in number of elements 
     * @return The array
     */
    static double[] createHostDataDouble(Pointer devicePointer, int n)
    {
        double hostData[] = new double[n];
        cudaMemcpy(Pointer.to(hostData), devicePointer,
            hostData.length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        return hostData;
    }
    
    /**
     * Free the given device pointer
     * 
     * @param devicePointer The device pointer
     */
    static void freeDevicePointer(Pointer devicePointer)
    {
        cudaFree(devicePointer);
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
