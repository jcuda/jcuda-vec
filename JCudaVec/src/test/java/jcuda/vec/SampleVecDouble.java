/*
 * JCudaVec - Vector operations for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2013-2018 Marco Hutter - http://www.jcuda.org
 */
package jcuda.vec;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * A sample showing how to use the JCuda vector library
 */
public class SampleVecDouble
{
    /**
     * Entry point of this simple
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        // Allocate and fill the host input data
        int n = 50000;
        double hostX[] = new double[n];
        double hostY[] = new double[n];
        for(int i = 0; i < n; i++)
        {
            hostX[i] = (double)i;
            hostY[i] = (double)i;
        }

        // Allocate the device pointers, and copy the
        // host input data to the device
        Pointer deviceX = new Pointer();
        cudaMalloc(deviceX, n * Sizeof.DOUBLE);
        cudaMemcpy(deviceX, Pointer.to(hostX), 
            n * Sizeof.DOUBLE, cudaMemcpyHostToDevice);

        Pointer deviceY = new Pointer();
        cudaMalloc(deviceY, n * Sizeof.DOUBLE); 
        cudaMemcpy(deviceY, Pointer.to(hostY), 
            n * Sizeof.DOUBLE, cudaMemcpyHostToDevice);

        Pointer deviceResult = new Pointer();
        cudaMalloc(deviceResult, n * Sizeof.DOUBLE);

        // Create a handle for the vector operations
        VecHandle handle = Vec.createHandle();
        
        // Perform the vector operations
        VecDouble.cos(handle, n, deviceX, deviceX);               // x = cos(x)  
        VecDouble.mul(handle, n, deviceX, deviceX, deviceX);      // x = x*x
        VecDouble.sin(handle, n, deviceY, deviceY);               // y = sin(y)
        VecDouble.mul(handle, n, deviceY, deviceY, deviceY);      // y = y*y
        VecDouble.add(handle, n, deviceResult, deviceX, deviceY); // result = x+y

        // Allocate host output memory and copy the device output
        // to the host.
        double hostResult[] = new double[n];
        cudaMemcpy(Pointer.to(hostResult), deviceResult, 
            n * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        // Verify the result
        boolean passed = true;
        for(int i = 0; i < n; i++)
        {
            double expected = (
                Math.cos(hostX[i])*Math.cos(hostX[i])+
                Math.sin(hostY[i])*Math.sin(hostY[i]));
            if (Math.abs(hostResult[i] - expected) > 1e-5)
            {
                System.out.println(
                    "At index " + i + " found " + hostResult[i] +
                    " but expected " + expected);
                passed = false;
                break;
            }
        }
        System.out.println("Test " + (passed ? "PASSED" : "FAILED"));

        // Clean up.
        cudaFree(deviceX);
        cudaFree(deviceY);
        cudaFree(deviceResult);
        Vec.destroyHandle(handle);
    }

}
