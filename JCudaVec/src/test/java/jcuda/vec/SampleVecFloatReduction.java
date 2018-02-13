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
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.util.Locale;
import java.util.stream.DoubleStream;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * A sample showing how to use the JCuda vector library
 */
public class SampleVecFloatReduction
{
    /**
     * Entry point of this simple
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        // Define the host- and device data
        float hostData[] = new float[] {
            2.0f, 1.0f, 20.0f, 10.0f, -1.0f, -5.0f 
        };
        int n = hostData.length;
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, n * Sizeof.FLOAT);
        cudaMemcpy(deviceData, Pointer.to(hostData), 
            n * Sizeof.FLOAT, cudaMemcpyHostToDevice);

        // Create a handle for the vector operations
        VecHandle handle = Vec.createHandle();
        
        // Perform the vector operations
        float add = VecFloat.reduceAdd(handle, n, deviceData);  
        float mul = VecFloat.reduceMul(handle, n, deviceData);  
        float min = VecFloat.reduceMin(handle, n, deviceData);  
        float max = VecFloat.reduceMax(handle, n, deviceData);  

        double addHost = doubleStreamOf(hostData).reduce(0.0, (x,y) -> x + y);
        double mulHost = doubleStreamOf(hostData).reduce(1.0, (x,y) -> x * y);
        double minHost = doubleStreamOf(hostData).reduce(
            Double.MAX_VALUE, Math::min);
        double maxHost = doubleStreamOf(hostData).reduce(
            Double.MIN_VALUE, Math::max);
        
        System.out.printf(
            "        %10s %10s %10s %10s\n",
            "add", "mul", "min", "max"); 
        System.out.printf(Locale.ENGLISH, 
            "Device: %10.3f %10.3f %10.3f %10.3f\n", 
            add, mul, min, max);
        System.out.printf(Locale.ENGLISH, 
            "Host  : %10.3f %10.3f %10.3f %10.3f\n", 
            addHost, mulHost, minHost, maxHost);
        

        // Clean up.
        cudaFree(deviceData);
        Vec.destroyHandle(handle);
    }
    
    /**
     * Creates a DoubleStream from the given float data
     * 
     * @param data The data
     * @return The stream
     */
    private static DoubleStream doubleStreamOf(float data[])
    {
        double doubleData[] = new double[data.length];
        for (int i = 0; i < data.length; i++)
        {
            doubleData[i] = data[i];
        }
        return DoubleStream.of(doubleData);
    }

}
