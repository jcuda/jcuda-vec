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
public class SampleVecDoubleReduction
{
    /**
     * Entry point of this simple
     * 
     * @param args Not used
     */
    public static void main(String[] args)
    {
        // Define the host- and device data
        double hostData[] = new double[] {
            2.0, 1.0, 20.0, 10.0, -1.0, -5.0 
        };
        int n = hostData.length;
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, n * Sizeof.DOUBLE);
        cudaMemcpy(deviceData, Pointer.to(hostData), 
            n * Sizeof.DOUBLE, cudaMemcpyHostToDevice);

        // Create a handle for the vector operations
        VecHandle handle = Vec.createHandle();
        
        // Perform the vector operations
        double add = VecDouble.reduceAdd(handle, n, deviceData);  
        double mul = VecDouble.reduceMul(handle, n, deviceData);  
        double min = VecDouble.reduceMin(handle, n, deviceData);  
        double max = VecDouble.reduceMax(handle, n, deviceData);  

        double addHost = DoubleStream.of(hostData).reduce(0.0, (x,y) -> x + y);
        double mulHost = DoubleStream.of(hostData).reduce(1.0, (x,y) -> x * y);
        double minHost = DoubleStream.of(hostData).reduce(
            Double.MAX_VALUE, Math::min);
        double maxHost = DoubleStream.of(hostData).reduce(
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
}
