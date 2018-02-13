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

import static jcuda.driver.JCudaDriver.cuCtxGetDevice;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdevice_attribute;
import jcuda.driver.CUresult;
import jcuda.runtime.cudaError;

/**
 * Utility methods for the jcuda.vec package
 */
class VecUtils
{
    /**
     * If the given result is not CUresult.CUDA_SUCCESS, then this method
     * throws a CudaException with the error message for the given result.
     * 
     * @param cuResult The result
     * @throws CudaException if the result is not CUresult.CUDA_SUCCESS
     */
    static void checkResultDriver(int cuResult)
    {
        if (cuResult != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(CUresult.stringFor(cuResult));
        }
    }

    /**
     * If the given result is not cudaError.cudaSuccess, then this method
     * throws a CudaException with the error message for the given result.
     * 
     * @param cudaResult The result
     * @throws CudaException if the result is not cudaError.cudaSuccess
     */
    static void checkResultRuntime(int cudaResult)
    {
        if (cudaResult != cudaError.cudaSuccess)
        {
            throw new CudaException(cudaError.stringFor(cudaResult));
        }
    }
    
    /**
     * Append a string <code>"_MODEL_ccVERSION.ptx</code> to the given string.
     * <br>
     * <br>
     * <code>MODEL</code> is the bitness of the JVM.<br>
     * <br>
     * <code>VERSION</code> is a version indicator for the compute capability 
     * of the device of the current context. It is currently at most "30".
     * 
     * @param ptxFileNamePrefix The prefix for the file name
     * @return The PTX file name
     * @throws CudaException If the model (i.e. the bitness of the JVM) is
     * not "64", or if the major version number of the compute capability 
     * is not at least 3. 
     */
    static String createPtxFileName(String ptxFileNamePrefix)
    {
        String modelString = System.getProperty("sun.arch.data.model");
        if (!"64".equals(modelString))
        {
            throw new CudaException(
                "Only 64 bit platforms are supported. Found " + modelString);
        }
        int ccMajor = getComputeCapabilityMajor();
        String ccString = "30";
        if (ccMajor <= 2)
        {
            throw new CudaException(
                "Compute capability of at least 3.0 is required. "
                + "Found " + ccMajor);
        }
        String ptxFileName =
            ptxFileNamePrefix + "_" + modelString + "_cc" + ccString + ".ptx";
        return ptxFileName;
    }
    
    /**
     * Obtain the CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR of the 
     * current device
     * 
     * @return The major version number part of the compute capability
     */
    private static int getComputeCapabilityMajor()
    {
        return getDeviceAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    }
    
    /**
     * Obtain the CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X of the current device
     * 
     * @return The maximum block dimension, in x-direction
     */
    static int getMaxBlockDimX()
    {
        return getDeviceAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    }
    
    /**
     * Obtain the CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X of the current device
     * 
     * @return The maximum grid dimension, in x-direction
     */
    static int getMaxGridDimX()
    {
        return getDeviceAttribute(
            CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    }
    
    /**
     * Obtain the value of the specified attribute for the device of the
     * current context. 
     * 
     * @param attribute The CUdevice_attribute
     * @return The value of the attribute
     */
    private static int getDeviceAttribute(int attribute)
    {
        CUdevice device = new CUdevice();
        VecUtils.checkResultDriver(cuCtxGetDevice(device));
        int value[] = {0};
        VecUtils.checkResultDriver(
            cuDeviceGetAttribute(value, attribute, device));
        return value[0];
    }
    
    
    /**
     * Reads the data from a file resource with the given name, and returns 
     * it as a 0-terminated byte array. 
     * 
     * @param ptxFileName The name of the file to read
     * @return The data from the file
     * @throws CudaException If there is an IO error
     */
    static byte[] loadData(String ptxFileName)
    {
        InputStream ptxInputStream = null;
        try
        {
            ptxInputStream = VecUtils.class.getResourceAsStream(ptxFileName);
            if (ptxInputStream != null)
            {
                return loadData(ptxInputStream);
            }
            else
            {
                throw new CudaException("Could not initialize the kernels: "
                    + "Resource " + ptxFileName + " not found");
            }
        }
        finally
        {
            if (ptxInputStream != null)
            {
                try
                {
                    ptxInputStream.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                        "Could not initialize the kernels", e);
                }
            }
        }
        
    }
    
    /**
     * Reads the data from the given inputStream and returns it as
     * a 0-terminated byte array. The caller is responsible to 
     * close the given stream.
     * 
     * @param inputStream The inputStream to read
     * @return The data from the inputStream
     * @throws CudaException If there is an IO error
     */
    private static byte[] loadData(InputStream inputStream)
    {
        ByteArrayOutputStream baos = null;
        try
        {
            baos = new ByteArrayOutputStream();
            byte buffer[] = new byte[8192];
            while (true)
            {
                int read = inputStream.read(buffer);
                if (read == -1)
                {
                    break;
                }
                baos.write(buffer, 0, read);
            }
            baos.write('\0');
            baos.flush();
            return baos.toByteArray();
        }
        catch (IOException e)
        {
            throw new CudaException(
                "Could not load data", e);
        }
        finally
        {
            if (baos != null)
            {
                try
                {
                    baos.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                        "Could not close output", e);
                }
            }
        }
    }

    
    /**
     * Create a pointer to the given arguments that can be used as
     * the parameters for a kernel launch.
     * 
     * @param args The arguments
     * @return The pointer for the kernel arguments 
     * @throws NullPointerException If one of the given arguments is 
     * <code>null</code>
     * @throws CudaException If one of the given arguments has a type
     * that can not be passed to a kernel (that is, a type that is
     * neither primitive nor a {@link Pointer})
     */
    static Pointer setupKernelParameters(Object ... args)
    {
        Pointer kernelParameters[] = new Pointer[args.length];
        for (int i = 0; i < args.length; i++)
        {
            Object arg = args[i];
            if (arg == null)
            {
                throw new NullPointerException("Argument " + i + " is null");
            }
            if (arg instanceof Pointer)
            {
                Pointer argPointer = (Pointer)arg;
                Pointer pointer = Pointer.to(argPointer);
                kernelParameters[i] = pointer;
            }
            else if (arg instanceof Byte)
            {
                Byte value = (Byte)arg;
                Pointer pointer = Pointer.to(new byte[]{value});
                kernelParameters[i] = pointer;
            }
            else if (arg instanceof Short)
            {
                Short value = (Short)arg;
                Pointer pointer = Pointer.to(new short[]{value});
                kernelParameters[i] = pointer;
            }
            else if (arg instanceof Integer)
            {
                Integer value = (Integer)arg;
                Pointer pointer = Pointer.to(new int[]{value});
                kernelParameters[i] = pointer;
            }
            else if (arg instanceof Long)
            {
                Long value = (Long)arg;
                Pointer pointer = Pointer.to(new long[]{value});
                kernelParameters[i] = pointer;
            }
            else if (arg instanceof Float)
            {
                Float value = (Float)arg;
                Pointer pointer = Pointer.to(new float[]{value});
                kernelParameters[i] = pointer;
            }
            else if (arg instanceof Double)
            {
                Double value = (Double)arg;
                Pointer pointer = Pointer.to(new double[]{value});
                kernelParameters[i] = pointer;
            }
            else
            {
                throw new CudaException("Type " + arg.getClass()
                    + " may not be passed to a function");
            }
        }
        return Pointer.to(kernelParameters);
    }

    /**
     * Private constructor to prevent instantiation
     */
    private VecUtils()
    {
        // Private constructor to prevent instantiation
    }
}
