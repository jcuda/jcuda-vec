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

import static jcuda.driver.JCudaDriver.cuCtxGetCurrent;
import static jcuda.driver.JCudaDriver.cuCtxGetDevice;
import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDevicePrimaryCtxRetain;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.runtime.JCuda.cudaGetDevice;

import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUstream;
import jcuda.runtime.cudaStream_t;

/**
 * Common methods for the JCudaVec library.<br>
 * <br>
 * The main methods in this class are used for creating and destroying 
 * {@link VecHandle} instances that summarize the context for the 
 * actual vector library calls. 
 */
public class Vec
{
    /**
     * The stream that will be used for the vector operations
     */
    private static volatile CUstream stream;
    
    /**
     * Create a new {@link VecHandle} instance. This method creates the 
     * internal structures that summarize the vector library context. The
     * returned handle must be passed to all {@link VecFloat} and 
     * {@link VecDouble} vector operation methods.
     * 
     * @return The {@link VecHandle}
     */
    public static synchronized VecHandle createHandle()
    {
        VecUtils.checkResultDriver(cuInit(0));
        
        // Obtain the current context
        CUcontext context = new CUcontext();
        VecUtils.checkResultDriver(cuCtxGetCurrent(context));

        CUdevice device = new CUdevice();
        
        // If there is no context, use the primary context
        boolean usingPrimaryContext = false;
        CUcontext NULL_CONTEXT = new CUcontext();
        if (context.equals(NULL_CONTEXT))
        {
            usingPrimaryContext = true;
            
            // Obtain the device that is currently selected via the runtime API
            int deviceArray[] = { 0 };
            VecUtils.checkResultRuntime(cudaGetDevice(deviceArray));
            int deviceIndex = deviceArray[0];
            
            // Obtain the device and its primary context
            VecUtils.checkResultDriver(
                cuDeviceGet(device, deviceIndex));
            VecUtils.checkResultDriver(
                cuDevicePrimaryCtxRetain(context, device));
            VecUtils.checkResultDriver(
                cuCtxSetCurrent(context));
        }
        else
        {
            VecUtils.checkResultDriver(
                cuCtxGetDevice(device));
        }
        
        VecKernels vecKernelsFloat = 
            new VecKernels(context, "float", "vec_", "f");
        VecKernels vecKernelsDouble = 
            new VecKernels(context, "double", "vec_", "");
        
        VecReductions vecReductionsFloat = 
            new VecReductions(context, "float", Sizeof.FLOAT);
        VecReductions vecReductionsDouble = 
            new VecReductions(context, "double", Sizeof.DOUBLE);
        
        VecHandle handle = new VecHandle(
            device, usingPrimaryContext, 
            vecKernelsFloat, vecKernelsDouble,
            vecReductionsFloat, vecReductionsDouble);
        return handle;
    }
    
    /**
     * Destroy the given {@link VecHandle} and release all associated
     * resources. Calling this method with a handle that already has
     * been destroyed has no effect.
     * 
     * @param handle The {@link VecHandle} to destroy
     */
    public static void destroyHandle(VecHandle handle)
    {
        handle.destroy();
    }
    
    /**
     * Sets the library stream, which will be used to execute all subsequent 
     * calls to the vector library functions. If the stream is not set, all 
     * kernels use the default (NULL) stream.
     *   
     * @param stream The stream to set
     */
    public static void setCudaStream(cudaStream_t stream)
    {
        if (stream == null)
        {
            Vec.stream = null;
        }
        else
        {
            Vec.stream = new CUstream(stream);
        }
    }
    
    /**
     * Returns the library stream, which is being used to execute all calls 
     * to the vector library functions. If the stream is not set, all 
     * kernels use the default (NULL) stream.
     *  
     * @return The current stream
     */
    public static cudaStream_t getCudaStream()
    {
        if (stream == null)
        {
            return null;
        }
        return new cudaStream_t(stream);
    }
    
    /**
     * Sets the library stream, which will be used to execute all subsequent 
     * calls to the vector library functions. If the stream is not set, all 
     * kernels use the default (NULL) stream.
     *   
     * @param stream The stream to set
     */
    public static void setStream(CUstream stream)
    {
        Vec.stream = stream;
    }
    
    /**
     * Returns the library stream, which is being used to execute all calls 
     * to the vector library functions. If the stream is not set, all 
     * kernels use the default (NULL) stream.
     *  
     * @return The current stream
     */
    public static CUstream getStream()
    {
        return Vec.stream;
    }
    
    
    
    /**
     * Private constructor to prevent instantiation
     */
    private Vec()
    {
        // Private constructor to prevent instantiation
    }

}
