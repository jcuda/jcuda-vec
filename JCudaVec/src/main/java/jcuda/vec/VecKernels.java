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

import static jcuda.driver.JCudaDriver.cuCtxSetCurrent;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoadDataEx;
import static jcuda.driver.JCudaDriver.cuModuleUnload;

import java.util.LinkedHashMap;
import java.util.Map;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

/**
 * A class that encapsulates a CUModule and allows calling the vector 
 * operation kernels that are defined in the module. 
 */
class VecKernels
{
    /**
     * The CUcontext for which these kernels have been created
     */
    private final CUcontext context;
    
    /**
     * The module from which the kernels (i.e. the CUfunctions)
     * are obtained
     */
    private final CUmodule module;

    /**
     * The prefix that should be added to a kernel name in order to build
     * the name for the CUfunction
     */
    private final String kernelNamePrefix;

    /**
     * The suffix that should be added to a kernel name in order to build
     * the name for the CUfunction
     */
    private final String kernelNameSuffix;
    
    /**
     * The mapping from kernel names to CUfunctions
     */
    private final Map<String, CUfunction> functions;
    
    /**
     * The block dimension, in x-direction, that should be used for the calls
     */
    private int blockDimX;
    
    /**
     * Creates a new kernel set that allows calling the functions that 
     * are contained in the CUDA module that is read from a PTX file.
     * 
     * @param context The CUDA context
     * @param dataTypeName The type for the kernels, either "float"
     * or "double"
     * @param kernelNamePrefix The prefix that should be added to a 
     * kernel name in order to build the name for the CUfunction
     * @param kernelNameSuffix The suffix that should be added to a 
     * kernel name in order to build the name for the CUfunction
     */
    VecKernels(
        CUcontext context,
    	String dataTypeName,
    	String kernelNamePrefix,
    	String kernelNameSuffix)
    {
        this.context = context;
        
        VecUtils.checkResultDriver(cuCtxSetCurrent(context));
        
        this.kernelNamePrefix = kernelNamePrefix;
        this.kernelNameSuffix = kernelNameSuffix;
        
        this.blockDimX = VecUtils.getMaxBlockDimX();

        this.module = new CUmodule();
        
        String ptxFileNamePrefix = 
            "/kernels/JCudaVec_kernels_" + dataTypeName;
        String ptxFileName = VecUtils.createPtxFileName(ptxFileNamePrefix);
        
        //System.out.println("Loading "+ptxFileName);
        byte ptxData[] = VecUtils.loadData(ptxFileName);
        
        VecUtils.checkResultDriver(
            cuModuleLoadDataEx(module, Pointer.to(ptxData), 
                0, new int[0], Pointer.to(new int[0])));

        this.functions = new LinkedHashMap<String, CUfunction>();
    }

    /**
     * Call the kernel that is identified by the given name, with the
     * given arguments. Note that the given name must not necessarily
     * be the name that the kernel has in the CUDA source code.
     * 
     * @param name The name identifying the kernel
     * @param workSize The global work size of the kernel
     * @param arguments The arguments for the kernel
     * @throws CudaException If the kernel could not be called
     */
	synchronized void call(
	    String name, long workSize, Object ... arguments) 
	{
        VecUtils.checkResultDriver(cuCtxSetCurrent(context));
        CUfunction function = obtainFunction(name);
        Pointer kernelParameters = VecUtils.setupKernelParameters(arguments);
        callKernel(workSize, function, kernelParameters);
	}

    /**
     * Obtain the CUfunction for the kernel that is identified with the 
     * given name, loading it from the module if necessary.
     * 
     * @param name The name of the kernel
     * @return The CUfunction for the kernel
     */
    private CUfunction obtainFunction(String name)
    {
    	CUfunction function = functions.get(name);
        if (function == null)
        {
            function = new CUfunction();
            VecUtils.checkResultDriver(cuModuleGetFunction(function, module,
                kernelNamePrefix + name + kernelNameSuffix));
            functions.put(name, function);
        }
        return function;
    }
    
	
    /**
     * Call the given CUDA function with the given parameters
     * 
     * @param workSize The global work size
     * @param function The CUDA function
     * @param kernelParameters The kernel parameters
     */
    private void callKernel(long workSize, CUfunction function, 
        Pointer kernelParameters)
    {
        int gridDimX = (int)Math.ceil((double)workSize / blockDimX);
        VecUtils.checkResultDriver(cuLaunchKernel(function,
            gridDimX,  1, 1,
            blockDimX, 1, 1,
            0, Vec.getStream(),
            kernelParameters, null));
    }

    /**
     * Perform a shutdown, releasing all resources that have been
     * allocated by this instance.
     */
    public void shutdown()
    {
        VecUtils.checkResultDriver(cuCtxSetCurrent(context));
        VecUtils.checkResultDriver(cuModuleUnload(module));
    }
}