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

/**
 * A set of CUDA kernels for vector operations. The kernels are CUfunction 
 * instances that are created from a CUmodule.
 */
interface VecKernels
{
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
    void call(String name, long workSize, Object ... arguments);
    
    /**
     * Perform a shutdown, releasing all resources that have been
     * allocated by this instance.
     */
    void shutdown();
}