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

import jcuda.driver.CUdeviceptr;

/**
 * The core of a test for one method of the VecFloat class
 */
abstract class AbstractCoreFloat
{
    /**
     * The name of the method
     */
    private final String name;
    
    /**
     * Creates a new core with the given name
     * 
     * @param name The name of the core
     */
    AbstractCoreFloat(String name)
    {
        this.name = name;
    }
    
    /**
     * Compute the reference result
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first input vector
     * @param y The second input vector
     * @param scalar The scalar
     */
    protected void computeHost(
        long n, float result[], float x[], float y[], float scalar)
    {
        for (int i=0; i<n; i++)
        {
            result[i] = computeHostElement(x[i], y[i], scalar);
        }
    }
    
    /**
     * Computes a single element of the reference result
     * 
     * @param x The first argument
     * @param y The second argument
     * @param scalar The scalar
     * @return The result
     */
    protected abstract float computeHostElement(
        float x, float y, float scalar);
    
    /**
     * Compute the result using the JCudaVec method
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first input vector
     * @param y The second input vector
     * @param scalar The scalar
     */
    protected abstract void computeDevice(long n, 
        CUdeviceptr result, CUdeviceptr x, CUdeviceptr y, float scalar);

    @Override
    public String toString()
    {
        return name;
    }
}
