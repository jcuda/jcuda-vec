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

import org.junit.Test;

import jcuda.Pointer;

/**
 * Tests for the vector arithmetic methods
 */
@SuppressWarnings("javadoc")
public class TestVecFloatArithmetic extends AbstractTestVecFloat
{
    @Test
    public void testAdd()
    {
        runTest(new AbstractCoreFloat("add")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x+y;
            }
            
            @Override
            protected void computeDevice(
                VecHandle handle, long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.add(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testSub()
    {
        runTest(new AbstractCoreFloat("sub")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x-y;
            }
            
            @Override
            protected void computeDevice(
                VecHandle handle, long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.sub(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testMul()
    {
        runTest(new AbstractCoreFloat("mul")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x*y;
            }
            
            @Override
            protected void computeDevice(
                VecHandle handle, long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.mul(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testDiv()
    {
        runTest(new AbstractCoreFloat("div")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x/y;
            }
            
            @Override
            protected void computeDevice(
                VecHandle handle, long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.div(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testNegate()
    {
        runTest(new AbstractCoreFloat("negate")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return -x;
            }
            
            @Override
            protected void computeDevice(
                VecHandle handle, long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.negate(handle, n, result, x);
            }
        });
    }

    
}
