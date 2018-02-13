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
 * Tests for the vector arithmetic scalar methods
 */
@SuppressWarnings("javadoc")
public class TestVecFloatArithmeticScalar extends AbstractTestVecFloat
{
    @Test
    public void testAddScalar()
    {
        runTest(new AbstractCoreFloat("addScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x+scalar;
            }
            
            @Override
            protected void computeDevice(VecHandle handle, 
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.addScalar(handle, n, result, x, scalar);
            }
        });
    }

    @Test
    public void testSubScalar()
    {
        runTest(new AbstractCoreFloat("subScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x-scalar;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.subScalar(handle, n, result, x, scalar);
            }
        });
    }

    @Test
    public void testMulScalar()
    {
        runTest(new AbstractCoreFloat("mulScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x*scalar;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.mulScalar(handle, n, result, x, scalar);
            }
        });
    }

    @Test
    public void testDivScalar()
    {
        runTest(new AbstractCoreFloat("divScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x/scalar;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.divScalar(handle, n, result, x, scalar);
            }
        });
    }


    @Test
    public void testScalarAdd()
    {
        runTest(new AbstractCoreFloat("scalarAdd")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return scalar+y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.scalarAdd(handle, n, result, scalar, y);
            }
        });
    }

    @Test
    public void testScalarSub()
    {
        runTest(new AbstractCoreFloat("scalarSub")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return scalar-y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.scalarSub(handle, n, result, scalar, y);
            }
        });
    }

    @Test
    public void testScalarMul()
    {
        runTest(new AbstractCoreFloat("scalarMul")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return scalar*y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.scalarMul(handle, n, result, scalar, y);
            }
        });
    }

    @Test
    public void testScalarDiv()
    {
        runTest(new AbstractCoreFloat("scalarDiv")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return scalar/y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.scalarDiv(handle, n, result, scalar, y);
            }
        });
    }
    
}
