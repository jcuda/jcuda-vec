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
 * Tests for the vector comparison scalar methods
 */
@SuppressWarnings("javadoc")
public class TestVecFloatComparisonScalar extends AbstractTestVecFloat
{
    @Test
    public void testLtScalar()
    {
        runTest(new AbstractCoreFloat("ltScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x<scalar?1.0f:0.0f;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.ltScalar(handle, n, result, x, scalar);
            }
        });
    }

    @Test
    public void testLteScalar()
    {
        runTest(new AbstractCoreFloat("lteScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x<=scalar?1.0f:0.0f;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.lteScalar(handle, n, result, x, scalar);
            }
        });
    }

    @Test
    public void testEqScalar()
    {
        runTest(new AbstractCoreFloat("eqScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x==scalar?1.0f:0.0f;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.eqScalar(handle, n, result, x, scalar);
            }
        });
    }

    @Test
    public void testGteScalar()
    {
        runTest(new AbstractCoreFloat("gteScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x>=scalar?1.0f:0.0f;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.gteScalar(handle, n, result, x, scalar);
            }
        });
    }

    @Test
    public void testGtScalar()
    {
        runTest(new AbstractCoreFloat("gtScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x>scalar?1.0f:0.0f;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.gtScalar(handle, n, result, x, scalar);
            }
        });
    }

    
    @Test
    public void testNeScalar()
    {
        runTest(new AbstractCoreFloat("neScalar")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x!=scalar?1.0f:0.0f;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result,
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.neScalar(handle, n, result, x, scalar);
            }
        });
    }

    
}
