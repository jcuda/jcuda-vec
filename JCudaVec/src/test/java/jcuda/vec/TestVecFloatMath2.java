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
 * Tests for the 2-argument vector math methods
 */
@SuppressWarnings("javadoc")
public class TestVecFloatMath2 extends AbstractTestVecFloat
{
    @Test
    public void testCopysign()
    {
        runTest(new AbstractCoreFloat("copysign")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return y<0?-Math.abs(x):Math.abs(x);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.copysign(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFdim()
    {
        runTest(new AbstractCoreFloat("fdim")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x-y<0?0:x-y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.fdim(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFdivide()
    {
        runTest(new AbstractCoreFloat("fdivide")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x/y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.fdivide(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFmax()
    {
        runTest(new AbstractCoreFloat("fmax")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return Math.max(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.fmax(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFmin()
    {
        runTest(new AbstractCoreFloat("fmin")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return Math.min(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.fmin(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFmod()
    {
        runTest(new AbstractCoreFloat("fmod")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return x % y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.fmod(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testHypot()
    {
        runTest(new AbstractCoreFloat("hypot")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.hypot(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.hypot(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testNextafter()
    {
        runTest(new AbstractCoreFloat("nextafter")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return Math.nextAfter(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.nextafter(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testPow()
    {
        runTest(new AbstractCoreFloat("pow")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.pow(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.pow(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testRemainder()
    {
        runTest(new AbstractCoreFloat("remainder")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.IEEEremainder(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, float scalar)
            {
                VecFloat.remainder(handle, n, result, x, y);
            }
        });
    }

    
}
