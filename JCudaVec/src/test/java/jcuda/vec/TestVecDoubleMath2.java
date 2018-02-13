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
public class TestVecDoubleMath2 extends AbstractTestVecDouble
{
    @Test
    public void testCopysign()
    {
        runTest(new AbstractCoreDouble("copysign")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return y<0?-Math.abs(x):Math.abs(x);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.copysign(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFdim()
    {
        runTest(new AbstractCoreDouble("fdim")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return x-y<0?0:x-y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.fdim(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFdivide()
    {
        runTest(new AbstractCoreDouble("fdivide")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return x/y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.fdivide(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFmax()
    {
        runTest(new AbstractCoreDouble("fmax")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.max(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.fmax(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFmin()
    {
        runTest(new AbstractCoreDouble("fmin")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.min(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.fmin(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testFmod()
    {
        runTest(new AbstractCoreDouble("fmod")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return x % y;
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.fmod(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testHypot()
    {
        runTest(new AbstractCoreDouble("hypot")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.hypot(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.hypot(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testNextafter()
    {
        runTest(new AbstractCoreDouble("nextafter")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.nextAfter(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.nextafter(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testPow()
    {
        runTest(new AbstractCoreDouble("pow")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.pow(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.pow(handle, n, result, x, y);
            }
        });
    }

    @Test
    public void testRemainder()
    {
        runTest(new AbstractCoreDouble("remainder")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.IEEEremainder(x, y);
            }
            
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.remainder(handle, n, result, x, y);
            }
        });
    }

    
}
