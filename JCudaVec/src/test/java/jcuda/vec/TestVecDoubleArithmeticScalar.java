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

import org.junit.Test;

import jcuda.driver.CUdeviceptr;

/**
 * Tests for the vector arithmetic scalar methods
 */
public class TestVecDoubleArithmeticScalar extends AbstractTestVecDouble
{
    @Test
    public void testAddScalar()
    {
        runTest(new AbstractCoreDouble("addScalar")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return x+scalar;
            }
            
            @Override
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, double scalar)
            {
                VecDouble.addScalar(n, result, x, scalar);
            }
        });
    }

    @Test
    public void testSubScalar()
    {
        runTest(new AbstractCoreDouble("subScalar")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return x-scalar;
            }
            
            @Override
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, double scalar)
            {
                VecDouble.subScalar(n, result, x, scalar);
            }
        });
    }

    @Test
    public void testMulScalar()
    {
        runTest(new AbstractCoreDouble("mulScalar")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return x*scalar;
            }
            
            @Override
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, double scalar)
            {
                VecDouble.mulScalar(n, result, x, scalar);
            }
        });
    }

    @Test
    public void testDivScalar()
    {
        runTest(new AbstractCoreDouble("divScalar")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return x/scalar;
            }
            
            @Override
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, double scalar)
            {
                VecDouble.divScalar(n, result, x, scalar);
            }
        });
    }


    @Test
    public void testScalarAdd()
    {
        runTest(new AbstractCoreDouble("scalarAdd")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return scalar+y;
            }
            
            @Override
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, double scalar)
            {
                VecDouble.scalarAdd(n, result, scalar, y);
            }
        });
    }

    @Test
    public void testScalarSub()
    {
        runTest(new AbstractCoreDouble("scalarSub")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return scalar-y;
            }
            
            @Override
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, double scalar)
            {
                VecDouble.scalarSub(n, result, scalar, y);
            }
        });
    }

    @Test
    public void testScalarMul()
    {
        runTest(new AbstractCoreDouble("scalarMul")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return scalar*y;
            }
            
            @Override
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, double scalar)
            {
                VecDouble.scalarMul(n, result, scalar, y);
            }
        });
    }

    @Test
    public void testScalarDiv()
    {
        runTest(new AbstractCoreDouble("scalarDiv")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return scalar/y;
            }
            
            @Override
            protected void computeDevice(long n, CUdeviceptr result,
                CUdeviceptr x, CUdeviceptr y, double scalar)
            {
                VecDouble.scalarDiv(n, result, scalar, y);
            }
        });
    }
    
}
