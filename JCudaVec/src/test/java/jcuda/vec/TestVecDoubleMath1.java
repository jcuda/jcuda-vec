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

/*
 * NOTE: Many of these tests are commented out, because they don't
 * have a simple translation to java.lang.Math methods. They may
 * be extended in future.
 */

/**
 * Tests for the 1-argument vector math methods
 */
@SuppressWarnings("javadoc")
public class TestVecDoubleMath1 extends AbstractTestVecDouble
{
    @Test
    public void testAcos()
    {
        runTest(new AbstractCoreDouble("acos")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.acos(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.acos(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testAcosh()
//    {
//        runTest(new AbstractTestCoreDouble("acosh")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.acosh(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.acosh(handle, n, result, x);
//            }
//        });
//    }

    @Test
    public void testAsin()
    {
        runTest(new AbstractCoreDouble("asin")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.asin(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.asin(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testAsinh()
//    {
//        runTest(new AbstractTestCoreDouble("asinh")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.asinh(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.asinh(handle, n, result, x);
//            }
//        });
//    }

    @Test
    public void testAtan()
    {
        runTest(new AbstractCoreDouble("atan")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.atan(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.atan(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testAtanh()
//    {
//        runTest(new AbstractTestCoreDouble("atanh")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.atanh(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.atanh(handle, n, result, x);
//            }
//        });
//    }

    @Test
    public void testCbrt()
    {
        runTest(new AbstractCoreDouble("cbrt")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.cbrt(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.cbrt(handle, n, result, x);
            }
        });
    }

    @Test
    public void testCeil()
    {
        runTest(new AbstractCoreDouble("ceil")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.ceil(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.ceil(handle, n, result, x);
            }
        });
    }

    @Test
    public void testCos()
    {
        runTest(new AbstractCoreDouble("cos")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.cos(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.cos(handle, n, result, x);
            }
        });
    }

    @Test
    public void testCosh()
    {
        runTest(new AbstractCoreDouble("cosh")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.cosh(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.cosh(handle, n, result, x);
            }
        });
    }

    @Test
    public void testCospi()
    {
        runTest(new AbstractCoreDouble("cospi")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.cos(x*Math.PI);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.cospi(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testErfc()
//    {
//        runTest(new AbstractTestCoreDouble("erfc")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.erfc(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.erfc(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testErfcinv()
//    {
//        runTest(new AbstractTestCoreDouble("erfcinv")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.erfcinv(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.erfcinv(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testErfcx()
//    {
//        runTest(new AbstractTestCoreDouble("erfcx")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.erfcx(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.erfcx(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testErf()
//    {
//        runTest(new AbstractTestCoreDouble("erf")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.erf(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.erf(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testErfinv()
//    {
//        runTest(new AbstractTestCoreDouble("erfinv")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.erfinv(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.erfinv(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testExp10()
//    {
//        runTest(new AbstractTestCoreDouble("exp10")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.exp10(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.exp10(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testExp2()
//    {
//        runTest(new AbstractTestCoreDouble("exp2")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.exp2(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.exp2(handle, n, result, x);
//            }
//        });
//    }

    @Test
    public void testExp()
    {
        runTest(new AbstractCoreDouble("exp")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.exp(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.exp(handle, n, result, x);
            }
        });
    }

    @Test
    public void testExpm1()
    {
        runTest(new AbstractCoreDouble("expm1")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.expm1(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.expm1(handle, n, result, x);
            }
        });
    }

    @Test
    public void testFabs()
    {
        runTest(new AbstractCoreDouble("fabs")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.abs(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.fabs(handle, n, result, x);
            }
        });
    }

    @Test
    public void testFloor()
    {
        runTest(new AbstractCoreDouble("floor")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.floor(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.floor(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testJ0()
//    {
//        runTest(new AbstractTestCoreDouble("j0")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.j0(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.j0(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testJ1()
//    {
//        runTest(new AbstractTestCoreDouble("j1")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.j1(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.j1(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testLgamma()
//    {
//        runTest(new AbstractTestCoreDouble("lgamma")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.lgamma(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.lgamma(handle, n, result, x);
//            }
//        });
//    }

    @Test
    public void testLog10()
    {
        runTest(new AbstractCoreDouble("log10")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.log10(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.log10(handle, n, result, x);
            }
        });
    }

    @Test
    public void testLog1p()
    {
        runTest(new AbstractCoreDouble("log1p")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.log1p(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.log1p(handle, n, result, x);
            }
        });
    }

    @Test
    public void testLog2()
    {
        runTest(new AbstractCoreDouble("log2")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return (Math.log(x) / Math.log(2));
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.log2(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testLogb()
//    {
//        runTest(new AbstractTestCoreDouble("logb")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.logb(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.logb(handle, n, result, x);
//            }
//        });
//    }

    @Test
    public void testLog()
    {
        runTest(new AbstractCoreDouble("log")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.log(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.log(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testNormcdf()
//    {
//        runTest(new AbstractTestCoreDouble("normcdf")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.normcdf(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.normcdf(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testNormcdfinv()
//    {
//        runTest(new AbstractTestCoreDouble("normcdfinv")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.normcdfinv(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.normcdfinv(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testRcbrt()
//    {
//        runTest(new AbstractTestCoreDouble("rcbrt")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.rcbrt(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.rcbrt(handle, n, result, x);
//            }
//        });
//    }

    @Test
    public void testRint()
    {
        runTest(new AbstractCoreDouble("rint")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.rint(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.rint(handle, n, result, x);
            }
        });
    }

    @Test
    public void testRound()
    {
        runTest(new AbstractCoreDouble("round")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.round(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.round(handle, n, result, x);
            }
        });
    }

    @Test
    public void testRsqrt()
    {
        runTest(new AbstractCoreDouble("rsqrt")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return (1.0 / Math.sqrt(x));
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.rsqrt(handle, n, result, x);
            }
        });
    }

    @Test
    public void testSin()
    {
        runTest(new AbstractCoreDouble("sin")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.sin(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.sin(handle, n, result, x);
            }
        });
    }

    @Test
    public void testSinh()
    {
        runTest(new AbstractCoreDouble("sinh")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.sinh(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.sinh(handle, n, result, x);
            }
        });
    }

    @Test
    public void testSinpi()
    {
        runTest(new AbstractCoreDouble("sinpi")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.sin(x*Math.PI);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.sinpi(handle, n, result, x);
            }
        });
    }

    @Test
    public void testSqrt()
    {
        runTest(new AbstractCoreDouble("sqrt")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.sqrt(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.sqrt(handle, n, result, x);
            }
        });
    }

    @Test
    public void testTan()
    {
        runTest(new AbstractCoreDouble("tan")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.tan(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.tan(handle, n, result, x);
            }
        });
    }

    @Test
    public void testTanh()
    {
        runTest(new AbstractCoreDouble("tanh")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return Math.tanh(x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.tanh(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testTgamma()
//    {
//        runTest(new AbstractTestCoreDouble("tgamma")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.tgamma(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.tgamma(handle, n, result, x);
//            }
//        });
//    }

    @Test
    public void testTrunc()
    {
        runTest(new AbstractCoreDouble("trunc")
        {
            @Override
            protected double computeHostElement(
                double x, double y, double scalar)
            {
                return ((long)x);
            }
             
            @Override
            protected void computeDevice(VecHandle handle,
                long n, Pointer result, 
                Pointer x, Pointer y, double scalar)
            {
                VecDouble.trunc(handle, n, result, x);
            }
        });
    }

//    @Test
//    public void testY0()
//    {
//        runTest(new AbstractTestCoreDouble("y0")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.y0(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.y0(handle, n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testY1()
//    {
//        runTest(new AbstractTestCoreDouble("y1")
//        {
//            @Override
//            protected double computeHostElement(
//                double x, double y, double scalar)
//            {
//                return Math.y1(x);
//            }
//             
//            @Override
//            protected void computeDevice(VecHandle handle,
//                Pointer result, Pointer x, 
//                Pointer y, double scalar, long n)
//            {
//                VecDouble.y1(handle, n, result, x);
//            }
//        });
//    }

    
}
