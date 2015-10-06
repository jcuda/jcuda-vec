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

/*
 * NOTE: Many of these tests are commented out, because they don't
 * have a simple translation to java.lang.Math methods. They may
 * be extended in future.
 */

/**
 * Tests for the 1-argument vector math methods
 */
public class TestVecFloatMath1 extends AbstractTestVecFloat
{
    @Test
    public void testAcos()
    {
        runTest(new AbstractCoreFloat("acos")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.acos(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.acos(n, result, x);
            }
        });
    }

//    @Test
//    public void testAcosh()
//    {
//        runTest(new AbstractTestCoreFloat("acosh")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.acosh(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.acosh(n, result, x);
//            }
//        });
//    }

    @Test
    public void testAsin()
    {
        runTest(new AbstractCoreFloat("asin")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.asin(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.asin(n, result, x);
            }
        });
    }

//    @Test
//    public void testAsinh()
//    {
//        runTest(new AbstractTestCoreFloat("asinh")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.asinh(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.asinh(n, result, x);
//            }
//        });
//    }

    @Test
    public void testAtan()
    {
        runTest(new AbstractCoreFloat("atan")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.atan(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.atan(n, result, x);
            }
        });
    }

//    @Test
//    public void testAtanh()
//    {
//        runTest(new AbstractTestCoreFloat("atanh")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.atanh(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.atanh(n, result, x);
//            }
//        });
//    }

    @Test
    public void testCbrt()
    {
        runTest(new AbstractCoreFloat("cbrt")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.cbrt(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.cbrt(n, result, x);
            }
        });
    }

    @Test
    public void testCeil()
    {
        runTest(new AbstractCoreFloat("ceil")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.ceil(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.ceil(n, result, x);
            }
        });
    }

    @Test
    public void testCos()
    {
        runTest(new AbstractCoreFloat("cos")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.cos(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.cos(n, result, x);
            }
        });
    }

    @Test
    public void testCosh()
    {
        runTest(new AbstractCoreFloat("cosh")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.cosh(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.cosh(n, result, x);
            }
        });
    }

    @Test
    public void testCospi()
    {
        runTest(new AbstractCoreFloat("cospi")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.cos(x*Math.PI);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.cospi(n, result, x);
            }
        });
    }

//    @Test
//    public void testErfc()
//    {
//        runTest(new AbstractTestCoreFloat("erfc")
//        {
//            @Override
//            protected float computeHostElement(
//  float x, float y, float scalar)
//            {
//                return (float)Math.erfc(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.erfc(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testErfcinv()
//    {
//        runTest(new AbstractTestCoreFloat("erfcinv")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.erfcinv(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.erfcinv(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testErfcx()
//    {
//        runTest(new AbstractTestCoreFloat("erfcx")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.erfcx(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.erfcx(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testErf()
//    {
//        runTest(new AbstractTestCoreFloat("erf")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.erf(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.erf(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testErfinv()
//    {
//        runTest(new AbstractTestCoreFloat("erfinv")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.erfinv(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.erfinv(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testExp10()
//    {
//        runTest(new AbstractTestCoreFloat("exp10")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.exp10(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.exp10(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testExp2()
//    {
//        runTest(new AbstractTestCoreFloat("exp2")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.exp2(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.exp2(n, result, x);
//            }
//        });
//    }

    @Test
    public void testExp()
    {
        runTest(new AbstractCoreFloat("exp")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.exp(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.exp(n, result, x);
            }
        });
    }

    @Test
    public void testExpm1()
    {
        runTest(new AbstractCoreFloat("expm1")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.expm1(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.expm1(n, result, x);
            }
        });
    }

    @Test
    public void testFabs()
    {
        runTest(new AbstractCoreFloat("fabs")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.abs(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.fabs(n, result, x);
            }
        });
    }

    @Test
    public void testFloor()
    {
        runTest(new AbstractCoreFloat("floor")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.floor(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.floor(n, result, x);
            }
        });
    }

//    @Test
//    public void testJ0()
//    {
//        runTest(new AbstractTestCoreFloat("j0")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.j0(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.j0(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testJ1()
//    {
//        runTest(new AbstractTestCoreFloat("j1")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.j1(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.j1(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testLgamma()
//    {
//        runTest(new AbstractTestCoreFloat("lgamma")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.lgamma(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.lgamma(n, result, x);
//            }
//        });
//    }

    @Test
    public void testLog10()
    {
        runTest(new AbstractCoreFloat("log10")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.log10(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.log10(n, result, x);
            }
        });
    }

    @Test
    public void testLog1p()
    {
        runTest(new AbstractCoreFloat("log1p")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.log1p(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.log1p(n, result, x);
            }
        });
    }

    @Test
    public void testLog2()
    {
        runTest(new AbstractCoreFloat("log2")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)(Math.log(x) / Math.log(2));
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.log2(n, result, x);
            }
        });
    }

//    @Test
//    public void testLogb()
//    {
//        runTest(new AbstractTestCoreFloat("logb")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.logb(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.logb(n, result, x);
//            }
//        });
//    }

    @Test
    public void testLog()
    {
        runTest(new AbstractCoreFloat("log")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.log(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.log(n, result, x);
            }
        });
    }

//    @Test
//    public void testNormcdf()
//    {
//        runTest(new AbstractTestCoreFloat("normcdf")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.normcdf(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.normcdf(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testNormcdfinv()
//    {
//        runTest(new AbstractTestCoreFloat("normcdfinv")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.normcdfinv(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.normcdfinv(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testRcbrt()
//    {
//        runTest(new AbstractTestCoreFloat("rcbrt")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.rcbrt(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.rcbrt(n, result, x);
//            }
//        });
//    }

    @Test
    public void testRint()
    {
        runTest(new AbstractCoreFloat("rint")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.rint(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.rint(n, result, x);
            }
        });
    }

    @Test
    public void testRound()
    {
        runTest(new AbstractCoreFloat("round")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.round(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.round(n, result, x);
            }
        });
    }

    @Test
    public void testRsqrt()
    {
        runTest(new AbstractCoreFloat("rsqrt")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)(1.0 / Math.sqrt(x));
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.rsqrt(n, result, x);
            }
        });
    }

    @Test
    public void testSin()
    {
        runTest(new AbstractCoreFloat("sin")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.sin(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.sin(n, result, x);
            }
        });
    }

    @Test
    public void testSinh()
    {
        runTest(new AbstractCoreFloat("sinh")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.sinh(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.sinh(n, result, x);
            }
        });
    }

    @Test
    public void testSinpi()
    {
        runTest(new AbstractCoreFloat("sinpi")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.sin(x*Math.PI);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.sinpi(n, result, x);
            }
        });
    }

    @Test
    public void testSqrt()
    {
        runTest(new AbstractCoreFloat("sqrt")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.sqrt(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.sqrt(n, result, x);
            }
        });
    }

    @Test
    public void testTan()
    {
        runTest(new AbstractCoreFloat("tan")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.tan(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.tan(n, result, x);
            }
        });
    }

    @Test
    public void testTanh()
    {
        runTest(new AbstractCoreFloat("tanh")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)Math.tanh(x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.tanh(n, result, x);
            }
        });
    }

//    @Test
//    public void testTgamma()
//    {
//        runTest(new AbstractTestCoreFloat("tgamma")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.tgamma(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.tgamma(n, result, x);
//            }
//        });
//    }

    @Test
    public void testTrunc()
    {
        runTest(new AbstractCoreFloat("trunc")
        {
            @Override
            protected float computeHostElement(
                float x, float y, float scalar)
            {
                return (float)((long)x);
            }
             
            @Override
            protected void computeDevice(long n, CUdeviceptr result, 
                CUdeviceptr x, CUdeviceptr y, float scalar)
            {
                VecFloat.trunc(n, result, x);
            }
        });
    }

//    @Test
//    public void testY0()
//    {
//        runTest(new AbstractTestCoreFloat("y0")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.y0(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.y0(n, result, x);
//            }
//        });
//    }

//    @Test
//    public void testY1()
//    {
//        runTest(new AbstractTestCoreFloat("y1")
//        {
//            @Override
//            protected float computeHostElement(
//                float x, float y, float scalar)
//            {
//                return (float)Math.y1(x);
//            }
//             
//            @Override
//            protected void computeDevice(CUdeviceptr result, CUdeviceptr x, 
//                CUdeviceptr y, float scalar, long n)
//            {
//                VecFloat.y1(n, result, x);
//            }
//        });
//    }

    
}
