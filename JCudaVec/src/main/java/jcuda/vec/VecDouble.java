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

import jcuda.CudaException;
import jcuda.Pointer;

/**
 * Vector operations for double-precision floating point vectors. <br>
 * <br>
 * Before any of the vector methods of this class can be called, it
 * has to be initialized by calling {@link #init()}. In order to 
 * release all resources allocated by this class, {@link #shutdown()}
 * has to be called.
 * <br>
 * NOTE: This class only forms a thin layer around the actual CUDA 
 * kernel calls. It can not verify the function parameters. The caller
 * is responsible for giving appropriate pointers and vector sizes.
 */
public class VecDouble 
{
    /**
     * The {@link VecKernels} that maintains the kernels that can be
     * called via the methods of this class.
     */
    private static VecKernels vecKernels = null;
    
    /**
     * Private constructor to prevent instantiation
     */
    private VecDouble()
    {
        // Private constructor to prevent instantiation
    }
    
    /**
     * Initialize this class. This method has to be called before any
     * of the vector operation methods of this class can be called.
     * The resources that are allocated by this call may be freed
     * by calling {@link #shutdown()}.
     * 
     * @throws CudaException If the class can not be initialized. 
     * Reasons for this may be (but are not limited to) : <br>
     * <ul>
     *   <li> 
     *     It is neither possible to attach to an existing context, 
     *     nor to create a new context.
     *   </li>
     *   <li> 
     *     The resource that contains the kernels (for example, a
     *     PTX file) can not be loaded
     *   </li>
     * </ul>
     */
    public static void init()
    {
        shutdown();
        String kernelNamePrefix = "vec_";
        String kernelNameType = "double";
        String kernelNameSuffix = "";
        vecKernels = new DefaultVecKernels(
            kernelNameType, kernelNamePrefix, kernelNameSuffix);
    }
    
    
    
    /**
     * Perform a shutdown and release all resources allocated by this class.
     */
    public static void shutdown()
    {
        if (vecKernels != null)
        {
            vecKernels.shutdown();
            vecKernels = null;
        }
    }
    
    /**
     * Passes the call to the {@link VecKernels#call(String, Object...)}
     * 
     * @param name The kernel name
     * @param workSize The work size for the kernel call
     * @param arguments The kernel arguments (including the vector size)
     */
    private static void call(String name, long workSize, Object ... arguments)
    {
        if (vecKernels == null)
        {
            throw new CudaException(
                "Kernels not initialized. Call init() first.");
        }
        vecKernels.call(name, workSize, arguments);
    }
    
    
    
    /**
     * Set all elements of the given vector to the given value
     * 
     * @param n The size of the vector
     * @param result The vector that will store the result
     * @param value The value to set
     */
    public static void set(long n, Pointer result, double value)
    {
        call("set", n, n, result, value);
    }
    
    //=== Vector arithmetic ==================================================
    
    /**
     * Add the given vectors.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector
     */
    public static void add(long n, Pointer result, Pointer x, Pointer y)
    {
        call("add", n, n, result, x, y);
    }

    /**
     * Subtract the given vectors.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector
     */
    public static void sub(long n, Pointer result, Pointer x, Pointer y)
    {
        call("sub", n, n, result, x, y);
    }

    /**
     * Multiply the given vectors.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector
     */
    public static void mul(long n, Pointer result, Pointer x, Pointer y)
    {
        call("mul", n, n, result, x, y);
    }

    /**
     * Divide the given vectors.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector
     */
    public static void div(long n, Pointer result, Pointer x, Pointer y)
    {
        call("div", n, n, result, x, y);
    }

    /**
     * Negate the given vector.
     * @param n The size of the vectors
     * 
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void negate(long n, Pointer result, Pointer x)
    {
        call("negate", n, n, result, x);
    }
    
    //=== Vector-and-scalar arithmetic =======================================
    
    /**
     * Add the given scalar to the given vector.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The scalar
     */
    public static void addScalar(long n, Pointer result, Pointer x, double y)
    {
        call("addScalar", n, n, result, x, y);
    }

    /**
     * Subtract the given scalar from the given vector.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The scalar
     */
    public static void subScalar(long n, Pointer result, Pointer x, double y)
    {
        call("subScalar", n, n, result, x, y);
    }

    /**
     * Multiply the given vector with the given scalar.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The scalar
     */
    public static void mulScalar(long n, Pointer result, Pointer x, double y)
    {
        call("mulScalar", n, n, result, x, y);
    }

    /**
     * Divide the given vector by the given scalar.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The scalar
     */
    public static void divScalar(long n, Pointer result, Pointer x, double y)
    {
        call("divScalar", n, n, result, x, y);
    }
    
    
    /**
     * Add the given vector to the given scalar.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The scalar
     * @param y The vector
     */
    public static void scalarAdd(long n, Pointer result, double x, Pointer y)
    {
        call("scalarAdd", n, n, result, x, y);
    }

    /**
     * Subtract the given vector from the given scalar.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The scalar
     * @param y The vector
     */
    public static void scalarSub(long n, Pointer result, double x, Pointer y)
    {
        call("scalarSub", n, n, result, x, y);
    }

    /**
     * Multiply the given scalar with the given vector.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The scalar
     * @param y The vector
     */
    public static void scalarMul(long n, Pointer result, double x, Pointer y)
    {
        call("scalarMul", n, n, result, x, y);
    }

    /**
     * Divide the given scalar by the given vector.
     * 
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The scalar
     * @param y The vector
     */
    public static void scalarDiv(long n, Pointer result, double x, Pointer y)
    {
        call("scalarDiv", n, n, result, x, y);
    }
    
    
    //=== Vector comparison ==================================================
    
    /**
     * Perform a '&lt;' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void lt(long n, Pointer result, Pointer x, Pointer y)
    {
        call("lt", n, n, result, x, y);
    }

    /**
     * Perform a '&lt;=' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void lte(long n, Pointer result, Pointer x, Pointer y)
    {
        call("lte", n, n, result, x, y);
    }

    /**
     * Perform a '==' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void eq(long n, Pointer result, Pointer x, Pointer y)
    {
        call("eq", n, n, result, x, y);
    }

    /**
     * Perform a '&gt;=' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void gte(long n, Pointer result, Pointer x, Pointer y)
    {
        call("gte", n, n, result, x, y);
    }

    /**
     * Perform a '&gt;' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void gt(long n, Pointer result, Pointer x, Pointer y)
    {
        call("gt", n, n, result, x, y);
    }
    
    /**
     * Perform a '!=' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void ne(long n, Pointer result, Pointer x, Pointer y)
    {
        call("ne", n, n, result, x, y);
    }
    
    
    //=== Vector-and-scalar comparison =======================================

    /**
     * Perform a '&lt;' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void ltScalar(long n, Pointer result, Pointer x, double y)
    {
        call("ltScalar", n, n, result, x, y);
    }

    /**
     * Perform a '&lt;=' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void lteScalar(long n, Pointer result, Pointer x, double y)
    {
        call("lteScalar", n, n, result, x, y);
    }

    /**
     * Perform a '==' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void eqScalar(long n, Pointer result, Pointer x, double y)
    {
        call("eqScalar", n, n, result, x, y);
    }

    /**
     * Perform a '&gt;=' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void gteScalar(long n, Pointer result, Pointer x, double y)
    {
        call("gteScalar", n, n, result, x, y);
    }

    /**
     * Perform a '&gt;' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void gtScalar(long n, Pointer result, Pointer x, double y)
    {
        call("gtScalar", n, n, result, x, y);
    }
    
    /**
     * Perform a '!=' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void neScalar(long n, Pointer result, Pointer x, double y)
    {
        call("neScalar", n, n, result, x, y);
    }

    
    //=== Vector math (one argument) =========================================
    

    /**
     * Calculate the arc cosine of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void acos(long n, Pointer result, Pointer x)
    {
        call("acos", n, n, result, x);
    }

    /**
     * Calculate the nonnegative arc hyperbolic cosine of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void acosh(long n, Pointer result, Pointer x)
    {
        call("acosh", n, n, result, x);
    }

    /**
     * Calculate the arc sine of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void asin(long n, Pointer result, Pointer x)
    {
        call("asin", n, n, result, x);
    }

    /**
     * Calculate the arc hyperbolic sine of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void asinh(long n, Pointer result, Pointer x)
    {
        call("asinh", n, n, result, x);
    }

    /**
     * Calculate the arc tangent of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void atan(long n, Pointer result, Pointer x)
    {
        call("atan", n, n, result, x);
    }

    /**
     * Calculate the arc hyperbolic tangent of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void atanh(long n, Pointer result, Pointer x)
    {
        call("atanh", n, n, result, x);
    }

    /**
     * Calculate the cube root of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void cbrt(long n, Pointer result, Pointer x)
    {
        call("cbrt", n, n, result, x);
    }

    /**
     * Calculate ceiling of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void ceil(long n, Pointer result, Pointer x)
    {
        call("ceil", n, n, result, x);
    }

    /**
     * Calculate the cosine of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void cos(long n, Pointer result, Pointer x)
    {
        call("cos", n, n, result, x);
    }

    /**
     * Calculate the hyperbolic cosine of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void cosh(long n, Pointer result, Pointer x)
    {
        call("cosh", n, n, result, x);
    }

    /**
     * Calculate the cosine of the input argument times pi
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void cospi(long n, Pointer result, Pointer x)
    {
        call("cospi", n, n, result, x);
    }

    /**
     * Calculate the complementary error function of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erfc(long n, Pointer result, Pointer x)
    {
        call("erfc", n, n, result, x);
    }

    /**
     * Calculate the inverse complementary error function of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erfcinv(long n, Pointer result, Pointer x)
    {
        call("erfcinv", n, n, result, x);
    }

    /**
     * Calculate the scaled complementary error function of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erfcx(long n, Pointer result, Pointer x)
    {
        call("erfcx", n, n, result, x);
    }

    /**
     * Calculate the error function of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erf(long n, Pointer result, Pointer x)
    {
        call("erf", n, n, result, x);
    }

    /**
     * Calculate the inverse error function of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erfinv(long n, Pointer result, Pointer x)
    {
        call("erfinv", n, n, result, x);
    }

    /**
     * Calculate the base 10 exponential of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void exp10(long n, Pointer result, Pointer x)
    {
        call("exp10", n, n, result, x);
    }

    /**
     * Calculate the base 2 exponential of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void exp2(long n, Pointer result, Pointer x)
    {
        call("exp2", n, n, result, x);
    }

    /**
     * Calculate the base e exponential of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void exp(long n, Pointer result, Pointer x)
    {
        call("exp", n, n, result, x);
    }

    /**
     * Calculate the base e exponential of the input argument, minus 1.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void expm1(long n, Pointer result, Pointer x)
    {
        call("expm1", n, n, result, x);
    }

    /**
     * Calculate the absolute value of its argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void fabs(long n, Pointer result, Pointer x)
    {
        call("fabs", n, n, result, x);
    }

    /**
     * Calculate the largest integer less than or equal to x.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void floor(long n, Pointer result, Pointer x)
    {
        call("floor", n, n, result, x);
    }

    /**
     * Calculate the value of the Bessel function of the first kind of 
     * order 0 for the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void j0(long n, Pointer result, Pointer x)
    {
        call("j0", n, n, result, x);
    }

    /**
     * Calculate the value of the Bessel function of the first kind of 
     * order 1 for the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void j1(long n, Pointer result, Pointer x)
    {
        call("j1", n, n, result, x);
    }

    /**
     * Calculate the natural logarithm of the absolute value of the gamma 
     * function of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void lgamma(long n, Pointer result, Pointer x)
    {
        call("lgamma", n, n, result, x);
    }

    /**
     * Calculate the base 10 logarithm of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void log10(long n, Pointer result, Pointer x)
    {
        call("log10", n, n, result, x);
    }

    /**
     * Calculate the value of l o g e ( 1 + x ) .
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void log1p(long n, Pointer result, Pointer x)
    {
        call("log1p", n, n, result, x);
    }

    /**
     * Calculate the base 2 logarithm of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void log2(long n, Pointer result, Pointer x)
    {
        call("log2", n, n, result, x);
    }

    /**
     * Calculate the floating point representation of the exponent of the 
     * input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void logb(long n, Pointer result, Pointer x)
    {
        call("logb", n, n, result, x);
    }

    /**
     * Calculate the natural logarithm of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void log(long n, Pointer result, Pointer x)
    {
        call("log", n, n, result, x);
    }

    /**
     * Calculate the standard normal cumulative distribution function.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void normcdf(long n, Pointer result, Pointer x)
    {
        call("normcdf", n, n, result, x);
    }

    /**
     * Calculate the inverse of the standard normal cumulative distribution 
     * function.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void normcdfinv(long n, Pointer result, Pointer x)
    {
        call("normcdfinv", n, n, result, x);
    }

    /**
     * Calculate reciprocal cube root function.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void rcbrt(long n, Pointer result, Pointer x)
    {
        call("rcbrt", n, n, result, x);
    }

    /**
     * Round input to nearest integer value in floating-point.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void rint(long n, Pointer result, Pointer x)
    {
        call("rint", n, n, result, x);
    }

    /**
     * Round to nearest integer value in floating-point.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void round(long n, Pointer result, Pointer x)
    {
        call("round", n, n, result, x);
    }

    /**
     * Calculate the reciprocal of the square root of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void rsqrt(long n, Pointer result, Pointer x)
    {
        call("rsqrt", n, n, result, x);
    }

    /**
     * Calculate the sine of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void sin(long n, Pointer result, Pointer x)
    {
        call("sin", n, n, result, x);
    }

    /**
     * Calculate the hyperbolic sine of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void sinh(long n, Pointer result, Pointer x)
    {
        call("sinh", n, n, result, x);
    }

    /**
     * Calculate the sine of the input argument times pi
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void sinpi(long n, Pointer result, Pointer x)
    {
        call("sinpi", n, n, result, x);
    }

    /**
     * Calculate the square root of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void sqrt(long n, Pointer result, Pointer x)
    {
        call("sqrt", n, n, result, x);
    }

    /**
     * Calculate the tangent of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void tan(long n, Pointer result, Pointer x)
    {
        call("tan", n, n, result, x);
    }

    /**
     * Calculate the hyperbolic tangent of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void tanh(long n, Pointer result, Pointer x)
    {
        call("tanh", n, n, result, x);
    }

    /**
     * Calculate the gamma function of the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void tgamma(long n, Pointer result, Pointer x)
    {
        call("tgamma", n, n, result, x);
    }

    /**
     * Truncate input argument to the integral part.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void trunc(long n, Pointer result, Pointer x)
    {
        call("trunc", n, n, result, x);
    }

    /**
     * Calculate the value of the Bessel function of the second kind of 
     * order 0 for the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void y0(long n, Pointer result, Pointer x)
    {
        call("y0", n, n, result, x);
    }

    /**
     * Calculate the value of the Bessel function of the second kind of 
     * order 1 for the input argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void y1(long n, Pointer result, Pointer x)
    {
        call("y1", n, n, result, x);
    }

    //=== Vector math (two arguments) ========================================
    


    /**
     * Create value with given magnitude, copying sign of second value.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void copysign(long n, Pointer result, Pointer x, Pointer y)
    {
        call("copysign", n, n, result, x, y);
    }

    /**
     * Compute the positive difference between x and y.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fdim(long n, Pointer result, Pointer x, Pointer y)
    {
        call("fdim", n, n, result, x, y);
    }

    /**
     * Divide two floating point values.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fdivide(long n, Pointer result, Pointer x, Pointer y)
    {
        call("fdivide", n, n, result, x, y);
    }

    /**
     * Determine the maximum numeric value of the arguments.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fmax(long n, Pointer result, Pointer x, Pointer y)
    {
        call("fmax", n, n, result, x, y);
    }

    /**
     * Determine the minimum numeric value of the arguments.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fmin(long n, Pointer result, Pointer x, Pointer y)
    {
        call("fmin", n, n, result, x, y);
    }

    /**
     * Calculate the floating-point remainder of x / y.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fmod(long n, Pointer result, Pointer x, Pointer y)
    {
        call("fmod", n, n, result, x, y);
    }

    /**
     * Calculate the square root of the sum of squares of two arguments.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void hypot(long n, Pointer result, Pointer x, Pointer y)
    {
        call("hypot", n, n, result, x, y);
    }

    /**
     * Return next representable single-precision floating-point value 
     * after argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void nextafter(long n, Pointer result, Pointer x, Pointer y)
    {
        call("nextafter", n, n, result, x, y);
    }

    /**
     * Calculate the value of first argument to the power of second argument.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void pow(long n, Pointer result, Pointer x, Pointer y)
    {
        call("pow", n, n, result, x, y);
    }

    /**
     * Compute single-precision floating-point remainder.
     *
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void remainder(long n, Pointer result, Pointer x, Pointer y)
    {
        call("remainder", n, n, result, x, y);
    }

    
    
}
