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

import jcuda.Pointer;

/**
 * Vector operations for double-precision floating point vectors. <br>
 * <br>
 * The methods in this class that perform vector operations expect a
 * {@link VecHandle} as their first parameter. Such a handle can be
 * created by calling {@link Vec#createHandle()}. The semantics of such
 * a handle are equivalent to that of other runtime libraries, for 
 * example, a <code>cublasHandle</code>.
 * <br>
 * NOTE: This class only forms a thin layer around the actual CUDA 
 * kernel calls. It can not verify the function parameters. The caller
 * is responsible for giving appropriate pointers and vector sizes.
 */
public class VecDouble 
{
    /**
     * Private constructor to prevent instantiation
     */
    private VecDouble()
    {
        // Private constructor to prevent instantiation
    }

    /**
     * Performs a "+"-reduction of the elements in the given vector
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vector
     * @param x The input vector
     * @return The reduction result
     */
    public static double reduceAdd(VecHandle handle, long n, Pointer x)
    {
        handle.validate();
        VecReductions vecReductions = handle.getVecReductionsDouble();
        VecReduction vecReduction = vecReductions.get("add");
        double result[] = { 0.0f };
        vecReduction.reduce(n, x, Pointer.to(result));
        return result[0];
    }

    /**
     * Performs a "*"-reduction of the elements in the given vector
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vector
     * @param x The input vector
     * @return The reduction result
     */
    public static double reduceMul(VecHandle handle, long n, Pointer x)
    {
        handle.validate();
        VecReductions vecReductions = handle.getVecReductionsDouble();
        VecReduction vecReduction = vecReductions.get("mul");
        double result[] = { 1.0f };
        vecReduction.reduce(n, x, Pointer.to(result));
        return result[0];
    }

    /**
     * Performs a min-reduction of the elements in the given vector
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vector
     * @param x The input vector
     * @return The reduction result
     */
    public static double reduceMin(VecHandle handle, long n, Pointer x)
    {
        handle.validate();
        VecReductions vecReductions = handle.getVecReductionsDouble();
        VecReduction vecReduction = vecReductions.get("min");
        double result[] = { Double.MAX_VALUE };
        vecReduction.reduce(n, x, Pointer.to(result));
        return result[0];
    }

    /**
     * Performs a max-reduction of the elements in the given vector
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vector
     * @param x The input vector
     * @return The reduction result
     */
    public static double reduceMax(VecHandle handle, long n, Pointer x)
    {
        handle.validate();
        VecReductions vecReductions = handle.getVecReductionsDouble();
        VecReduction vecReduction = vecReductions.get("max");
        double result[] = { -Double.MAX_VALUE };
        vecReduction.reduce(n, x, Pointer.to(result));
        return result[0];
    }

    
    /**
     * Passes the call to {@link VecKernels#call(String, long, Object...)}
     * 
     * @param handle The {@link VecHandle}
     * @param name The kernel name
     * @param workSize The work size for the kernel call
     * @param arguments The kernel arguments (including the vector size)
     */
    private static void call(
        VecHandle handle, String name, long workSize, Object ... arguments)
    {
        handle.validate();
        VecKernels vecKernels = handle.getVecKernelsDouble();
        vecKernels.call(name, workSize, arguments);
    }
    
    
    /**
     * Set all elements of the given vector to the given value
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vector
     * @param result The vector that will store the result
     * @param value The value to set
     */
    public static void set(VecHandle handle, long n, Pointer result, double value)
    {
        call(handle, "set", n, n, result, value);
    }
    
    //=== Vector arithmetic ==================================================
    
    /**
     * Add the given vectors.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector
     */
    public static void add(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "add", n, n, result, x, y);
    }

    /**
     * Subtract the given vectors.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector
     */
    public static void sub(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "sub", n, n, result, x, y);
    }

    /**
     * Multiply the given vectors.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector
     */
    public static void mul(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "mul", n, n, result, x, y);
    }

    /**
     * Divide the given vectors.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector
     */
    public static void div(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "div", n, n, result, x, y);
    }

    /**
     * Negate the given vector.
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * 
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void negate(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "negate", n, n, result, x);
    }
    
    //=== Vector-and-scalar arithmetic =======================================
    
    /**
     * Add the given scalar to the given vector.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The scalar
     */
    public static void addScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "addScalar", n, n, result, x, y);
    }

    /**
     * Subtract the given scalar from the given vector.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The scalar
     */
    public static void subScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "subScalar", n, n, result, x, y);
    }

    /**
     * Multiply the given vector with the given scalar.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The scalar
     */
    public static void mulScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "mulScalar", n, n, result, x, y);
    }

    /**
     * Divide the given vector by the given scalar.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The scalar
     */
    public static void divScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "divScalar", n, n, result, x, y);
    }
    
    
    /**
     * Add the given vector to the given scalar.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The scalar
     * @param y The vector
     */
    public static void scalarAdd(VecHandle handle, long n, Pointer result, double x, Pointer y)
    {
        call(handle, "scalarAdd", n, n, result, x, y);
    }

    /**
     * Subtract the given vector from the given scalar.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The scalar
     * @param y The vector
     */
    public static void scalarSub(VecHandle handle, long n, Pointer result, double x, Pointer y)
    {
        call(handle, "scalarSub", n, n, result, x, y);
    }

    /**
     * Multiply the given scalar with the given vector.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The scalar
     * @param y The vector
     */
    public static void scalarMul(VecHandle handle, long n, Pointer result, double x, Pointer y)
    {
        call(handle, "scalarMul", n, n, result, x, y);
    }

    /**
     * Divide the given scalar by the given vector.
     * 
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The scalar
     * @param y The vector
     */
    public static void scalarDiv(VecHandle handle, long n, Pointer result, double x, Pointer y)
    {
        call(handle, "scalarDiv", n, n, result, x, y);
    }
    
    
    //=== Vector comparison ==================================================
    
    /**
     * Perform a '&lt;' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void lt(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "lt", n, n, result, x, y);
    }

    /**
     * Perform a '&lt;=' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void lte(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "lte", n, n, result, x, y);
    }

    /**
     * Perform a '==' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void eq(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "eq", n, n, result, x, y);
    }

    /**
     * Perform a '&gt;=' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void gte(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "gte", n, n, result, x, y);
    }

    /**
     * Perform a '&gt;' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void gt(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "gt", n, n, result, x, y);
    }
    
    /**
     * Perform a '!=' comparison of the given vectors. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The second vector
     */
    public static void ne(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "ne", n, n, result, x, y);
    }
    
    
    //=== Vector-and-scalar comparison =======================================

    /**
     * Perform a '&lt;' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void ltScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "ltScalar", n, n, result, x, y);
    }

    /**
     * Perform a '&lt;=' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void lteScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "lteScalar", n, n, result, x, y);
    }

    /**
     * Perform a '==' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void eqScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "eqScalar", n, n, result, x, y);
    }

    /**
     * Perform a '&gt;=' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void gteScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "gteScalar", n, n, result, x, y);
    }

    /**
     * Perform a '&gt;' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void gtScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "gtScalar", n, n, result, x, y);
    }
    
    /**
     * Perform a '!=' comparison of the given vector and scalar. 
     * The result will be set to <code>1.0f</code> where the comparison yields 
     * <code>true</code>, and to <code>0.0f</code> where the comparison yields 
     * <code>false</code>.
     *  
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result.
     * @param x The first vector
     * @param y The scalar
     */
    public static void neScalar(VecHandle handle, long n, Pointer result, Pointer x, double y)
    {
        call(handle, "neScalar", n, n, result, x, y);
    }

    
    //=== Vector math (one argument) =========================================
    

    /**
     * Calculate the arc cosine of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void acos(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "acos", n, n, result, x);
    }

    /**
     * Calculate the nonnegative arc hyperbolic cosine of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void acosh(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "acosh", n, n, result, x);
    }

    /**
     * Calculate the arc sine of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void asin(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "asin", n, n, result, x);
    }

    /**
     * Calculate the arc hyperbolic sine of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void asinh(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "asinh", n, n, result, x);
    }

    /**
     * Calculate the arc tangent of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void atan(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "atan", n, n, result, x);
    }

    /**
     * Calculate the arc hyperbolic tangent of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void atanh(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "atanh", n, n, result, x);
    }

    /**
     * Calculate the cube root of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void cbrt(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "cbrt", n, n, result, x);
    }

    /**
     * Calculate ceiling of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void ceil(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "ceil", n, n, result, x);
    }

    /**
     * Calculate the cosine of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void cos(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "cos", n, n, result, x);
    }

    /**
     * Calculate the hyperbolic cosine of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void cosh(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "cosh", n, n, result, x);
    }

    /**
     * Calculate the cosine of the input argument times pi
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void cospi(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "cospi", n, n, result, x);
    }

    /**
     * Calculate the complementary error function of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erfc(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "erfc", n, n, result, x);
    }

    /**
     * Calculate the inverse complementary error function of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erfcinv(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "erfcinv", n, n, result, x);
    }

    /**
     * Calculate the scaled complementary error function of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erfcx(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "erfcx", n, n, result, x);
    }

    /**
     * Calculate the error function of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erf(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "erf", n, n, result, x);
    }

    /**
     * Calculate the inverse error function of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void erfinv(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "erfinv", n, n, result, x);
    }

    /**
     * Calculate the base 10 exponential of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void exp10(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "exp10", n, n, result, x);
    }

    /**
     * Calculate the base 2 exponential of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void exp2(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "exp2", n, n, result, x);
    }

    /**
     * Calculate the base e exponential of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void exp(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "exp", n, n, result, x);
    }

    /**
     * Calculate the base e exponential of the input argument, minus 1.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void expm1(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "expm1", n, n, result, x);
    }

    /**
     * Calculate the absolute value of its argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void fabs(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "fabs", n, n, result, x);
    }

    /**
     * Calculate the largest integer less than or equal to x.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void floor(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "floor", n, n, result, x);
    }

    /**
     * Calculate the value of the Bessel function of the first kind of 
     * order 0 for the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void j0(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "j0", n, n, result, x);
    }

    /**
     * Calculate the value of the Bessel function of the first kind of 
     * order 1 for the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void j1(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "j1", n, n, result, x);
    }

    /**
     * Calculate the natural logarithm of the absolute value of the gamma 
     * function of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void lgamma(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "lgamma", n, n, result, x);
    }

    /**
     * Calculate the base 10 logarithm of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void log10(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "log10", n, n, result, x);
    }

    /**
     * Calculate the value of l o g e ( 1 + x ) .
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void log1p(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "log1p", n, n, result, x);
    }

    /**
     * Calculate the base 2 logarithm of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void log2(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "log2", n, n, result, x);
    }

    /**
     * Calculate the floating point representation of the exponent of the 
     * input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void logb(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "logb", n, n, result, x);
    }

    /**
     * Calculate the natural logarithm of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void log(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "log", n, n, result, x);
    }

    /**
     * Calculate the standard normal cumulative distribution function.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void normcdf(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "normcdf", n, n, result, x);
    }

    /**
     * Calculate the inverse of the standard normal cumulative distribution 
     * function.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void normcdfinv(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "normcdfinv", n, n, result, x);
    }

    /**
     * Calculate reciprocal cube root function.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void rcbrt(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "rcbrt", n, n, result, x);
    }

    /**
     * Round input to nearest integer value in floating-point.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void rint(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "rint", n, n, result, x);
    }

    /**
     * Round to nearest integer value in floating-point.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void round(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "round", n, n, result, x);
    }

    /**
     * Calculate the reciprocal of the square root of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void rsqrt(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "rsqrt", n, n, result, x);
    }

    /**
     * Calculate the sine of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void sin(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "sin", n, n, result, x);
    }

    /**
     * Calculate the hyperbolic sine of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void sinh(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "sinh", n, n, result, x);
    }

    /**
     * Calculate the sine of the input argument times pi
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void sinpi(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "sinpi", n, n, result, x);
    }

    /**
     * Calculate the square root of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void sqrt(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "sqrt", n, n, result, x);
    }

    /**
     * Calculate the tangent of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void tan(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "tan", n, n, result, x);
    }

    /**
     * Calculate the hyperbolic tangent of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void tanh(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "tanh", n, n, result, x);
    }

    /**
     * Calculate the gamma function of the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void tgamma(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "tgamma", n, n, result, x);
    }

    /**
     * Truncate input argument to the integral part.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void trunc(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "trunc", n, n, result, x);
    }

    /**
     * Calculate the value of the Bessel function of the second kind of 
     * order 0 for the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void y0(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "y0", n, n, result, x);
    }

    /**
     * Calculate the value of the Bessel function of the second kind of 
     * order 1 for the input argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     */
    public static void y1(VecHandle handle, long n, Pointer result, Pointer x)
    {
        call(handle, "y1", n, n, result, x);
    }

    //=== Vector math (two arguments) ========================================
    


    /**
     * Create value with given magnitude, copying sign of second value.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void copysign(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "copysign", n, n, result, x, y);
    }

    /**
     * Compute the positive difference between x and y.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fdim(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "fdim", n, n, result, x, y);
    }

    /**
     * Divide two floating point values.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fdivide(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "fdivide", n, n, result, x, y);
    }

    /**
     * Determine the maximum numeric value of the arguments.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fmax(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "fmax", n, n, result, x, y);
    }

    /**
     * Determine the minimum numeric value of the arguments.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fmin(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "fmin", n, n, result, x, y);
    }

    /**
     * Calculate the floating-point remainder of x / y.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void fmod(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "fmod", n, n, result, x, y);
    }

    /**
     * Calculate the square root of the sum of squares of two arguments.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void hypot(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "hypot", n, n, result, x, y);
    }

    /**
     * Return next representable single-precision floating-point value 
     * after argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void nextafter(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "nextafter", n, n, result, x, y);
    }

    /**
     * Calculate the value of first argument to the power of second argument.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void pow(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "pow", n, n, result, x, y);
    }

    /**
     * Compute single-precision floating-point remainder.
     *
     * @param handle The {@link VecHandle}
     * @param n The size of the vectors
     * @param result The vector that will store the result
     * @param x The first vector
     * @param y The second vector 
     */
    public static void remainder(VecHandle handle, long n, Pointer result, Pointer x, Pointer y)
    {
        call(handle, "remainder", n, n, result, x, y);
    }

    
    
}
