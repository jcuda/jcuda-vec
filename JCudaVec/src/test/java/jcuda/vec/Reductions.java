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

/**
 * Host implementations of different reductions, as comparison values for tests
 */
class Reductions
{
    /**
     * Implementation of a Kahan summation reduction in plain Java
     * 
     * @param data The input
     * @return The reduction result
     */
    static float reduceAdd(float data[])
    {
        float sum = data[0];
        float c = 0.0f;
        for (int i = 1; i < data.length; i++)
        {
            float y = data[i] - c;
            float t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }
    
    /**
     * Implementation of a multiplication reduction in plain Java
     * 
     * @param data The input
     * @return The reduction result
     */
    static float reduceMul(float data[])
    {
        float product = 1.0f;
        for (float f : data)
        {
            product *= f;
        }
        return product;
    }

    /**
     * Implementation of a minimum reduction in plain Java
     * 
     * @param data The input
     * @return The reduction result
     */
    static float reduceMin(float data[])
    {
        float result = Float.MAX_VALUE;
        for (float f : data)
        {
            result = Math.min(result, f);
        }
        return result;
    }

    /**
     * Implementation of a maximum reduction in plain Java
     * 
     * @param data The input
     * @return The reduction result
     */
    static float reduceMax(float data[])
    {
        float result = -Float.MAX_VALUE;
        for (float f : data)
        {
            result = Math.max(result, f);
        }
        return result;
    }


    /**
     * Implementation of a Kahan summation reduction in plain Java
     * 
     * @param data The input
     * @return The reduction result
     */
    static double reduceAdd(double data[])
    {
        double sum = data[0];
        double c = 0.0f;
        for (int i = 1; i < data.length; i++)
        {
            double y = data[i] - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        return sum;
    }

    /**
     * Implementation of a multiplication reduction in plain Java
     * 
     * @param data The input
     * @return The reduction result
     */
    static double reduceMul(double data[])
    {
        double product = 1.0f;
        for (double f : data)
        {
            product *= f;
        }
        return product;
    }

    /**
     * Implementation of a minimum reduction in plain Java
     * 
     * @param data The input
     * @return The reduction result
     */
    static double reduceMin(double data[])
    {
        double result = Double.MAX_VALUE;
        for (double f : data)
        {
            result = Math.min(result, f);
        }
        return result;
    }

    /**
     * Implementation of a maximum reduction in plain Java
     * 
     * @param data The input
     * @return The reduction result
     */
    static double reduceMax(double data[])
    {
        double result = -Double.MAX_VALUE;
        for (double f : data)
        {
            result = Math.max(result, f);
        }
        return result;
    }
    
}
