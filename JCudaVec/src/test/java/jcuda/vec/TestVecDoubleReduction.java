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

import static org.junit.Assert.assertEquals;

import java.util.Random;

import org.junit.Test;

import jcuda.Pointer;

/**
 * Tests for the VecDouble reduction methods
 */
@SuppressWarnings("javadoc")
public class TestVecDoubleReduction
{
    @Test
    public void testReduceAdd()
    {
        boolean verbose = false;
        double epsilon = 1e-6f;
        int n = 1000;
        Random random = new Random(0);
        double hostX[] = TestUtil.createRandomHostDataDouble(n, random);
        
        VecHandle handle = Vec.createHandle(); 
        Pointer deviceX = TestUtil.createDevicePointerDouble(hostX);
        
        double deviceResult = VecDouble.reduceAdd(handle, n, deviceX);
        
        Vec.destroyHandle(handle);

        double hostResult = Reductions.reduceAdd(hostX);
        
        TestUtil.freeDevicePointer(deviceX);
        
        if (verbose)
        {
            System.out.println(
                "Add: Expected " + hostResult + ", actual " + deviceResult);
        }
        
        assertEquals(hostResult, deviceResult, epsilon * hostResult);
    }

    @Test
    public void testReduceMul()
    {
        boolean verbose = false;
        double epsilon = 1e-6f;
        int n = 10;
        Random random = new Random(0);
        double hostX[] = TestUtil.createRandomHostDataDouble(n, random);
        
        VecHandle handle = Vec.createHandle(); 
        Pointer deviceX = TestUtil.createDevicePointerDouble(hostX);
        
        double deviceResult = VecDouble.reduceMul(handle, n, deviceX);
        
        Vec.destroyHandle(handle);

        double hostResult = Reductions.reduceMul(hostX);
        
        TestUtil.freeDevicePointer(deviceX);
        
        if (verbose)
        {
            System.out.println(
                "Mul: Expected " + hostResult + ", actual " + deviceResult);
        }
        
        assertEquals(hostResult, deviceResult, epsilon * hostResult);
        
    }

    @Test
    public void testReduceMin()
    {
        boolean verbose = false;
        double epsilon = 1e-6f;
        int n = 1000;
        Random random = new Random(0);
        double hostX[] = TestUtil.createRandomHostDataDouble(n, random);
        
        VecHandle handle = Vec.createHandle(); 
        Pointer deviceX = TestUtil.createDevicePointerDouble(hostX);
        
        double deviceResult = VecDouble.reduceMin(handle, n, deviceX);
        
        Vec.destroyHandle(handle);

        double hostResult = Reductions.reduceMin(hostX);
        
        TestUtil.freeDevicePointer(deviceX);
        
        if (verbose)
        {
            System.out.println(
                "Min: Expected " + hostResult + ", actual " + deviceResult);
        }
        
        assertEquals(hostResult, deviceResult, epsilon * hostResult);
        
    }
    
    @Test
    public void testReduceMax()
    {
        boolean verbose = false;
        double epsilon = 1e-6f;
        int n = 1000;
        Random random = new Random(0);
        double hostX[] = TestUtil.createRandomHostDataDouble(n, random);
        
        VecHandle handle = Vec.createHandle(); 
        Pointer deviceX = TestUtil.createDevicePointerDouble(hostX);
        
        double deviceResult = VecDouble.reduceMax(handle, n, deviceX);
        
        Vec.destroyHandle(handle);

        double hostResult = Reductions.reduceMax(hostX);
        
        TestUtil.freeDevicePointer(deviceX);
        
        if (verbose)
        {
            System.out.println(
                "Max: Expected " + hostResult + ", actual " + deviceResult);
        }
        
        assertEquals(hostResult, deviceResult, epsilon * hostResult);
        
    }

}
