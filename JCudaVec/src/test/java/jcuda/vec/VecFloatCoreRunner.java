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

import java.util.Random;

import jcuda.Pointer;

/**
 * Utility class to run an {@link AbstractCoreFloat}
 */
class VecFloatCoreRunner
{
    /**
     * Executes the given {@link AbstractCoreFloat} on random host and
     * device data, and compares the results
     * 
     * @param testCore The core to execute
     * @return Whether the results have been equal
     */
    static boolean runTest(AbstractCoreFloat testCore)
    {
        final boolean verbose = false;
        final float epsilon = 1e-5f;
        final int n = 1000;
        final Random random = new Random(0);
        final float hostX[] = TestUtil.createRandomHostDataFloat(n, random);
        final float hostY[] = TestUtil.createRandomHostDataFloat(n, random);
        final float hostResultReference[] = new float[n];
        final float scalar = 0.5f;
        
        VecHandle handle = Vec.createHandle(); 
        
        Pointer deviceX = TestUtil.createDevicePointerFloat(hostX);
        Pointer deviceY = TestUtil.createDevicePointerFloat(hostY);
        Pointer deviceResult = TestUtil.createDevicePointerFloat(n);
        
        testCore.computeHost(n, hostResultReference, hostX, hostY, scalar);
        testCore.computeDevice(
            handle, n, deviceResult, deviceX, deviceY, scalar);
        
        Vec.destroyHandle(handle);

        float hostResult[] = TestUtil.createHostDataFloat(deviceResult, n);
        boolean passed = TestUtil.equalFloat(
            hostResult, hostResultReference, epsilon, verbose);
        
        TestUtil.freeDevicePointer(deviceX);
        TestUtil.freeDevicePointer(deviceY);
        TestUtil.freeDevicePointer(deviceResult);
        
        if (verbose)
        {
            System.out.println(
                String.format("%10s", testCore)+" passed? "+passed);
        }
        return passed;
    }

    /**
     * Private constructor to prevent instantiation
     */
    private VecFloatCoreRunner()
    {
        // Private constructor to prevent instantiation
    }
    
}

