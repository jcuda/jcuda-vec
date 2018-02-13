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

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;

import jcuda.Pointer;

/**
 * A test that simply calls all vector operation methods of the JCudaVec
 * classes via reflection, to test whether they are compiled properly 
 * and found in the module.
 */
public class VecCaller
{
    /**
     * Calls all declared methods in the given class
     *  
     * @param c The class
     * @return Whether the calls succeeded 
     */
    static boolean testCalls(Class<?> c)
    {
        final boolean verbose = false;
        final int n = 1000;
        final float scalar = 0.5f;
        final Pointer devicePointer = TestUtil.createDevicePointerDouble(n);
        final VecHandle handle = Vec.createHandle();
        
        Set<String> excludedMethods = new LinkedHashSet<String>(Arrays.asList(
            "call"
        ));
        
        boolean passed = true;
        for (Method method : c.getDeclaredMethods())
        {
            String name = method.getName();
            if (excludedMethods.contains(name))
            {
                continue;
            }
            
            boolean parametersValid = true;
            Class<?>[] parameterTypes = method.getParameterTypes();
            Object parameters[] = new Object[parameterTypes.length];
            for (int i=0; i<parameters.length; i++)
            {
                if (parameterTypes[i] == VecHandle.class)
                {
                    parameters[i] = handle;
                }
                else if (parameterTypes[i] == Pointer.class)
                {
                    parameters[i] = devicePointer;
                }
                else if (parameterTypes[i] == float.class)
                {
                    parameters[i] = scalar;
                }
                else if (parameterTypes[i] == double.class)
                {
                    parameters[i] = (double)scalar;
                }
                else if (parameterTypes[i] == long.class)
                {
                    parameters[i] = (long)n;
                }
                else
                {
                    System.err.println(
                        "Unexpected parameter type in " + method);
                    parametersValid = false;
                    passed = false;
                }
            }

            if (parametersValid)
            {
                if (verbose)
                {
                    System.out.println("Call " + method);
                }
                try
                {
                    method.invoke(null, parameters);
                }
                catch (IllegalAccessException e)
                {
                    e.printStackTrace();
                    passed = false;
                }
                catch (IllegalArgumentException e)
                {
                    e.printStackTrace();
                    passed = false;
                }
                catch (InvocationTargetException e)
                {
                    e.printStackTrace();
                    passed = false;
                }
            }
            if (verbose)
            {
                System.out.println("Call for " + method + " passed? " + passed);
            }
        }
        
        TestUtil.freeDevicePointer(devicePointer);
        return passed;
    }
}
