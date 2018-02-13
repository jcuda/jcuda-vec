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

import java.util.LinkedHashMap;
import java.util.Map;

import jcuda.driver.CUcontext;

/**
 * A class summarizing several {@link VecReduction} instances that allow
 * different reduction operations.
 */
class VecReductions
{
    /**
     * The mapping from reduction operator names to {@link VecReduction} 
     * instances
     */
    private final Map<String, VecReduction> vecReductions;
    
    /**
     * Creates a new instance
     * 
     * @param context The CUDA context
     * @param dataTypeName The data type, "float" or "double"
     * @param elementSize The size of one element, Sizeof.FLOAT or Sizeof.DOUBLE
     */
    VecReductions(CUcontext context, String dataTypeName, int elementSize)
    {
        this.vecReductions = new LinkedHashMap<String, VecReduction>();

        vecReductions.put("add", 
            new VecReduction(context, elementSize, dataTypeName, "add"));
        vecReductions.put("mul", 
            new VecReduction(context, elementSize, dataTypeName, "mul"));
        vecReductions.put("min", 
            new VecReduction(context, elementSize, dataTypeName, "min"));
        vecReductions.put("max", 
            new VecReduction(context, elementSize, dataTypeName, "max"));
    }
    
    /**
     * Return the {@link VecReduction} instance for the given reduction
     * operator. Returns <code>null</code> if the given operator name
     * is not known.
     * 
     * @param reductionOperatorName The reduction operator
     * @return The {@link VecReduction} instance
     */
    VecReduction get(String reductionOperatorName)
    {
        return vecReductions.get(reductionOperatorName);
    }
    
    /**
     * Perform a shutdown, releasing all resources that have been
     * allocated by this instance.
     */
    void shutdown()
    {
        for (VecReduction vecReduction : vecReductions.values())
        {
            vecReduction.shutdown();
        }
    }
}
