/*
 * JCudaVec - Vector operations for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2013-2017 Marco Hutter - http://www.jcuda.org
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

import static jcuda.driver.JCudaDriver.cuDevicePrimaryCtxRelease;

import jcuda.CudaException;
import jcuda.driver.CUdevice;

/**
 * An opaque handle for the vector functions. An instance of this class
 * must be passed to each vector operation method. It contains information
 * about the context in which the vector operations take place.<br>
 * <br>
 * Instances of this class may be created with the {@link Vec#createHandle()} 
 * method. This will allocate all internal resources. When the handle is no 
 * longer used, it should be destroyed by passing it to 
 * {@link Vec#destroyHandle(VecHandle)}.
 */
public final class VecHandle
{
    /**
     * The device of the context that this handle was created for
     */
    private final CUdevice device;
    
    /**
     * Whether the context is primary
     */
    private boolean usingPrimaryContext;
    
    /**
     * The {@link VecKernels} for float operations
     */
    private final VecKernels vecKernelsFloat;

    /**
     * The {@link VecKernels} for double operations
     */
    private final VecKernels vecKernelsDouble;

    /**
     * The {@link VecReduction} instances for float values
     */
    private final VecReductions vecReductionsFloat;
    
    /**
     * The {@link VecReduction} instances for double values
     */
    private final VecReductions vecReductionsDouble;
    
    /**
     * Whether this handle was already destroyed    
     */
    private boolean destroyed;

    /**
     * Default constructor
     * 
     * @param device The device
     * @param usingPrimaryContext Whether the context is primary
     * @param vecKernelsFloat The {@link VecKernels} instance
     * @param vecKernelsDouble The {@link VecKernels} instance
     * @param vecReductionsFloat The {@link VecReduction} instances for float
     * @param vecReductionsDouble The {@link VecReduction} instances for double
     */
    VecHandle(CUdevice device, boolean usingPrimaryContext, 
        VecKernels vecKernelsFloat, VecKernels vecKernelsDouble, 
        VecReductions vecReductionsFloat, 
        VecReductions vecReductionsDouble)
    {
        this.device = device;
        this.usingPrimaryContext = usingPrimaryContext;
        this.vecKernelsFloat = vecKernelsFloat;
        this.vecKernelsDouble = vecKernelsDouble;
        this.vecReductionsFloat= vecReductionsFloat;
        this.vecReductionsDouble= vecReductionsDouble;
        this.destroyed = false;
    }
    
    /**
     * Returns the {@link VecKernels} for float
     * 
     * @return The {@link VecKernels}
     */
    VecKernels getVecKernelsFloat()
    {
        return vecKernelsFloat;
    }

    /**
     * Returns the {@link VecKernels} for double
     * 
     * @return The {@link VecKernels}
     */
    VecKernels getVecKernelsDouble()
    {
        return vecKernelsDouble;
    }

    /**
     * Returns the {@link VecReductions} for float values
     * 
     * @return The {@link VecReductions}
     */
    VecReductions getVecReductionsFloat()
    {
        return vecReductionsFloat;
    }
    
    /**
     * Returns the {@link VecReductions} for double values
     * 
     * @return The {@link VecReductions}
     */
    VecReductions getVecReductionsDouble()
    {
        return vecReductionsDouble;
    }
    
    /**
     * Destroy this handle and release all associated resources, if the
     * handle was not destroyed yet.
     */
    void destroy()
    {
        if (!destroyed)
        {
            vecKernelsFloat.shutdown();
            vecKernelsDouble.shutdown();
            vecReductionsFloat.shutdown();
            vecReductionsDouble.shutdown();
            if (usingPrimaryContext)
            {
                VecUtils.checkResultDriver(
                    cuDevicePrimaryCtxRelease(device));
            }
            destroyed = true;
        }
    }

    /**
     * Ensure that this handle is valid (meaning that it has not been
     * destroyed yet), and throw a {@link CudaException} otherwise.
     */
    void validate()
    {
        if (destroyed) 
        {
            throw new CudaException("The VecHandle has already been destroyed");
        }
    }
    
}
