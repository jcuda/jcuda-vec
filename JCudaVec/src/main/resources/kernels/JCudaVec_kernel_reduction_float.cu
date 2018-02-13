/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright (c) 2013-2018 Marco Hutter - http://www.jcuda.org
 *
 * This code is based on the NVIDIA 'reduction' CUDA sample,
 * Copyright 1993-2010 NVIDIA Corporation.
 */
 
#ifdef REDUCTION_ADD

    #define REDUCTION_INITIAL_VALUE (0)
    #define REDUCTION_OPERATOR(x, y) ((x)+(y))
    #define REDUCTION_NAME reduce_add

#elif REDUCTION_MUL

    #define REDUCTION_INITIAL_VALUE (1)
    #define REDUCTION_OPERATOR(x, y) ((x)*(y))
    #define REDUCTION_NAME reduce_mul

#elif REDUCTION_MIN

    #define FLT_MAX 3.40282347E+38F
    #define REDUCTION_INITIAL_VALUE (FLT_MAX)
    #define REDUCTION_OPERATOR(x, y) (min(x,y))
    #define REDUCTION_NAME reduce_min

#elif REDUCTION_MAX

    #define FLT_MAX 3.40282347E+38F
    #define REDUCTION_INITIAL_VALUE (-(FLT_MAX))
    #define REDUCTION_OPERATOR(x, y) (max(x,y))
    #define REDUCTION_NAME reduce_max

#endif

extern "C"
__global__ void REDUCTION_NAME(size_t n, float *g_idata, float *g_odata)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    size_t gridSize = blockDim.x*2*gridDim.x;

    float result = REDUCTION_INITIAL_VALUE;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        result = REDUCTION_OPERATOR(result, g_idata[i]);
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            result = REDUCTION_OPERATOR(result, g_idata[i+blockDim.x]);
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = result;
    __syncthreads();


    // do reduction in shared mem
    if (blockDim.x >= 512) { 
        if (tid < 256) { 
            result = REDUCTION_OPERATOR(result, sdata[tid + 256]);
            sdata[tid] = result;
        } 
        __syncthreads(); 
    }
    if (blockDim.x >= 256) { 
        if (tid < 128) { 
            result = REDUCTION_OPERATOR(result, sdata[tid + 128]);
            sdata[tid] = result;
        } 
        __syncthreads(); 
    }
    if (blockDim.x >= 128) { 
        if (tid < 64) { 
            result = REDUCTION_OPERATOR(result, sdata[tid + 64]);
            sdata[tid] = result;
        } 
        __syncthreads(); 
    }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float* smem = sdata;
        if (blockDim.x >=  64) { 
            result = REDUCTION_OPERATOR(result, smem[tid + 32]);
            smem[tid] = result;
        }
        if (blockDim.x >=  32) { 
            result = REDUCTION_OPERATOR(result, smem[tid + 16]);
            smem[tid] = result;
        }
        if (blockDim.x >=  16) { 
            result = REDUCTION_OPERATOR(result, smem[tid +  8]);
            smem[tid] = result;
        }
        if (blockDim.x >=   8) { 
            result = REDUCTION_OPERATOR(result, smem[tid +  4]);
            smem[tid] = result;
        }
        if (blockDim.x >=   4) { 
            result = REDUCTION_OPERATOR(result, smem[tid +  2]);
            smem[tid] = result;
        }
        if (blockDim.x >=   2) { 
            result = REDUCTION_OPERATOR(result, smem[tid +  1]);
            smem[tid] = result;
        }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

