// 1 / (1 + e^(-x))
extern "C"
__global__ void logistic(size_t n, double *result, double *x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        result[i] = 1.0 / (1.0 + exp(-x[i]));
    }

}
