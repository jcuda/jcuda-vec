/*
 * JCudaVec - Vector operations for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2013-2015 Marco Hutter - http://www.jcuda.org
 */
 
extern "C"
__global__ void vec_set (size_t n, double *result, double  value)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = value;
    }
}


//=== Vector arithmetic ======================================================

extern "C"
__global__ void vec_add (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] + y[id];
    }
}


extern "C"
__global__ void vec_sub (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] - y[id];
    }
}


extern "C"
__global__ void vec_mul (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] * y[id];
    }
}


extern "C"
__global__ void vec_div (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] / y[id];
    }
}

extern "C"
__global__ void vec_negate (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = -x[id];
    }
}




//=== Vector-and-scalar arithmetic ===========================================

extern "C"
__global__ void vec_addScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] + y;
    }
}


extern "C"
__global__ void vec_subScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] - y;
    }
}


extern "C"
__global__ void vec_mulScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] * y;
    }
}


extern "C"
__global__ void vec_divScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] / y;
    }
}




extern "C"
__global__ void vec_scalarAdd (size_t n, double *result, double  x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x + y[id];
    }
}


extern "C"
__global__ void vec_scalarSub (size_t n, double *result, double  x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x - y[id];
    }
}


extern "C"
__global__ void vec_scalarMul (size_t n, double *result, double  x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x * y[id];
    }
}


extern "C"
__global__ void vec_scalarDiv (size_t n, double *result, double  x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x / y[id];
    }
}









//=== Vector comparison ======================================================

extern "C"
__global__ void vec_lt (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] < y[id])?1.0:0.0;
    }
}


extern "C"
__global__ void vec_lte (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] <= y[id])?1.0:0.0;
    }
}


extern "C"
__global__ void vec_eq (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] == y[id])?1.0:0.0;
    }
}


extern "C"
__global__ void vec_gte (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] >= y[id])?1.0:0.0;
    }
}


extern "C"
__global__ void vec_gt (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] > y[id])?1.0:0.0;
    }
}



extern "C"
__global__ void vec_ne (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] != y[id])?1.0:0.0;
    }
}




//=== Vector-and-scalar comparison ===========================================

extern "C"
__global__ void vec_ltScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] < y)?1.0:0.0;
    }
}


extern "C"
__global__ void vec_lteScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] <= y)?1.0:0.0;
    }
}


extern "C"
__global__ void vec_eqScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] == y)?1.0:0.0;
    }
}


extern "C"
__global__ void vec_gteScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] >= y)?1.0:0.0;
    }
}


extern "C"
__global__ void vec_gtScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] > y)?1.0:0.0;
    }
}


extern "C"
__global__ void vec_neScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = (x[id] != y)?1.0:0.0;
    }
}











//=== Vector math (one argument) =============================================


// Calculate the arc cosine of the input argument.
extern "C"
__global__ void vec_acos (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = acos(x[id]);
    }
}


// Calculate the nonnegative arc hyperbolic cosine of the input argument.
extern "C"
__global__ void vec_acosh (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = acosh(x[id]);
    }
}


// Calculate the arc sine of the input argument.
extern "C"
__global__ void vec_asin (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = asin(x[id]);
    }
}


// Calculate the arc hyperbolic sine of the input argument.
extern "C"
__global__ void vec_asinh (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = asinh(x[id]);
    }
}


// Calculate the arc tangent of the input argument.
extern "C"
__global__ void vec_atan (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = atan(x[id]);
    }
}


// Calculate the arc hyperbolic tangent of the input argument.
extern "C"
__global__ void vec_atanh (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = atanh(x[id]);
    }
}


// Calculate the cube root of the input argument.
extern "C"
__global__ void vec_cbrt (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = cbrt(x[id]);
    }
}


// Calculate ceiling of the input argument.
extern "C"
__global__ void vec_ceil (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = ceil(x[id]);
    }
}


// Calculate the cosine of the input argument.
extern "C"
__global__ void vec_cos (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = cos(x[id]);
    }
}


// Calculate the hyperbolic cosine of the input argument.
extern "C"
__global__ void vec_cosh (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = cosh(x[id]);
    }
}


// Calculate the cosine of the input argument × p .
extern "C"
__global__ void vec_cospi (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = cospi(x[id]);
    }
}


// Calculate the complementary error function of the input argument.
extern "C"
__global__ void vec_erfc (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erfc(x[id]);
    }
}


// Calculate the inverse complementary error function of the input argument.
extern "C"
__global__ void vec_erfcinv (size_t n, double *result, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erfcinv(y[id]);
    }
}


// Calculate the scaled complementary error function of the input argument.
extern "C"
__global__ void vec_erfcx (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erfcx(x[id]);
    }
}


// Calculate the error function of the input argument.
extern "C"
__global__ void vec_erf (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erf(x[id]);
    }
}


// Calculate the inverse error function of the input argument.
extern "C"
__global__ void vec_erfinv (size_t n, double *result, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = erfinv(y[id]);
    }
}


// Calculate the base 10 exponential of the input argument.
extern "C"
__global__ void vec_exp10 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = exp10(x[id]);
    }
}


// Calculate the base 2 exponential of the input argument.
extern "C"
__global__ void vec_exp2 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = exp2(x[id]);
    }
}


// Calculate the base e exponential of the input argument.
extern "C"
__global__ void vec_exp (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = exp(x[id]);
    }
}


// Calculate the base e exponential of the input argument, minus 1.
extern "C"
__global__ void vec_expm1 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = expm1(x[id]);
    }
}


// Calculate the absolute value of its argument.
extern "C"
__global__ void vec_fabs (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fabs(x[id]);
    }
}


// Calculate the largest integer less than or equal to x.
extern "C"
__global__ void vec_floor (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = floor(x[id]);
    }
}


// Calculate the value of the Bessel function of the first kind of order 0 for the input argument.
extern "C"
__global__ void vec_j0 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = j0(x[id]);
    }
}


// Calculate the value of the Bessel function of the first kind of order 1 for the input argument.
extern "C"
__global__ void vec_j1 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = j1(x[id]);
    }
}


// Calculate the natural logarithm of the absolute value of the gamma function of the input argument.
extern "C"
__global__ void vec_lgamma (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = lgamma(x[id]);
    }
}


// Calculate the base 10 logarithm of the input argument.
extern "C"
__global__ void vec_log10 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = log10(x[id]);
    }
}


// Calculate the value of l o g e ( 1 + x ) .
extern "C"
__global__ void vec_log1p (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = log1p(x[id]);
    }
}


// Calculate the base 2 logarithm of the input argument.
extern "C"
__global__ void vec_log2 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = log2(x[id]);
    }
}


// Calculate the doubleing point representation of the exponent of the input argument.
extern "C"
__global__ void vec_logb (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = logb(x[id]);
    }
}


// Calculate the natural logarithm of the input argument.
extern "C"
__global__ void vec_log (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = log(x[id]);
    }
}


// Calculate the standard normal cumulative distribution function.
extern "C"
__global__ void vec_normcdf (size_t n, double *result, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = normcdf(y[id]);
    }
}


// Calculate the inverse of the standard normal cumulative distribution function.
extern "C"
__global__ void vec_normcdfinv (size_t n, double *result, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = normcdfinv(y[id]);
    }
}


// Calculate reciprocal cube root function.
extern "C"
__global__ void vec_rcbrt (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = rcbrt(x[id]);
    }
}


// Round input to nearest integer value in doubleing-point.
extern "C"
__global__ void vec_rint (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = rint(x[id]);
    }
}


// Round to nearest integer value in doubleing-point.
extern "C"
__global__ void vec_round (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = round(x[id]);
    }
}


// Calculate the reciprocal of the square root of the input argument.
extern "C"
__global__ void vec_rsqrt (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = rsqrt(x[id]);
    }
}


// Calculate the sine of the input argument.
extern "C"
__global__ void vec_sin (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = sin(x[id]);
    }
}


// Calculate the hyperbolic sine of the input argument.
extern "C"
__global__ void vec_sinh (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = sinh(x[id]);
    }
}


// Calculate the sine of the input argument × p .
extern "C"
__global__ void vec_sinpi (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = sinpi(x[id]);
    }
}


// Calculate the square root of the input argument.
extern "C"
__global__ void vec_sqrt (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = sqrt(x[id]);
    }
}


// Calculate the tangent of the input argument.
extern "C"
__global__ void vec_tan (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = tan(x[id]);
    }
}


// Calculate the hyperbolic tangent of the input argument.
extern "C"
__global__ void vec_tanh (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = tanh(x[id]);
    }
}


// Calculate the gamma function of the input argument.
extern "C"
__global__ void vec_tgamma (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = tgamma(x[id]);
    }
}


// Truncate input argument to the integral part.
extern "C"
__global__ void vec_trunc (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = trunc(x[id]);
    }
}


// Calculate the value of the Bessel function of the second kind of order 0 for the input argument.
extern "C"
__global__ void vec_y0 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = y0(x[id]);
    }
}


// Calculate the value of the Bessel function of the second kind of order 1 for the input argument.
extern "C"
__global__ void vec_y1 (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = y1(x[id]);
    }
}











//=== Vector math (two arguments) ============================================





// Create value with given magnitude, copying sign of second value.
extern "C"
__global__ void vec_copysign (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = copysign(x[id], y[id]);
    }
}

// Compute the positive difference between x and y.
extern "C"
__global__ void vec_fdim (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fdim(x[id], y[id]);
    }
}

// Divide two doubleing point values.
extern "C"
__global__ void vec_fdivide (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fdivide(x[id], y[id]);
    }
}

// Determine the maximum numeric value of the arguments.
extern "C"
__global__ void vec_fmax (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fmax(x[id], y[id]);
    }
}

// Determine the minimum numeric value of the arguments.
extern "C"
__global__ void vec_fmin (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fmin(x[id], y[id]);
    }
}

// Calculate the doubleing-point remainder of x / y.
extern "C"
__global__ void vec_fmod (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = fmod(x[id], y[id]);
    }
}

// Calculate the square root of the sum of squares of two arguments.
extern "C"
__global__ void vec_hypot (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = hypot(x[id], y[id]);
    }
}

// Return next representable single-precision doubleing-point value afer argument.
extern "C"
__global__ void vec_nextafter (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = nextafter(x[id], y[id]);
    }
}

// Calculate the value of first argument to the power of second argument.
extern "C"
__global__ void vec_pow (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = pow(x[id], y[id]);
    }
}

// Compute single-precision doubleing-point remainder.
extern "C"
__global__ void vec_remainder (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = remainder(x[id], y[id]);
    }
}




