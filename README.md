# ExchangeLeastsq

_ExchangeLeastsq.jl_ is a module to perform sparse model selection in least squares regression without shrinkage.

## Installation

From a Julia prompt, type

    Pkg.add("https://bitbucket.org/kevin_keys/exchangeleastsq.jl")

## Usage

The workhorse of _ExchangeLeastsq.jl_ is the function `exlstsq` which requires five arguments:

* `x`, the statistical design matrix
* `y`, the response vector
* `models`, an integer vector of model sizes to test

The function `exlstsq` returns a sparse matrix `betas` of estimated models:
 
    betas = exlstsq(x, y, models) 

Optional arguments with defaults include:

* `v = ELSQVariables(x, y)` is a container object of temporary arrays for `exlstsq`
* `window = maximum(models)` is a window size of active predictors that `exlstsq` uses when searching through active predictors. Generally a smaller value of `window` means that `exlstsq` sifts through fewer active models, thereby increasing speed and sacrificing accuracy.
* `max_iter = 100` is the maximum number of iterations that `exlstsq` will take in any inner loop
* `tol = 1e-6` is the convergence tolerance
* `quiet = true` controls output to the console. Setting `quiet = false` causes `exlstsq` to print all inner loop information.
 
## Crossvalidation

_ExchangeLeastsq.jl_ is best used to obtain the ideal model size to predict `y`.
It furnishes a crossvalidation routine for this purpose.
_ExchangeLeastsq.jl_ makes use of `SharedArrays` to enable crossvalidation in a multicore shared memory environment.
Users can perform _q_-fold crossvalidation for a vector `models` of model sizes by calling 

    cv_output = cv_exlstsq(x, y, models, q)

Here `cv_output` is an `ELSQCrossvalidationResults` container object with the following fields: 

* `mses` contains the vector of mean squared errors
* `k` is the best crossvalidated model size
* `b` and `bidx` contain the coefficients and indices, respectively, of the model size `k`.


## GWAS

_ExchangeLeastsq.jl_ interfaces with the [PLINK.jl](https://github.com/klkeys/PLINK.jl) package to enable GWAS analysis.
_PLINK.jl_ furnishes both multicore and GPU interfaces for GWAS analysis.
The multicore environment makes heavy used of `SharedArray` data.
The GPU environment uses [OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl) wrappers to port the computational bottleneck `x' * y` to the GPU. 
