# ExchangeLeastsq

_ExchangeLeastsq.jl_ is a module to perform sparse model selection in least squares regression without shrinkage.

## Installation

From a Julia prompt, type

    Pkg.clone("https://github.com/klkeys/ExchangeLeastsq.jl")

## Usage

The workhorse of _ExchangeLeastsq.jl_ is the function `exlstsq` which requires only two arguments:

* `x`, the statistical design matrix
* `y`, the response vector

The function `exlstsq` returns a sparse matrix `betas` of estimated models:
 
    betas = exlstsq(x, y) 

Optional arguments with defaults include:

* `v = ELSQVariables(x, y)` is a container object of temporary arrays for `exlstsq`
* `models` is the integer vector of model sizes to test. It defaults to `collect(1:p)`, where `p = min(20, size(x,2))`.
* `window = maximum(models)` is a window size of active predictors that `exlstsq` uses when searching through active predictors. Generally a smaller value of `window` means that `exlstsq` sifts through fewer active models, thereby increasing speed and sacrificing accuracy.
* `max_iter = 100` is the maximum number of iterations that `exlstsq` will take in any inner loop
* `tol = 1e-6` is the convergence tolerance
* `quiet = true` controls output to the console. Setting `quiet = false` causes `exlstsq` to print all inner loop information.
 
## Crossvalidation

_ExchangeLeastsq.jl_ is best used to obtain the ideal model size to predict `y`.
It furnishes a crossvalidation routine for this purpose.
_ExchangeLeastsq.jl_ makes use of `SharedArrays` to enable crossvalidation in a multicore shared memory environment.
Users can perform _q_-fold crossvalidation for a vector `models` of model sizes by calling 

    cv_output = cv_exlstsq(x, y)

Important optional arguments include:

* `models = collect(1:min(20,p))`, with `p = size(x,2)`, is the `Int` vector of model sizes to test.
* `q = 5` is the number of crossvalidation folds
* `folds` controls the fold structure. The default `RegressionTools.cv_get_folds(y, q)` distributes data to `q` folds as evenly as possible.

Here `cv_output` is an `ELSQCrossvalidationResults` container object with the following fields: 

* `mses` contains the vector of mean squared errors
* `k` is the best crossvalidated model size
* `b` and `bidx` contain the coefficients and indices, respectively, of the model size `k`.


## GWAS

_ExchangeLeastsq.jl_ interfaces with the [PLINK.jl](https://github.com/klkeys/PLINK.jl) package to enable GWAS analysis.
_PLINK.jl_ furnishes both multicore and GPU interfaces for GWAS analysis.
The multicore environment makes heavy used of `SharedArray` interfaces.

For genotype data housed in a `PLINK.BEDFile` object `x` and a `SharedVector` `y`, the function call to `exlstsq` is unchanged: 

    output = exlstsq(x, y)

However, the call to `cv_exlstsq` changes dramatically in order to accomodate `SharedArray` computing. Most users should use the call

    cv_output = cv_exlstsq("PATH_TO_BEDFILE.bed", "PATH_TO_COVARIATES.txt", "PATH_TO_Y.bin")

Note the file extensions! The first file path points to the BED file, the second points to the covariates stored in a delimited text file, and the last points to the response variable `y` stored as a [binary file](https://en.wikipedia.org/wiki/Binary_file). 

## GPU
_PLINK.jl_ also ships with a GPU interface for GWAS analysis.
The GPU environment uses [OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl) wrappers to port the computational bottleneck `x' * y` to the GPU. 
_PLINK.jl_ automatically loads the OpenCL kernels into the variables `PLINK.gpucode64` (for 64-bit kernels) and `PLINK.gpucode32` (for 32-bit kernels).
Assuming that a suitable device is available, then the calls from _ExchangeLeastsq.jl_ to use GPUs are 

    output = exlstsq(x, y, PLINK.gpucode64)
    cv_output = cv_exlstsq("PATH_TO_BEDFILE.bed", "PATH_TO_COVARIATES.txt", "PATH_TO_RESPONSE.bin", PLINK.gpucode64) 
