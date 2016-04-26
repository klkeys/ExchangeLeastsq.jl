# ExchangeLeastsq

_ExchangeLeastsq.jl_ is a module to perform sparse model selection in least squares regression without shrinkage.

## Installation

From a Julia prompt, type

    Pkg.add("https://bitbucket.org/kevin_keys/exchangeleastsq.jl")

## Usage

The workhorse of _ExchangeLeastsq.jl_ is the function `exchange_leastsq!` which requires five arguments:

* `bvec`, a vector beta of statistical coefficients
* `x`, the statistical design matrix
* `y`, the response vector
* `perm`, an index vector that sorts `bvec` by magnitude
* `r`, the number of coefficients to include in the model

Testing one value of `r` is unlikely to provide much insight. `exchange_leastsq!` works best in a loop.
For example, if `models` is an integer vector of model sizes to try, then one can construct a regularization path via

    for i in models
        exchange_leastsq!(bvec, x, y, perm, i)
    end

Observe that `exchange_leastsq!` mutates `bvec`. When looping through several model sizes,
any information about the model should be stored in a separate copy of `bvec`.

## Crossvalidation

_ExchangeLeastsq.jl_ is best used to obtain the ideal model size to predict `y`.
It furnishes a crossvalidation routine for this purpose.
_ExchangeLeastsq.jl_ makes use of `SharedArrays` to enable crossvalidation in a multicore shared memory environment.
Users can perform _q_-fold crossvalidation for a number `n` of model sizes by calling 

    mses, b, bidx = cv_exlstsq(x, y, n, q)

`b` and `bidx` contain the coefficients and indices, respectively, of the ideal model size.
`mses` contains the vector of mean squared errors.

## GWAS

_ExchangeLeastsq.jl_ interfaces with the [PLINK.jl](https://github.com/klkeys/PLINK.jl) package to enable GWAS analysis.
_PLINK.jl_ furnishes both multicore and GPU interfaces for GWAS analysis.
The multicore environment makes heavy used of `SharedArray` data.
The GPU environment uses [OpenCL.jl](https://github.com/JuliaGPU/OpenCL.jl) wrappers to port the computational bottleneck `x' * y` to the GPU. 
