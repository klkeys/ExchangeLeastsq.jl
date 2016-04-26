"""
An iterative exchange algorithm for penalized least squares regression.
The exchange algorithm operates similar to cyclic coordinate descent,
except that it swaps (exchanges) predictors to improve the residual sum of squares. 
"""
module ExchangeLeastsq

using GLMNet
using RegressionTools
using PLINK
using Distances: euclidean, chebyshev
using StatsBase: sample
using OpenCL

export exchange_leastsq!
export cv_exlstsq
export test_exchangeleastsq
export test_exleastsq
export test_cv_exlstsq
export compare_exlstsq
export test_exchangeleastsq_plink

"An alias for the `OpenCL` module name"
const cl = OpenCL

# typealias for Float32/Float64
typealias Float Union{Float32, Float64}

include("gpu.jl")
include("gwas.jl")
include("cv.jl")
include("exlstsq.jl")
include("test.jl")

end # end module ExchangeLeastsq
