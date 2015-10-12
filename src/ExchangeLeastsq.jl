module ExchangeLeastsq

using GLMNet
using RegressionTools
using PLINK
using Distances: euclidean, chebyshev
using StatsBase: sample
using Compat

#import Base.LinAlg.BLAS.gemv
#import Base.LinAlg.BLAS.gemv!
#import Base.LinAlg.BLAS.axpy!

export exchange_leastsq
export exchange_leastsq!
export cv_exlstsq
export test_exchangeleastsq
export test_exleastsq
export test_cv_exlstsq
export compare_exlstsq
export test_exchangeleastsq_plink

include("exlstsq64.jl")
include("exlstsq32.jl")
include("cv.jl")
include("test.jl")
include("gwas64.jl")
include("gwas32.jl")


end # end module ExchangeLeastsq
