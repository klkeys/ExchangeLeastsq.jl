type ELSQVariables{T <: Float, V <: DenseVector}
    b        :: V
    nrmsq    :: Vector{T}
    df       :: V
#    dotprods :: Vector{T}
    dotprods :: V 
#    tempp    :: Vector{T} 
    tempp    :: V 
    r        :: V
#    tempn    :: Vector{T} 
#    tempn2   :: Vector{T}
    tempn    :: V 
    tempn2   :: V
    xb       :: V
    perm     :: Vector{Int} 
    inner    :: Dict{Int, Vector{T}}
    mask_n   :: Vector{Int}
    idx      :: BitArray{1}

    ELSQVariables(b::DenseVector{T}, nrmsq::DenseVector{T}, df::DenseVector{T}, dotprods::DenseVector{T}, tempp::DenseVector{T}, r::DenseVector{T}, tempn::DenseVector{T}, tempn2::DenseVector{T}, xb::DenseVector{T}, perm::Vector{Int}, inner::Dict{Int, Vector{T}}, mask_n::Vector{Int}, idx::BitArray{1}) = new(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, xb, perm, inner, mask_n, idx)
end

function ELSQVariables{T <: Float}(
    b        :: DenseVector{T},
    nrmsq    :: Vector{T},
    df       :: DenseVector{T},
    dotprods :: Vector{T},
#    tempp    :: Vector{T},
    tempp    :: DenseVector{T},
    r        :: DenseVector{T},
#    tempn    :: Vector{T},
#    tempn2   :: Vector{T},
    tempn    :: DenseVector{T},
    tempn2   :: DenseVector{T},
    xb       :: DenseVector{T},
    perm     :: Vector{Int}, 
    inner    :: Dict{Int, Vector{T}},
    mask_n   :: Vector{Int},
    idx      :: BitArray{1}
)
    ELSQVariables{T, typeof(b)}(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, xb, perm, inner, mask_n, idx)
end

function ELSQVariables{T <: Float}(
    x :: DenseMatrix{T},
    y :: DenseVector{T}
)
    # dimensions of arrays
    n,p = size(x)

    # form arrays
    b        = zeros(T, p) 
    nrmsq    = vec(sumabs2(x,1)) :: Vector{T}
    df       = zeros(T, p)
    dotprods = zeros(T, p)
    tempp    = zeros(T, p)
    r        = zeros(T, n)
    tempn    = zeros(T, n)
    tempn2   = zeros(T, n)
    xb       = zeros(T, n) 
    perm     = collect(1:p)
#    mask_n   = zeros(Int, n)
    mask_n   = ones(Int, n)
    idx      = falses(p)

    # form dictionary
#    inner = Dict{Int, DenseVector{T}}()
    inner = Dict{Int, Vector{T}}()

    # return container object
    ELSQVariables{eltype(y), typeof(y)}(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, xb, perm, inner, mask_n, idx)
end


function ELSQVariables{T <: Float}(
    x :: BEDFile{T},
    y :: SharedVector{T},
#    z :: DenseVector{Int}
)
    # dimensions of arrays
    n,p = size(x)

    # process ids?
    pids = procs(x)

    # form arrays
    b        = SharedArray(T, (p,), pids=pids) :: typeof(y)
    nrmsq    = (length(y) - 1) * ones(T, p) 
    df       = SharedArray(T, (p,), pids=pids) :: typeof(y)
#    dotprods = zeros(T, p) 
    dotprods = SharedArray(T, (p,), pids=pids) :: typeof(y) 
#    tempp    = zeros(T, p) 
    tempp    = SharedArray(T, (p,), pids=pids) :: typeof(y)
    r        = SharedArray(T, (n,), pids=pids) :: typeof(y)
#    tempn    = zeros(T, n) 
#    tempn2   = zeros(T, n) 
    tempn    = SharedArray(T, (n,), pids=pids) :: typeof(y)
    tempn2   = SharedArray(T, (n,), pids=pids) :: typeof(y)
    xb       = SharedArray(T, (n,), pids=pids) :: typeof(y)
    perm     = collect(1:p)
#    mask_n   = zeros(Int, n)
    mask_n   = ones(Int, n)
    idx      = falses(p)

    # form dictionary
    inner = Dict{Int, Vector{T}}()

    # return container object
    ELSQVariables{eltype(y), typeof(y)}(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, xb, perm, inner, mask_n, idx)
end

function ELSQVariables{T <: Float}(
    x :: BEDFile{T},
    y :: SharedVector{T},
    z :: DenseVector{Int} # <-- this should be the bitmask
)
    w = ELSQVariables(x,y)
    copy!(w.mask_n, z)
    return w
end

immutable ELSQCrossvalidationResults{T <: Float}
    mses :: Vector{T}
    b    :: Vector{T}
    bidx :: Vector{Int}
    k    :: Int
    path :: Vector{Int}
    bids :: Vector{UTF8String}
end

#function Base.display(x::ELSQCrossvalidationResults)
#    println("Crossvalidation Results:")
#    println("Best model size is $k predictors")
#    println("\tPredictor\tValue\tMSE")
#    for i in eachindex(x.bidx)
#        println("\t", x.bidx[i], "\t", x.b[i], "\t", x.mses[i])
#    end
#end

# constructor for when bids are not available
# simply makes vector of "V$i" where $i are drawn from bidx
function ELSQCrossvalidationResults{T <: Float}(
    mses :: Vector{T},
    b    :: Vector{T},
    bidx :: Vector{Int},
    k    :: Int,
    path :: Vector{Int},
)  
    bids = convert(Vector{UTF8String}, ["V" * "$i" for i in bidx]) :: Vector{UTF8String}
    ELSQCrossvalidationResults{eltype(mses)}(mses, b, bidx, k, path, bids)
end

# function to view an ELSQCrossvalidationResults object
function Base.show(io::IO, x::ELSQCrossvalidationResults)
    println(io, "Crossvalidation results:") 
    println(io, "Minimum MSE ", minimum(x.mses), " occurs at k = $(x.k).")
    println(io, "Best model β has the following nonzero coefficients:")
    println(io, DataFrame(Predictor=x.bidx, Name=x.bids, Estimated_β=x.b))
    return nothing
end

function Gadfly.plot(x::ELSQCrossvalidationResults)
    df = DataFrame(ModelSize=x.path, MSE=x.mses)
    plot(df, x="ModelSize", y="MSE", xintercept=[x.k], Geom.line, Geom.vline(color=colorant"red"), Guide.xlabel("Model size"), Guide.ylabel("MSE"), Guide.title("MSE versus model size"))
end

function check_finiteness{T <: Float}(x::T)
    isnan(x) && throw(error("Objective function is NaN, aborting..."))
    isinf(x) && throw(error("Objective function is Inf, aborting..."))
end

function print_descent_error{T <: Float}(iter::Int, loss::T, next_loss::T)
    print_with_color(:red, "\nExchange algorithm fails to descend!\n")
    print_with_color(:red, "Iteration: $(iter)\n")
    print_with_color(:red, "Current Objective: $(loss)\n")
    print_with_color(:red, "Next Objective: $(next_loss)\n")
    print_with_color(:red, "Difference in objectives: $(abs(next_loss - loss))\n")
    throw(error("Descent failure!"))
end

function print_maxiter{T <: Float}(max_iter::Int, loss::T)
    print_with_color(:red, "Exchange algorithm has hit maximum iterations $(max_iter)!\n")
    print_with_color(:red, "Return value may be incorrect\n")
    print_with_color(:red, "Current Loss: $(loss)\n")
end 

function errorcheck{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
#    k        :: Int,
    tol      :: T,
    max_iter :: Int,
    window   :: Int,
    p        :: Int = size(x,2)
)
#    0 <= k <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(bvec)"))
    tol >= eps()          || throw(ArgumentError("Global tolerance must exceed machine precision"))
    max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
#    0 <= window <= k      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))
    return nothing
end

function errorcheck{T <: Float}(
    x        :: BEDFile{T},
    y        :: SharedVector{T},
#    k        :: Int,
    tol      :: T,
    max_iter :: Int,
    window   :: Int,
    p        :: Int = size(x,2)
)
#    0 <= k <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(bvec)"))
    tol >= eps()          || throw(ArgumentError("Global tolerance must exceed machine precision"))
    max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
#    0 <= window <= k      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))
    return nothing
end

function print_cv_results{T <: Float}(errors::DenseVector{T}, path::DenseVector{Int}, k::Int)
    println("\n\nCrossvalidation Results:")
    println("k\tMSE")
    for i = 1:length(errors)
        println(path[i], "\t", errors[i])
    end
    println("\nThe lowest MSE is achieved at k = ", k)
end

# subroutine compares current predictor i against all predictors k+1, k+2, ..., p
# these predictors are candidates for inclusion in set
# _exlstsq_innerloop! find best new predictor
function _exlstsq_innerloop!{T <: Float}(
    v   :: ELSQVariables{T},
    k   :: Int, 
    i   :: Int,
    p   :: Int,
    tol :: T
)
    l     = v.perm[i] :: Int
    betal = v.b[l]    :: T

    # save values to determine best estimate for current predictor
    b   = v.nrmsq[l] :: T
    a   = (v.df[l] + betal*b) :: T
    adb = a / b :: T
    r   = i

    # inner loop compares current predictor j against all remaining predictors j+1,...,p
    for j = (k+1):p
        idx = v.perm[j] :: Int
        c   = (v.df[idx] + betal*v.dotprods[idx]) :: T
        d   = v.nrmsq[idx] :: T

        # if current inactive predictor beats current active predictor,
        # then save info for swapping
        if c*c/d > a*adb + tol
            a   = c
            b   = d
            r   = j
            adb = a / b
        end
    end # end inner loop over remaining predictor set

    return a, b, r, adb
end

# subroutine to swap predictors
# in exchange_leastsq!, will swap best predictor with current predictor
function _swap_predictors!{T <: Float}(
    v   :: ELSQVariables{T},
    i   :: Int,
    r   :: Int,
    m   :: Int,
    adb :: T
)
    # save ith best predictor 
    j = v.perm[i]

    # replace index of ith best with new best predictor
    v.perm[i] = v.perm[r]

    # replace new best predictor with ith best
    # at this point, swap of indices is complete
    v.perm[r] = j

    # replace coefficient of mth best predictor with a / b 
    v.b[m] = adb

    # if rth and ith predictors coincide,
    # then set ith best predictor coefficient to zero 
    if r != i
        v.b[j] = zero(T)
    end
    return nothing
end

"""
    axpymbz!(y,a,x,b,z[, p=length(y)])

The silly name is based on BLAS `axpy()` (A*X Plus Y), except that this function performs *A**X* *P*lus *Y* *M*inus *B**Z*.
The idea behind `axpymz!()` is to perform the computation in one pass over the arrays. The output is the same as `y = y + a*x - b*z`.
"""
function axpymbz!{T <: Float}(
    y :: DenseVector{T},
    a :: T,
    x :: DenseVector{T},
    b :: T,
    z :: DenseVector{T};
)
    @inbounds @simd for i in eachindex(y) 
        y[i] = y[i] + a*x[i] - b*z[i]
    end
    return nothing
end

# subroutine to refit preditors after crossvalidation
function refit_exlstsq{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    k        :: Int;
    models   :: DenseVector{Int} = collect(1:min(20,size(x,2))),
    tol      :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    window   :: Int  = maximum(models),
    quiet    :: Bool = true,
)
    # initialize β vector and temporary arrays
    v = ELSQVariables(x, y)

    # first use exchange algorithm to extract model
    exchange_leastsq!(v, x, y, k, max_iter=max_iter, quiet=quiet, tol=tol, window=k)

    # which components of beta are nonzero?
    # cannot use binary indices here since we need to return Int indices
    bidx = find(v.b)

    # allocate the submatrix of x corresponding to the inferred model
    # cannot use SubArray since result is not StridedArray?
    # issue is that bidx is Vector{Int} and not a Range object
    # use of SubArray is more memory efficient; a pity that it doesn't work!
#    x_inferred = sub(sdata(x), :, bidx)
    x_inferred = x[:,bidx]

    # now estimate β with the ordinary least squares estimator β = inv(x'x)x'y
    # return it with the vector of MSEs
    xty = BLAS.gemv('T', x_inferred, sdata(y)) :: Vector{T}
    xtx = BLAS.gemm('T', 'N', x_inferred, x_inferred) :: Matrix{T}
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end
    
    return b, bidx
end

# refitting routine for GWAS data with x', mean, prec files
function refit_exlstsq(
    T        :: Type,
    xfile    :: ASCIIString,
    xtfile   :: ASCIIString,
    x2file   :: ASCIIString,
    yfile    :: ASCIIString,
    meanfile :: ASCIIString,
    precfile :: ASCIIString,
    k        :: Int;
    models   :: DenseVector{Int} = collect(1:min(20,size(x,2))),
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-6),
    max_iter :: Int   = 100,
    window   :: Int   = maximum(models),
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # initialize all variables 
    x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, pids=pids, header=header)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}
    v = ELSQVariables(x, y, ones(Int, length(y)))

    # first use exchange algorithm to extract model
    exchange_leastsq!(v, x, y, k, max_iter=max_iter, quiet=quiet, tol=tol, window=k)

    # which components of beta are nonzero?
    inferred_model = v.b .!= zero(T)
    bidx = find(inferred_model)
    
    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model) 

    # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
    # return it with the vector of MSEs
    xty = BLAS.gemv('T', x_inferred, sdata(y)) :: Vector{T}
    xtx = BLAS.gemm('T', 'N', x_inferred, x_inferred) :: Matrix{T}
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

    return b, bidx
end


# refitting routine for GWAS data with just genotypes, covariates, y 
function refit_exlstsq(
    T        :: Type,
    xfile    :: ASCIIString,
    x2file   :: ASCIIString,
    yfile    :: ASCIIString,
    k        :: Int;
    models   :: DenseVector{Int} = collect(1:min(20,size(x,2))),
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-6),
    max_iter :: Int   = 100,
    window   :: Int   = maximum(models),
    quiet    :: Bool  = true,
    header   :: Bool  = false
)

    # initialize all variables 
    x = BEDFile(T, xfile, x2file, pids=pids, header=header)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}
    v = ELSQVariables(x, y, ones(Int, length(y)))

    # first use exchange algorithm to extract model
    exchange_leastsq!(v, x, y, k, max_iter=max_iter, quiet=quiet, tol=tol, window=k)

    # which components of β are nonzero?
    inferred_model = v.b .!= zero(T)
    bidx = find(inferred_model)
    
    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model) 

    # now estimate β with the ordinary least squares estimator b = inv(x'x)x'y
    # return it with the vector of MSEs
    xty = BLAS.gemv('T', x_inferred, sdata(y)) :: Vector{T}
    xtx = BLAS.gemm('T', 'N', x_inferred, x_inferred) :: Matrix{T}
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

    return b, bidx
end


function update_current_best_predictor!{T <: Float}(
    v     :: ELSQVariables{T},
    x     :: DenseMatrix{T},
    betal :: T,
    adb   :: T,
    r     :: Int
)
    # first get index of current best predictor
    m = v.perm[r] :: Int

    # tempn2 = x[:,m] 
    #update_col!(v.tempn2, x, m)
    copy!(v.tempn2, sub(x, :, m))

    # v.r = betal*v.tempn + adb*v.tempn2
    axpymbz!(v.r, betal, v.tempn, adb, v.tempn2)

    return m
end

function update_current_best_predictor!{T <: Float}(
    v     :: ELSQVariables{T},
    x     :: BEDFile{T},
    betal :: T,
    adb   :: T,
    r     :: Int
)
    # first get index of current best predictor
    m = v.perm[r] :: Int

    # v.tempn2 = x[:,m]
    decompress_genotypes!(v.tempn2, x, m) 

    # v.tempn2[v.mask_n .== 0] = 0.0
    mask!(v.tempn2, v.mask_n, 0, zero(T))

    # v.r = betal*v.tempn + adb*v.tempn2
    axpymbz!(v.r, betal, v.tempn, adb, v.tempn2)

    # v.r[v.mask_n .== 0] = 0.0
    mask!(v.r, v.mask_n, 0, zero(T))

    return m
end


function get_inner_product!{T <: Float}(
    z :: DenseVector{T}, 
    w :: DenseVector{T}, 
    v :: ELSQVariables{T}, 
    x :: DenseMatrix{T}, 
    i :: Int
)
    if !haskey(v.inner, i)
        v.inner[i] = BLAS.gemv('T', one(T), x, w)
    end
    copy!(z, v.inner[i])
end


function get_inner_product!{T <: Float}(
    z :: DenseVector{T}, 
    w :: DenseVector{T}, 
    v :: ELSQVariables{T}, 
    x :: BEDFile{T}, 
    i :: Int;
    pids :: DenseVector{Int} = procs(x)
)
    if !haskey(v.inner, i)
        At_mul_B!(z, x, w, v.mask_n, pids=pids)
        v.inner[i] = copy(z) 
    end
    copy!(z, v.inner[i])
end

function get_inner_product!{T <: Float}(
    z :: DenseVector{T}, 
    w :: DenseVector{T}, 
    v :: ELSQVariables{T}, 
    x :: BEDFile{T}, 
    a :: PlinkGPUVariables{T},
    i :: Int;
    pids :: DenseVector{Int} = procs(x)
)
    if !haskey(v.inner, i)
        At_mul_B!(z, x, w, v.mask_n, a, pids=pids)
        v.inner[i] = copy(z) 
    end
    copy!(z, v.inner[i])
end
