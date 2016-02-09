"""
    update_partial_residuals!(r, y, x, perm, b, k [, n=length(r), p=length(b)])

A subroutine to compute the partial residuals `r = Y - X*b` in-place based on a permutation vector `perm` that indexes the nonzeroes in `b`. 
"""
function update_partial_residuals!(
    r    :: DenseVector{Float64}, 
    y    :: DenseVector{Float64}, 
    x    :: DenseMatrix{Float64}, 
    perm :: DenseArray{Int,1}, 
    b    :: DenseVector{Float64}, 
    k    :: Int; 
    n    :: Int = length(r), 
    p    :: Int = length(b)
)
    k <= p || throw(error("update_partial_residuals!: k cannot exceed the length of b!"))
    copy!(r, y)
    @inbounds for j = 1:k 
        idx = perm[j]
        @inbounds @simd for i = 1:n 
            r[i] += -b[idx]*x[i,idx]
        end 
    end 
    return nothing 
end



function axpymbz!(
    j :: Int,
    y :: DenseVector{Float64},
    a :: Float64,
    x :: DenseVector{Float64},
    b :: Float64,
    z :: DenseVector{Float64}
)
    y[j] + a*x[j] - b*z[j]
end

"""
    axpymbz!(y,a,x,b,z[, p=length(y)])

The silly name is based on BLAS `axpy()` (A*X Plus Y), except that this function performs *A**X* *P*lus *Y* *M*inus *B**Z*.
The idea behind `axpymz!()` is to perform the computation in one pass over the arrays. The output is the same as `y = y + a*x - b*z`.
"""
function axpymbz!(
    y :: Vector{Float64},
    a :: Float64,
    x :: Vector{Float64},
    b :: Float64,
    z :: Vector{Float64};
    p :: Int = length(y)
)
    @inbounds for i = 1:p
        y[i] = axpymbz!(i, y, a, x, b, z)
    end
end

function axpymbz!(
    y :: SharedVector{Float64},
    a :: Float64,
    x :: SharedVector{Float64},
    b :: Float64,
    z :: SharedVector{Float64};
    p :: Int = length(y)
)
    @inbounds for i = 1:p
        y[i] = axpymbz!(i, y, a, x, b, z)
    end
end

#function myrange(q::SharedVector{Float64})
#    idx = indexpids(q)
#    if idx == 0
#        # This worker is not assigned a piece
#        return one(Float64), one(Float64)
#    end
#    nchunks = length(procs(q))
#    splits = [round(Int,s) for s in linspace(0,length(q),nchunks+1)]
#    return splits[idx]+1 : splits[idx+1]
#end
#
#function axpymbz_shared_chunk!(
#    y      :: SharedVector{Float64},
#    a      :: Float64,
#    x      :: SharedVector{Float64},
#    b      :: Float64,
#    z      :: SharedVector{Float64},
#    irange :: UnitRange{Int}
#)
#    @inbounds for i in irange
#        y[i] = axpymbz!(i,y,a,x,b,z)
#    end
#end
#
#axpymbz_shared!(y::SharedVector{Float64}, a::Float64, x::SharedVector{Float64}, b::Float64, z::SharedVector{Float64}) = axpymbz_shared_chunk!(y,a,x,b,z,myrange(y))
#
#"""
#If called with `SharedArray` vectors, then `axpymbz!()` automatically partitions the indices of the vectors and farms the computations to all available processes.
#"""
#function axpymbz!(
#    y::SharedVector{Float64},
#    a::Float64,
#    x::SharedVector{Float64},
#    b::Float64,
#    z::SharedVector{Float64}
#)
#    @sync begin
#        for p in procs(y)
#            @async remotecall_wait(p, axpymbz_shared!, y, a, x, b, z)
#        end
#    end
#end

"""
    exchange_leastsq!(bvec,x,y,perm,r) -> bvec

This function minimizes the residual sum of squares

    0.5*sumabs2( y - x*bvec )

subject to `bvec` having no more than `r` nonzero components. The function will compute a `bvec` for a given value of `r`.
For optimal accuracy, this function should be run for multiple values of `r` over a path.
For optimal performance over regularization path computations, use warmstarts for the arguments `bvec`, `perm`, and `inner`.

Arguments:

- `bvec` is the `p`-dimensional model.
- `X` is the n x p statistical design matrix.
- `Y` is the n-dimensional response vector.
- `perm` is a p-dimensional array of integers that sort beta in descending order by magnitude.
- `r` is the desired number of nonzero components in `bvec`.

Optional Arguments:

- `inner` is a `Dict` for storing Hessian inner products dynamically as needed instead of precomputing all of `x' * x`. Defaults to an empty `Dict{Int,DenseVector{Float}}()`.
- `n` and `p` are the dimensions of `x`; the former defaults to `length(y)` while the latter defaults to `size(x,2)`.
- `nrmsq` is the vector to store the squared norms of the columns of `x`. Defaults to `vec(sumabs2(x,1))`.
- `df` is the temporary array to store the gradient. Defaults to `zeros(p)`.
- `dotprods` is the temporary array to store the current column of dot products from `inner`. Defaults to `zeros(p)`.
- `tempp` is a temporary array of length `p`. Defaults to `zeros(p)`.
- `res` is the temporary array to store the vector of *res*iduals. Defaults to `zeros(n)`.
- `tempn` is a temporary array of length `n`. Defaults to `zeros(n)`.
- `tempn2` is another temporary array of length `n`. Also defaults to `zeros(n)`.
- `window` is an `Int` to dictate the dimension of the search window for potentially exchanging predictors.
   Defaults to `r` (potentially exchange all current predictors). Decreasing this quantity tells the algorithm to search through
   fewer current active predictors, which can decrease compute time but can also degrade model recovery performance.
- `max_iter` is the maximum permissible number of iterations. Defaults to `100`.
- `tol` is the convergence tolerance. Defaults to `1e-6`.
- `quiet` is a `Bool` to control output. Defaults to `false` (full output).
"""
function exchange_leastsq!(
    bvec     :: DenseVector{Float64},
    x        :: DenseMatrix{Float64},
    y        :: DenseVector{Float64},
    perm     :: DenseVector{Int},
    r        :: Int;
    inner    :: Dict{Int, DenseVector{Float64}} = Dict{Int,DenseVector{Float64}}(),
    n        :: Int = length(y),
    p        :: Int = size(x,2),
    nrmsq    :: DenseVector{Float64} = vec(sumabs2(x,1)),
    df       :: DenseVector{Float64} = zeros(Float64, p),
    dotprods :: DenseVector{Float64} = zeros(Float64, p),
    tempp    :: DenseVector{Float64} = zeros(Float64, p),
    res      :: DenseVector{Float64} = zeros(Float64, n),
    tempn    :: DenseVector{Float64} = zeros(Float64, n),
    tempn2   :: DenseVector{Float64} = zeros(Float64, n),
    window   :: Int     = r,
    max_iter :: Int     = 100,
    tol      :: Float64 = 1e-6,
    quiet    :: Bool    = false
)
    # error checking
    n == size(x,1)        || throw(DimensionMismatch("length(Y) != size(X,1)"))
    n == length(tempn)    || throw(DimensionMismatch("length(Y) != length(tempn)"))
    n == length(tempn2)   || throw(DimensionMismatch("length(Y) != length(tempn2)"))
    n == length(res)      || throw(DimensionMismatch("length(Y) != length(res)"))
    p == length(bvec)     || throw(DimensionMismatch("length(bvec) != length(bvec)"))
    p == length(df)       || throw(DimensionMismatch("length(bvec) != length(df)"))
    p == length(tempp)    || throw(DimensionMismatch("length(bvec) != length(tempp)"))
    p == length(dotprods) || throw(DimensionMismatch("length(bvec) != length(dotprods)"))
    p == length(nrmsq)    || throw(DimensionMismatch("length(bvec) != length(nrmsq)"))
    p == length(perm)     || throw(DimensionMismatch("length(bvec) != length(perm)"))
    0 <= r <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(bvec)"))
    tol >= eps()          || throw(ArgumentError("Global tolerance must exceed machine precision"))
    max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
    0 <= window <= r      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))


    # declare algorithm variables
    i       = 0                         # used for iterations
    iter    = 0                         # used for outermost loop
    j       = 0                         # used for iterations
    k       = 0                         # used for indexing
    l       = 0                         # used for indexing
    m       = 0                         # used for indexing
    idx     = 0                         # used for indexing
    a       = zero(Float64)
    b       = zero(Float64)
    adb     = zero(Float64)             # = a / b
    c       = zero(Float64)
    d       = zero(Float64)
    betal   = zero(Float64)             # store lth component of bvec
    rss     = zero(Float64)             # residual sum of squares || Y - XB ||^2
    old_rss = oftype(zero(Float64),Inf) # previous residual sum of squares

    # obtain top r components of bvec in magnitude
#    selectpermk!(perm,bvec, r, p=p)
    selectperm!(perm, bvec, r, by=abs, rev=true, initialized=true)

    # compute partial residuals based on top r components of perm vector
    update_partial_residuals!(res, y, x, perm, bvec, r, n=n, p=p)

    # save value of RSS before starting algorithm
    rss = 0.5*sumabs2(res)

    # compute inner products of X and residuals
    # this is basically the negative gradient
    BLAS.gemv!('T', one(Float64), x, res, zero(Float64), df)

    # outer loop controls number of total iterations for algorithm run on one r
    for iter = 1:(max_iter)

        # output algorithm progress to console
        quiet || println("\titer = ", iter, ", RSS = ", rss)

        # middle loop tests each of top r parameters (by magnitude?)
        for i = abs(r-window+1):r

            # save information for current value of i
            l     = perm[i]
            betal = bvec[l]
            update_col!(tempn, x, l, n=n, p=p, a=one(Float64))  # tempn now holds X[:,l]

            # if necessary, compute inner products of current predictor against all other predictors
            # store this information in Dict inner
            # for current index, hold dot products in memory for duration of inner loop
            # the if/else statement below is the same as but faster than
            # > dotprods = get!(inner, l, BLAS.gemv('T', one(Float64), X, tempn))
            if !haskey(inner, l)
                inner[l] = BLAS.gemv('T', one(Float64), x, tempn)
            end
            copy!(dotprods,inner[l])

            # save values to determine best estimate for current predictor
            b   = nrmsq[l]
            a   = df[l] + betal*b
            adb = a / b
            k   = i

            # inner loop compares current predictor j against all remaining predictors j+1,...,p
            for j = (r+1):p
                idx = perm[j]
                c   = df[idx] + betal*dotprods[idx]
                d   = nrmsq[idx]

                # if current inactive predictor beats current active predictor,
                # then save info for swapping
                if c*c/d > a*adb + tol
                    a   = c
                    b   = d
                    k   = j
                    adb = a / b
                end
            end # end inner loop over remaining predictor set

            # now want to update residuals with current best predictor
            m = perm[k]
            update_col!(tempn2, x, m, n=n, p=p, a=one(Float64)) # tempn2 now holds X[:,m]
            axpymbz!(res, betal, tempn, adb, tempn2, p=n)

            # if necessary, compute inner product of current predictor against all other predictors
            # save in our Dict for future reference
            # compare in performance to
            # > tempp = get!(inner, m, BLAS.gemv('T', one(Float64), X, tempn2))
            if !haskey(inner, m)
                inner[m] = BLAS.gemv('T', one(Float64), x, tempn2)
            end
            copy!(tempp, inner[m])

            # also update df
            axpymbz!(df, betal, dotprods, adb, tempp, p=p)

            # now swap best predictor with current predictor
            j          = perm[i]
            perm[i]    = perm[k]
            perm[k]    = j
            bvec[m] = adb
            if k != i
                bvec[j] = zero(Float64)
            end

        end # end middle loop over predictors

        # update residual sum of squares
        rss = 0.5*sumabs2(res)

        # test for descent failure
        # if no descent failure, then test for convergence
        # if not converged, then save RSS and continue
        ascent    = rss > old_rss + tol
        converged = abs(old_rss - rss) / abs(old_rss + 1) < tol

        ascent && throw(error("Descent error detected at iteration $(iter)!\nOld RSS: $(old_rss)\nRSS: $(rss)"))
        (converged || ascent) && return bvec
        old_rss = rss
        isnan(rss) && throw(error("Objective function is NaN!"))
        isinf(rss) && throw(error("Objective function is Inf!"))

    end # end outer iteration loop

    # at this point, maximum iterations reached
    # warn and return bvec
    throw(error("Maximum iterations $(max_iter) reached! Return value may not be correct.\n"))
    return bvec

end # end exchange_leastsq!
