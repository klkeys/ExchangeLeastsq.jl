function axpymbz!(
    j :: Int,
    y :: DenseVector{Float32},
    a :: Float32,
    x :: DenseVector{Float32},
    b :: Float32,
    z :: DenseVector{Float32}
)
    y[j] + a*x[j] - b*z[j]
end

function axpymbz!(
    y :: Vector{Float32},
    a :: Float32,
    x :: Vector{Float32},
    b :: Float32,
    z :: Vector{Float32};
    p :: Int = length(y)
)
    @inbounds for i = 1:p
        y[i] = axpymbz!(i, y, a, x, b, z)
    end
end


function myrange(q::SharedVector{Float32})
    idx = indexpids(q)
    if idx == 0
        # This worker is not assigned a piece
        return one(Float32), one(Float32)
    end
    nchunks = length(procs(q))
    splits = [round(Int,s) for s in linspace(0,length(q),nchunks+1)]
    return splits[idx]+1 : splits[idx+1]
end

function axpymbz_shared_chunk!(
    y      :: SharedVector{Float32},
    a      :: Float32,
    x      :: SharedVector{Float32},
    b      :: Float32,
    z      :: SharedVector{Float32},
    irange :: UnitRange{Int}
)
    @inbounds for i in irange
        y[i] = axpymbz!(i,y,a,x,b,z)
    end
end

axpymbz_shared!(y::SharedVector{Float32}, a::Float32, x::SharedVector{Float32}, b::Float32, z::SharedVector{Float32}) = axpymbz_shared_chunk!(y,a,x,b,z,myrange(y))

function axpymbz!(
    y::SharedVector{Float32},
    a::Float32,
    x::SharedVector{Float32},
    b::Float32,
    z::SharedVector{Float32}
)
    @sync begin
        for p in procs(y)
            @async remotecall_wait(p, axpymbz_shared!, y, a, x, b, z)
        end
    end
end

function exchange_leastsq!(
    bvec     :: DenseVector{Float32},
    x        :: DenseMatrix{Float32},
    y        :: DenseVector{Float32},
    perm     :: DenseVector{Int},
    r        :: Int;
    inner    :: Dict{Int, DenseVector{Float32}} = Dict{Int,DenseVector{Float32}}(),
    n        :: Int = length(y),
    p        :: Int = size(x,2),
    nrmsq    :: DenseVector{Float32} = vec(sumabs2(x,1)),
    df       :: DenseVector{Float32} = zeros(Float32, p),
    dotprods :: DenseVector{Float32} = zeros(Float32, p),
    tempp    :: DenseVector{Float32} = zeros(Float32, p),
    res      :: DenseVector{Float32} = zeros(Float32, n),
    tempn    :: DenseVector{Float32} = zeros(Float32, n),
    tempn2   :: DenseVector{Float32} = zeros(Float32, n),
    window   :: Int     = r,
    max_iter :: Int     = 100,
    tol      :: Float32 = 1f-4,
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
    a       = zero(Float32)
    b       = zero(Float32)
    adb     = zero(Float32)             # = a / b
    c       = zero(Float32)
    d       = zero(Float32)
    betal   = zero(Float32)             # store lth component of bvec
    rss     = zero(Float32)             # residual sum of squares || Y - XB ||^2
    old_rss = oftype(zero(Float32),Inf) # previous residual sum of squares

    # obtain top r components of bvec in magnitude
#    selectpermk!(perm,bvec, r, p=p)
    selectperm!(perm, bvec, r, by=abs, rev=true, initialized=true)

    # compute partial residuals based on top r components of perm vector
    RegressionTools.update_partial_residuals!(res, y, x, perm, bvec, r, n=n, p=p)

    # save value of RSS before starting algorithm
    rss = 0.5f0*sumabs2(res)

    # compute inner products of X and residuals
    # this is basically the negative gradient
    BLAS.gemv!('T', one(Float32), x, res, zero(Float32), df)

    # outer loop controls number of total iterations for algorithm run on one r
    for iter = 1:(max_iter)

        # output algorithm progress to console
        quiet || println("\titer = ", iter, ", RSS = ", rss)

        # middle loop tests each of top r parameters (by magnitude?)
        for i = abs(r-window+1):r

            # save information for current value of i
            l     = perm[i]
            betal = bvec[l]
            update_col!(tempn, x, l, n=n, p=p, a=one(Float32))  # tempn now holds X[:,l]

            # if necessary, compute inner products of current predictor against all other predictors
            # store this information in Dict inner
            # for current index, hold dot products in memory for duration of inner loop
            # the if/else statement below is the same as but faster than
            # > dotprods = get!(inner, l, BLAS.gemv('T', one(Float32), X, tempn))
            if !haskey(inner, l)
                inner[l] = BLAS.gemv('T', one(Float32), x, tempn)
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
            update_col!(tempn2, x, m, n=n, p=p, a=one(Float32)) # tempn2 now holds X[:,m]
            axpymbz!(res, betal, tempn, adb, tempn2, p=n)

            # if necessary, compute inner product of current predictor against all other predictors
            # save in our Dict for future reference
            # compare in performance to
            # > tempp = get!(inner, m, BLAS.gemv('T', one(Float32), X, tempn2))
            if !haskey(inner, m)
                inner[m] = BLAS.gemv('T', one(Float32), x, tempn2)
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
                bvec[j] = zero(Float32)
            end

        end # end middle loop over predictors

        # update residual sum of squares
        rss = 0.5f0*sumabs2(res)

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

end # end exchange_leastsq
