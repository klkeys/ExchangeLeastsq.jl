"""
    exchange_leastsq!(b, x::BEDFile, y, perm, r) -> b

If called with a `BEDFile` object `x`, then `exchange_leastsq!()` typeasserts all arrays as `SharedArray`s. The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.
- `Xb` is an `n`-vector storing the estimated response `x*b`.
- `indices` is a `BitArray` that indexes the nonzeroes in `b`.
- `mask_n` is an `Int` array that applies a bitmask to the samples. This enables efficient crossvalidation calculations by avoiding subsetting operations on the `BEDFile` `x`.
  `mask_n` should only contain `0`s (exclude) and `1`s (include). Defaults to a `SharedArray` of `n` ones, which includes all data.
- `n64` is a floating point copy of problem dimension `n` used in intermediate calculations to avoid repeated `Int` conversion. Defaults to `convert(T,n)`.

`BEDFile` objects require on-the-fly column standardization, which eliminates the need for the `nrmsq` argument.
"""
function exchange_leastsq!{T <: Float}(
    b        :: SharedVector{T},
    x        :: BEDFile,
    y        :: SharedVector{T},
    perm     :: SharedVector{Int},
    r        :: Int;
    inner    :: Dict{Int,SharedVector{T}} = Dict{Int,SharedVector{T}}(),
    pids     :: DenseVector{Int} = procs(),
    means    :: SharedVector{T}  = mean(T,x, shared=true, pids=pids),
    invstds  :: SharedVector{T}  = invstd(x, means, pids=pids),
    n        :: Int              = length(y),
    p        :: Int              = size(x,2),
    df       :: SharedVector{T}  = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids),
    dotprods :: SharedVector{T}  = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids),
    tempp    :: SharedVector{T}  = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids),
    Xb       :: SharedVector{T}  = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids),
    res      :: SharedVector{T}  = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids),
    tempn    :: SharedVector{T}  = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids),
    tempn2   :: SharedVector{T}  = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids),
    mask_n   :: SharedVector{T}  = SharedArray(Int,     n, init = S -> S[localindexes(S)] = one(Int),      pids=pids),
    indices  :: BitArray{1}      = falses(p),
    window   :: Int              = r,
    max_iter :: Int              = 100,
    n64      :: T                = convert(T, n),
    tol      :: T                = 1e-6,
    quiet    :: Bool             = false
)

    # error checking
    n == length(tempn)    || throw(DimensionMismatch("length(y) != length(tempn)"))
    n == length(tempn2)   || throw(DimensionMismatch("length(y) != length(tempn2)"))
    n == length(res)      || throw(DimensionMismatch("length(y) != length(res)"))
    n == length(mask_n)   || throw(DimensionMismatch("length(y) != length(res)"))
    p == length(b)        || throw(DimensionMismatch("Number of predictors != length(b)"))
    p == length(df)       || throw(DimensionMismatch("length(b) != length(df)"))
    p == length(tempp)    || throw(DimensionMismatch("length(b) != length(tempp)"))
    p == length(dotprods) || throw(DimensionMismatch("length(b) != length(dotprods)"))
    p == length(perm)     || throw(DimensionMismatch("length(b) != length(perm)"))
    0 <= r <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(b)"))
    tol >= eps(T)   || throw(ArgumentError("Global tolerance must exceed machine precision"))
    max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
    0 <= window <= r      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))

    # argument mask_n should only have 0 or 1
    sum((mask_n .== 1) $ (mask_n .== 0)) == n || throw(ArgumentError("Argument mask_n can only contain 1s and 0s"))

    # declare algorithm variables
    i       = 0               # used for iterations
    iter    = 0               # used for outermost loop
    j       = 0               # used for iterations
    k       = 0               # used for indexing
    l       = 0               # used for indexing
    m       = 0               # used for indexing
    idx     = 0               # used for indexing
    a       = zero(T)
    adb     = zero(T)         # = a / b
    c       = zero(T)
    betal   = zero(T)         # store lth component of b
    rss     = zero(T)         # residual sum of squares || Y - XB ||^2
    old_rss = oftype(tol,Inf) # previous residual sum of squares

    # obtain top r components of b in magnitude
    selectperm!(perm, sdata(b), k, by=abs, rev=true, initialized=true)
    update_indices!(indices, b, p=p)

    # update X*b
    xb!(Xb,x,b,indices,r,mask_n, means=means, invstds=invstds, pids=pids)

    # update residuals based on Xb
    difference!(res, y, Xb, n=n)
    mask!(res, mask_n, 0, zero(T), n=n)

    # save value of RSS before starting algorithm
    rss = sumabs2(res) / 2

    # compute inner products of X and residuals
    # this is basically the negative gradient
    xty!(df, x, res, mask_n, means=means, invstds=invstds, pids=pids)

    # outer loop controls number of total iterations for algorithm run on one r
    for iter = 1:(max_iter)

        # output algorithm progress to console
        quiet || println("\titer = ", iter, ", RSS = ", rss)

        # middle loop tests each of top r parameters (by magnitude?)
        for i = abs(r-window+1):r

            # save information for current value of i
            l     = perm[i]
            betal = b[l]
            decompress_genotypes!(tempn, x, l, means, invstds) # tempn now holds X[:,l]
            mask!(tempn, mask_n, 0, zero(T), n=n)

            # if necessary, compute inner products of current predictor against all other predictors
            # store this information in Dict inner
            # for current index, hold dot products in memory for duration of inner loop
            # the if/else statement below is the same as but faster than
            # > dotprods = get!(inner, l, BLAS.gemv('T', 1.0, X, tempn))
            if !haskey(inner, l)
                inner[l] = xty(x, tempn, mask_n, means=means, invstds=invstds, pids=pids)
            end
            copy!(dotprods,inner[l])

            # save values to determine best estimate for current predictor
            a   = df[l] + betal*n64
            adb = a / n64
            k   = i

            # inner loop compares current predictor j against all remaining predictors j+1,...,p
            for j = (r+1):p
                idx = perm[j]
                c   = df[idx] + betal*dotprods[idx]

                # if current inactive predictor beats current active predictor,
                # then save info for swapping
                if c*c/n64 > a*adb + tol
                    a   = c
                    k   = j
                    adb = a / n64
                end
            end # end inner loop over remaining predictor set

            # now want to update residuals with current best predictor
            m = perm[k]
            decompress_genotypes!(tempn2, x, m, means, invstds) # tempn now holds X[:,l]
            mask!(tempn2, mask_n, 0, zero(T), n=n)
            axpymbz!(res, betal, tempn, adb, tempn2, p=n)
            mask!(res, mask_n, 0, zero(T), n=n)

            # if necessary, compute inner product of current predictor against all other predictors
            # save in our Dict for future reference
            if !haskey(inner, m)
                inner[m] = xty(x, tempn2, mask_n, means=means, invstds=invstds, pids=pids)
            end
            copy!(tempp, inner[m])

            # also update df
            axpymbz!(df, betal, dotprods, adb, tempp, p=p)

            # now swap best predictor with current predictor
            j       = perm[i]
            perm[i] = perm[k]
            perm[k] = j
            b[m]    = adb
            if k != i
                b[j] = zero(T)
            end

        end # end middle loop over predictors

        # update residual sum of squares
        mask!(res, mask_n, 0, zero(T), n=n)
        rss = sumabs2(res) / 2

        # test for numerical instability
        isnan(rss) && throw(error("Objective function is NaN!"))
        isinf(rss) && throw(error("Objective function is Inf!"))

        # test for descent failure
        # if no descent failure, then test for convergence
        # if not converged, then save RSS and continue
        ascent    = rss > old_rss + tol
        converged = abs(old_rss - rss) / abs(old_rss + 1) < tol

        ascent && throw(error("Descent error detected at iteration $(iter)!\nOld RSS: $(old_rss)\nRSS: $(rss)"))
        (converged || ascent) && return b
        old_rss = rss

    end # end outer iteration loop

    # at this point, maximum iterations reached
    # warn and return b
    warn("Maximum iterations $(max_iter) reached! Return value may not be correct.\n")
    return b

end # end exchange_leastsq


"""
    one_fold(x::BEDFile, y, path_length, folds, fold) -> mses

If called with a `BEDFile` object `x`, then `one_fold()` typeasserts all arrays as `SharedArray`s. The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.

Because `BEDFile` objects require on-the-fly column standardization, the `nrmsq` argument is not used here.
"""
function one_fold{T <: Float}(
    x           :: BEDFile,
    y           :: SharedVector{T},
    path_length :: Int,
    folds       :: SharedVector{Int},
    fold        :: Int;
    pids        :: DenseVector{Int} = procs(),
    means       :: SharedVector{T}  = mean(T, x, shared=true, pids=pids),
    invstds     :: SharedVector{T}  = invstd(x, y=means, pids=pids),
    tol         :: T    = 1e-6,
    p           :: Int  = size(x,2),
    max_iter    :: Int  = 100,
    window      :: Int  = 20,
    quiet       :: Bool = true
)

    # find testing indices
    test_idx = folds .== fold

    # preallocate vector for output
    errors = zeros(T, sum(test_idx))

    # train_idx is the vector that indexes the TRAINING set
    train_idx = convert(SharedVector{Int}, !test_idx)
    test_idx  = convert(Vector{Int}, test_idx)

    # how big is training sample?
    n = sum(train_idx)

    # declare all temporary arrays
    inner    = Dict{Int,SharedVector{T}}()
    b        = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids)
    perm     = SharedArray(T, p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids)
    df       = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids) # X'(Y - Xbeta)
    tempp    = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids) # temporary array of length p
    dotprods = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids) # hold in memory the dot products for current index
    bout     = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids) # output array for beta
    tempn    = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids) # temporary array of length n
    tempn2   = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids) # temporary array of length n
    res      = SharedArray(T, n, init = S -> S[localindexes(S)] = zero(T), pids=pids) # Y - Xbeta
    indices  = falses(p)                                                                          # indicate nonzero components of beta
    n64      = convert(T, n)

    # will return betas in a sparse matrix
    betas     = spzeros(T, p, path_length)

    # loop over each element of path
    @inbounds for i = 1:path_length

        # compute the regularization path on the training set
        exchange_leastsq!(b, x, y, perm, i, inner=inner, max_iter=max_iter, quiet=quiet, n=n, p=p, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window = min(window, i), n64=n64, indices=indices, pids=pids, means=means, invstds=invstds, n=n, p=p, Xb=Xb, tol=tol, mask_n = train_idx)

        # find the support of bout
        update_indices!(indices, b, p=p)

        # compute estimated response Xb with $(path[i]) nonzeroes
        xb!(Xb,x,b,indices,path[i],test_idx, means=means, invstds=invstds, pids=pids)

        # compute residuals
        difference!(r,y,Xb)
        mask!(r,test_idx,0,zero(T),n=n)

        # compute out-of-sample error as squared residual averaged over size of test set
        errors[i] = sumabs2(r) / (2*test_size)

        # store b
        betas[:,i] = sparsevec(b)
    end

    # compute the mean out-of-sample error for the TEST set
    return errors
end


"""
    cv_exlstsq(x::BEDFile, y, path_length, q [, refit=false]) -> mses [, b, bidx]

If called with a `BEDFile` object `x`, then `one_fold()` typeasserts all arrays as `SharedArray`s. The additional optional arguments are:

- `pids`, a vector of process IDs. Defaults to `procs()`.
- `means`, a vector of SNP means. Defaults to `mean(T, x, shared=true, pids=procs()`.
- `invstds`, a vector of SNP precisions. Defaults to `invstd(x, means, shared=true, pids=procs()`.

`BEDFile` objects require on-the-fly column standardization, which renders the `nrmsq` argument unnecessary. 
"""
function cv_exlstsq{T <: Float}(
    x           :: BEDFile,
    y           :: SharedVector{T},
    path_length :: Int,
    q           :: Int;
    pids        :: DenseVector{Int}  = procs(),
    means       :: SharedVector{T}   = mean(T, x, shared=true, pids=pids),
    invstds     :: SharedVector{T}   = invstd(x, y=means, shared=true, pids=pids),
    folds       :: SharedVector{Int} = cv_get_folds(sdata(y),q),
    tol         :: T    = convert(T, 1e-6),
    n           :: Int  = length(y),
    p           :: Int  = size(x,2),
    max_iter    :: Int  = 100,
    window      :: Int  = 20,
    refit       :: Bool = false,
    quiet       :: Bool = true
)

    0 <= path_length <= p || throw(ArgumentError("Path length must be positive and cannot exceed number of predictors"))

    # preallocate vectors used in xval
    mses    = zeros(T, path_length)   # vector to save mean squared errors

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # the @sync macro ensures that we wait for all of them to finish before proceeding
    @sync @inbounds for i = 1:q

        # one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression)
        mses[i] = @fetch(one_fold(x, y, path_length, folds, i, max_iter=max_iter, quiet=quiet, window=window, n=n, p=p, means=means, invstds=invstds, tol=tol))
    end

    # average mses
    mses ./= q

    # store a vector for path
    path = collect(1:path_length)

    # what is the best model size?
    k = convert(Int, floor(mean(path[mses .== minimum(mses)])))

    # print results
    quiet || begin
        println("\n\nCrossvalidation Results:")
        println("k\tMSE")
        for i = 1:length(mses)
            println(path[i], "\t", mses[i])
        end
        println("\nThe lowest MSE is achieved at k = ", k)
    end

    # recompute ideal model
    if refit

        # initialize beta vector
        fill!(sdata(b), zero(T))
        perm = collect(1:p)

        # first use exchange algorithm to extract model
        exchange_leastsq!(b, x, y, perm, k, max_iter=max_iter, quiet=quiet, n=n, p=p, tol=tol, window=k)

        # which components of beta are nonzero?
        # cannot use binary indices here since we need to return Int indices
        inferred_model = b .!= zero(T)
        bidx = find(b)

        # allocate the submatrix of x corresponding to the inferred model
        x_inferred = zeros(T, n, sum(inferred_model))
        decompress_genotypes!(x_inferred, x, inferred_model, means=means, invstds=invstds)

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
        # return it with the vector of MSEs
        Xty = BLAS.gemv('T', one(T), x_inferred, y)
        XtX = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
        b2   = XtX \ Xty
        return mses, b2, bidx
    end

    return mses
end
