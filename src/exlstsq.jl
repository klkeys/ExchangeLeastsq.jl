"""
    update_partial_residuals!(r, y, x, perm, b, k)

A subroutine to compute the partial residuals `r = Y - X*b` in-place based on a permutation vector `perm` that indexes the nonzeroes in `b`.
"""
function update_partial_residuals!{T <: Float}(
    r    :: DenseVector{T},
    y    :: DenseVector{T},
    x    :: DenseMatrix{T},
    perm :: DenseArray{Int,1},
    b    :: DenseVector{T},
    k    :: Int
)
    copy!(r, y)
    @inbounds for j = 1:k
        idx = perm[j]
        @inbounds @simd for i in eachindex(y) 
            r[i] += -b[idx]*x[i,idx]
        end
    end
    return nothing
end



"""
    exchange_leastsq!(v, x, y, k) -> b

This function minimizes the residual sum of squares

    0.5*sumabs2( y - x*b )

subject to `b` having no more than `k` nonzero components. The function will compute a `b` for a given value of `k`.
For optimal accuracy, this function should be run for multiple values of `k` over a path of model sizes.
For optimal performance, allocate `v` once and then reuse it between different runs of `exchange_leastsq!`.

Arguments:

- `v` is the `ELSQVariables` object housing all temporary arrays, including `b` 
- `x` is the n x p statistical design matrix.
- `y` is the n-dimensional response vector.
- `k` is the desired number of nonzero components in `b`. 

Optional Arguments:

- `n` and `p` are the dimensions of `x`; the former defaults to `length(y)` while the latter defaults to `size(x,2)`.
- `window` is an `Int` to dictate the dimension of the search window for potentially exchanging predictors.
   Defaults to `k` (potentially exchange all current predictors). Decreasing this quantity tells the algorithm to search through
   fewer current active predictors, which can decrease compute time but can also degrade model recovery performance.
- `max_iter` is the maximum permissible number of iterations. Defaults to `100`.
- `tol` is the convergence tolerance. Defaults to `1e-6`.
- `quiet` is a `Bool` to control output. Defaults to `false` (full output).
"""
function exchange_leastsq!{T <: Float}(
    v        :: ELSQVariables{T},
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    k        :: Int;
    n        :: Int  = length(y),
    p        :: Int  = size(x,2),
    window   :: Int  = k,
    max_iter :: Int  = 100,
    tol      :: T    = convert(T, 1e-6),
    quiet    :: Bool = true 
)
    # initial value for previous residual sum of squares
    old_rss = oftype(tol,Inf) 

    # obtain top r components of bvec in magnitude
    selectperm!(v.perm, v.b, k, by=abs, rev=true, initialized=true)

    # compute partial residuals based on top r components of perm vector
    update_partial_residuals!(v.r, y, x, v.perm, v.b, k)

    # save value of RSS before starting algorithm
    rss = sumabs2(v.r) / 2

    # compute inner products of X and residuals
    # this is basically the negative gradient
    BLAS.gemv!('T', one(T), x, v.r, zero(T), v.df)

    # outer loop controls number of total iterations for algorithm run on one r
    for iter = 1:(max_iter)

        # output algorithm progress to console
        quiet || println("\titer = ", iter, ", RSS = ", rss)

        # middle loop tests each of top r parameters (by magnitude?)
        for i = abs(k-window+1):k

            # save information for current value of i
            l     = v.perm[i]
            betal = v.b[l]
            #update_col!(v.tempn, x, l)  # tempn now holds X[:,l]
            copy!(v.tempn, sub(x, :, l))

            # if necessary, compute inner products of current predictor against all other predictors
            # store this information in Dict inner
            # for current index, hold dot products in memory for duration of inner loop
            # the if/else statement below is the same as but faster than
            # > dotprods = get!(inner, l, BLAS.gemv('T', one(T), X, tempn))
#            if !haskey(v.inner, l)
#                v.inner[l] = BLAS.gemv('T', one(T), x, v.tempn)
#            end
#            copy!(v.dotprods, v.inner[l])
            get_inner_product!(v.dotprods, v.tempn, v, x, l)

            # subroutine compares current predictor i against all predictors k+1, k+2, ..., p
            # these predictors are candidates for inclusion in set
            # _exlstsq_innerloop! find best new predictor r
            a, b, r, adb = _exlstsq_innerloop!(v, k, i, p, tol)

            # now want to update residuals with current best predictor
            m = update_current_best_predictor!(v, x, betal, adb, r)

            # if necessary, compute inner product of current predictor against all other predictors
            # save in our Dict for future reference
            # compare in performance to
            # > tempp = get!(inner, m, BLAS.gemv('T', one(T), X, tempn2))
#            if !haskey(v.inner, m)
#                v.inner[m] = BLAS.gemv('T', one(T), x, v.tempn2)
#            end
#            copy!(v.tempp, v.inner[m])
            get_inner_product!(v.tempp, v.tempn2, v, x, m)

            # also update df
            axpymbz!(v.df, betal, v.dotprods, adb, v.tempp)

            # now swap best predictor with current predictor
            _swap_predictors!(v, i, r, m, adb)

        end # end middle loop over predictors

        # update residual sum of squares
        rss = sumabs2(v.r) / 2

        # test for descent failure
        # if no descent failure, then test for convergence
        # if not converged, then save RSS and check finiteness
        # if not converged and still finite, then save RSS and continue 
        ascent    = rss > old_rss + tol
        converged = abs(old_rss - rss) / abs(old_rss + 1) < tol
        old_rss = rss
        ascent && print_descent_error(iter, old_rss, rss)
        converged && return nothing 

        check_finiteness(rss)

    end # end outer iteration loop

    # at this point, maximum iterations reached
    # warn and return
    print_maxiter(max_iter, rss)
    return nothing

end # end exchange_leastsq!

"""
    exlstsq(x, y) -> ELSQResults 

For a statistical model `b`, `exlstsq` minimizes the residual sum of squares

    0.5*sumabs2( y - x*b )

subject to `b` having no more than `k` nonzero components. `exlstsq` will compute a `b` for several model sizes `k`.

Arguments:

- `x` is the n x p statistical design matrix.
- `y` is the n-dimensional response vector.

Optional Arguments:

- `v` is the `ELSQVariables` object housing all temporary arrays, including `b`. 
* `models` is the integer vector of model sizes to test. It defaults to `collect(1:p)`, where `p = min(20, size(x,2))`.
- `window` is an `Int` to dictate the maximum size of the search window for potentially exchanging predictors.
   Defaults to `maximum(models)` (all predictors for each model size are exchangeable). 
   Decreasing this quantity tells the algorithm to search through fewer current active predictors, 
   which can decrease compute time but can also degrade model recovery performance.
- `max_iter` is the maximum permissible number of iterations. Defaults to `100`.
- `tol` is the convergence tolerance. Defaults to `1e-6`.
- `quiet` is a `Bool` to control output. Defaults to `false` (full output).
"""
function exlstsq{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T};
    v        :: ELSQVariables{T} = ELSQVariables(x, y), 
    models   :: DenseVector{Int} = collect(1:min(20, size(x,2))),
    window   :: Int  = maximum(models),
    max_iter :: Int  = 100,
    tol      :: T    = convert(T, 1e-6),
    quiet    :: Bool = true 
)
    # dimensions of problem
    nmodels = length(models)
    n,p = size(x)

    # error checking
    errorcheck(x, y, tol, max_iter, window, p)

    # initialize sparse matrix to return
    betas = spzeros(T, p, nmodels)

    # loop through models
    for i in models

        # monitor output
        quiet || println("Testing model size $i.")
        exchange_leastsq!(v, x, y, i, window=min(window,i), max_iter=max_iter, tol=tol, quiet=quiet, n=n, p=p)
        betas[:,i] = sparse(sdata(v.b))
    end

    # return matrix of betas
    return betas 
end
