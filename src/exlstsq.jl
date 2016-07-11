"""
    update_partial_residuals!(r, y, x, perm, b, k [, n=length(r), p=length(b)])

A subroutine to compute the partial residuals `r = Y - X*b` in-place based on a permutation vector `perm` that indexes the nonzeroes in `b`.
"""
function update_partial_residuals!{T <: Float}(
    r    :: DenseVector{T},
    y    :: DenseVector{T},
    x    :: DenseMatrix{T},
    perm :: DenseArray{Int,1},
    b    :: DenseVector{T},
    k    :: Int;
    p    :: Int = length(b)
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
- `k` is the desired number of nonzero components in `bvec`.

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
    quiet    :: Bool = false
)

    # declare algorithm variables
    i       = 0               # used for iterations
    iter    = 0               # used for outermost loop
    j       = 0               # used for iterations
    r       = 0               # used for indexing
    l       = 0               # used for indexing
    m       = 0               # used for indexing
    idx     = 0               # used for indexing
    a       = zero(T)
    b       = zero(T)
    adb     = zero(T)         # = a / b
    c       = zero(T)
    d       = zero(T)
    betal   = zero(T)         # store lth component of bvec
    rss     = zero(T)         # residual sum of squares || Y - XB ||^2
    old_rss = oftype(tol,Inf) # previous residual sum of squares

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
            update_col!(v.tempn, x, l)  # tempn now holds X[:,l]

            # if necessary, compute inner products of current predictor against all other predictors
            # store this information in Dict inner
            # for current index, hold dot products in memory for duration of inner loop
            # the if/else statement below is the same as but faster than
            # > dotprods = get!(inner, l, BLAS.gemv('T', one(T), X, tempn))
            if !haskey(v.inner, l)
                v.inner[l] = BLAS.gemv('T', one(T), x, v.tempn)
            end
            copy!(v.dotprods, v.inner[l])

            # save values to determine best estimate for current predictor
            b   = v.nrmsq[l]
            a   = v.df[l] + betal*b
            adb = a / b
            r   = i

            # inner loop compares current predictor j against all remaining predictors j+1,...,p
            for j = (k+1):p
                idx = v.perm[j]
                c   = v.df[idx] + betal*v.dotprods[idx]
                d   = v.nrmsq[idx]

                # if current inactive predictor beats current active predictor,
                # then save info for swapping
                if c*c/d > a*adb + tol
                    a   = c
                    b   = d
                    r   = j
                    adb = a / b
                end
            end # end inner loop over remaining predictor set

            # now want to update residuals with current best predictor
            m = v.perm[r]
            update_col!(v.tempn2, x, m) # tempn2 now holds X[:,m]
            axpymbz!(v.r, betal, v.tempn, adb, v.tempn2)

            # if necessary, compute inner product of current predictor against all other predictors
            # save in our Dict for future reference
            # compare in performance to
            # > tempp = get!(inner, m, BLAS.gemv('T', one(T), X, tempn2))
            if !haskey(v.inner, m)
                v.inner[m] = BLAS.gemv('T', one(T), x, v.tempn2)
            end
            copy!(v.tempp, v.inner[m])

            # also update df
            axpymbz!(v.df, betal, v.dotprods, adb, v.tempp)

            # now swap best predictor with current predictor
            j         = v.perm[i]
            v.perm[i] = v.perm[r]
            v.perm[r] = j
            v.b[m]    = adb
            if r != i
                v.b[j] = zero(T)
            end

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



function exlstsq{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    models   :: DenseVector{Int};
    v        :: ELSQVariables{T} = ELSQVariables(x, y), 
    window   :: Int  = maximum(models),
    max_iter :: Int  = 100,
    tol      :: T    = convert(T, 1e-6),
    quiet    :: Bool = true 
)
    # dimensions of problem
    nmodels = length(models)
    n,p = size(x)

    # error checking
#    errorcheck(x, y, k, tol, max_iter, window, p)
    errorcheck(x, y, tol, max_iter, window, p)

    # initialize sparse matrix to return
    betas = spzeros(T, p, nmodels)

    # loop through models
    for i in models
        exchange_leastsq!(v, x, y, i, window=i, max_iter=max_iter, tol=tol, quiet=quiet, n=n, p=p)
        betas[:,i] = sparse(v.b)
    end

    # return matrix of betas
    return betas 
end
