function one_fold(
    x           :: DenseMatrix{Float32},
    y           :: DenseVector{Float32},
    path_length :: Int,
    folds       :: DenseVector{Int},
    fold        :: Int;
    nrmsq       :: DenseVector{Float32} = vec(sumsq(x,1)),
    tol         :: Float32 = 1f-6,
    p           :: Int     = size(x,2),
    max_iter    :: Int     = 100,
    window      :: Int     = p,
    quiet       :: Bool    = true
)

    # find testing indices
    test_idx = folds .== fold

    # preallocate vector for output
    myerrors = zeros(Float32, sum(test_idx))

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # how big is training sample?
    n = sum(train_idx)

    # allocate the arrays for the training set
    x_train   = x[train_idx,:]
    y_train   = y[train_idx]

    # allocate Dict to store inner prodcuts
    inner = Dict{Int,DenseVector{Float32}}()

    # declare all temporary arrays
    b          = zeros(Float32, p)
    perm       = collect(1:p)
    res        = zeros(Float32, n)  # Y - Xbeta
    df         = zeros(Float32, p)  # X'(Y - Xbeta)
    tempn      = zeros(Float32, n)  # temporary array of length n
    tempn2     = zeros(Float32, n)  # temporary array of length n
    tempp      = zeros(Float32, p)  # temporary array of length p
    dotprods   = zeros(Float32, p)  # hold in memory the dot products for current index
#    bnonzeroes = falses(p)          # indicate nonzero components of beta
#    bout       = zeros(Float32, p)  # output array for beta

    # will return a sparse matrix of betas
    betas      = spzeros(Float32, p,path_length)

    # loop over each element of path
    @inbounds for i = 1:path_length

        # compute the regularization path on the training set
        exchange_leastsq!(b, x_train, y_train, perm, i, inner=inner, max_iter=max_iter, quiet=quiet, n=n, p=p, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window = min(window, i), nrmsq=nrmsq, tol=tol)

#        # find the support of bout
#        update_indices!(bnonzeroes, bout, p=p)
#
#        # subset training indices of x with support
#        x_refit    = x_train[:,bnonzeroes]
#
#        # perform ordinary least squares to refit support of bout
#        Xty        = BLAS.gemv('T', one(Float32), x_refit, y_train)
#        XtX        = BLAS.gemm('T', 'N', one(Float32), x_refit, x_refit)
#        b_refit    = XtX \ Xty
#
#        # put refitted values back in bout
#        bout[bnonzeroes] = b_refit

#        # copy bout back to b
#        copy!(b, bout)

        # store b
#        update_col!(betas, b, i, n=p, p=path_length, a=one(Float32))
        betas[:,i] = sparsevec(b)
    end

#    # sparsify the betas
#    betas = sparse(betas)

    # compute the mean out-of-sample error for the TEST set
    errors  = vec(0.5f0*sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ length(test_idx)

    return errors
end




function cv_exlstsq(
    x             :: DenseMatrix{Float32},
    y             :: DenseVector{Float32},
    path_length   :: Int,
    q             :: Int;
    nrmsq         :: DenseVector{Float32} = vec(sumabs2(x,1)),
    folds         :: DenseVector{Int}     = cv_get_folds(sdata(y),q),
    tol           :: Float32 = 1f-6,
    n             :: Int     = length(y),
    p             :: Int     = size(x,2),
    max_iter      :: Int     = 100,
    window        :: Int     = 20,
    compute_model :: Bool    = false,
    quiet         :: Bool    = true
)

    0 <= path_length <= p || throw(ArgumentError("Path length must be positive and cannot exceed number of predictors"))

    # preallocate vectors used in xval
    mses = zeros(Float32, path_length)   # vector to save mean squared errors

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # the @sync macro ensures that we wait for all of them to finish before proceeding
    @sync @inbounds for i = 1:q

        # one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression)
        # @spawn(one_fold(...)) returns a RemoteRef to the result
        # store that RemoteRef so that we can query the result later
        mses[i] = @fetch(one_fold(x, y, path_length, folds, i, max_iter=max_iter, quiet=quiet, window=window, p=p, nrmsq=nrmsq, tol=tol))
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
        @inbounds for i = 1:length(mses)
            println(path[i], "\t", mses[i])
        end
        println("\nThe lowest MSE is achieved at k = ", k)
    end

    # recompute ideal model
    if compute_model

        # initialize beta vector
        bp = zeros(Float32,p)
        perm = collect(1:p)

        # first use exchange algorithm to extract model
        exchange_leastsq!(bp, x, y, perm, k, max_iter=max_iter, quiet=quiet, n=n, p=p, tol=tol, nrmsq=nrmsq, window=k)

        # which components of beta are nonzero?
        # cannot use binary indices here since we need to return Int indices
        bidx = find( function f(x) x.!= zero(Float32); end, bp)

        # allocate the submatrix of x corresponding to the inferred model
        x_inferred = x[:,bidx]

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
        # return it with the vector of MSEs
        Xty = BLAS.gemv('T', one(Float32), x_inferred, y)
        XtX = BLAS.gemm('T', 'N', one(Float32), x_inferred, x_inferred)
        b = XtX \ Xty
        return mses, b, bidx
    end

    return mses
end
