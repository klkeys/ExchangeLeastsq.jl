"""
    one_fold(x,y,path_length,folds,fold) -> Vector{Float}

For a regularization `path_length`,
this function computes an out-of-sample error for fold given by `fold` in a `q`-fold crossvalidation scheme.
The folds are indexed by the `Int` vector `folds`.

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `path_length` is an `Int` to determine the length of the regularization path to compute.
- `folds` is the `Int` array that indicates which data to hold out for testing.
- `fold` indexes the current fold.

Optional Arguments:

- `nrmsq` is the vector to store the squared norms of the columns of `x`. Defaults to `vec(sumabs2(x,1))`.
- `tol` is the convergence tolerance to pass to the path computations. Defaults to `1e-6`.
- `p` is the number of predictors in `x`. Defaults to `size(x,2)`.
- `max_iter` caps the number of permissible iterations in the algorithm. Defaults to `100`.
- `window` is an `Int` to dictate the dimension of the search window for potentially exchanging predictors.
   Defaults to `p` (potentially exchange all predictors). Decreasing this quantity tells the algorithm to search through
   fewer current active predictors, which can decrease compute time but can also degrade model recovery performance.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).
"""
function one_fold(
    x           :: DenseMatrix{Float64},
    y           :: DenseVector{Float64},
    path_length :: Int,
    folds       :: DenseVector{Int},
    fold        :: Int;
    nrmsq       :: DenseVector{Float64} = vec(sumsq(x,1)),
    tol         :: Float64 = 1e-6,
    p           :: Int     = size(x,2),
    max_iter    :: Int     = 100,
    window      :: Int     = p,
    quiet       :: Bool    = true
)

    # find testing indices
    test_idx = folds .== fold

    # preallocate vector for output
    myerrors = zeros(Float64, sum(test_idx))

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # how big is training sample?
    n = sum(train_idx)

    # allocate the arrays for the training set
    x_train   = x[train_idx,:]
    y_train   = y[train_idx]

    # allocate Dict to store inner prodcuts
    inner = Dict{Int,DenseVector{Float64}}()

    # declare all temporary arrays
    b          = zeros(Float64, p)
    perm       = collect(1:p)
    res        = zeros(Float64, n)  # Y - Xbeta
    df         = zeros(Float64, p)  # X'(Y - Xbeta)
    tempn      = zeros(Float64, n)  # temporary array of length n
    tempn2     = zeros(Float64, n)  # temporary array of length n
    tempp      = zeros(Float64, p)  # temporary array of length p
    dotprods   = zeros(Float64, p)  # hold in memory the dot products for current index
#    bnonzeroes = falses(p)          # indicate nonzero components of beta
#    bout       = zeros(Float64, p)  # output array for beta

    # will return a sparse matrix of betas
    betas      = spzeros(Float64, p,path_length)

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
#        Xty        = BLAS.gemv('T', one(Float64), x_refit, y_train)
#        XtX        = BLAS.gemm('T', 'N', one(Float64), x_refit, x_refit)
#        b_refit    = XtX \ Xty
#
#        # put refitted values back in bout
#        bout[bnonzeroes] = b_refit

#        # copy bout back to b
#        copy!(b, bout)

        # store b
#        update_col!(betas, b, i, n=p, p=path_length, a=one(Float64))
        betas[:,i] = sparsevec(b)
    end

#    # sparsify the betas
#    betas = sparse(betas)

    # compute the mean out-of-sample error for the TEST set
    errors  = vec(0.5*sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ length(test_idx)

    return errors
end

"""
    pfold(x, y, path, folds, numfolds) -> Vector

This function is the parallel execution kernel in `cv_exlstsq`. It is not meant to be called outside of `cv_exlstsq`.
It will distribute `numfolds` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold` for each fold.
Each fold will compute a regularization path `1:path`.
`pfold` collects the vectors of MSEs returned by calling `one_fold` for each process, reduces them, and returns their average across all folds.
"""
function pfold(
    x         :: SharedMatrix{Float64},
    y         :: SharedVector{Float64},
	path      :: Int, 
	folds     :: SharedVector{Int},
	numfolds  :: Int;
    n         :: Int              = length(y),
    p         :: Int              = size(x,2),
	pids      :: DenseVector{Int} = procs(),
    nrmsq     :: DenseVector{Float64} = vec(sumsq(x,1)),
    tol       :: Float64          = 1e-6,
	max_iter  :: Int              = 100,
    window    :: Int              = p, 
	quiet     :: Bool             = true,
	refit     :: Bool             = true,
)
	# how many CPU processes can pfold use?
	np = length(pids)

	# report on CPU processes
	quiet || println("pfold: np = ", np)
	quiet || println("pids = ", pids)

	# set up function to share state (indices of folds)
	i = 1
	nextidx() = (idx=i; i+=1; idx)

	# preallocate cell array for results
	results = cell(numfolds)

	# master process will distribute tasks to workers
	# master synchronizes results at end before returning
	@sync begin

		# loop over all workers
		for worker in pids

			# exclude process that launched pfold, unless only one process is available
			if worker != myid() || np == 1

				# asynchronously distribute tasks
				@async begin
					while true

						# grab next fold
						current_fold = nextidx()

						# if current fold exceeds total number of folds then exit loop
						current_fold > numfolds && break

						# report distribution of fold to worker and device
						quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

						# launch job on worker
						# worker loads data from file paths and then computes the errors in one fold
#                        sendto([worker], criterion=criterion, max_iter=max_iter, max_step=max_step)
						results[current_fold] = remotecall_fetch(worker) do
                            one_fold(x, y, path, folds, current_fold, nrmsq=nrmsq, tol=tol, p=p, max_iter=max_iter, window=window, quiet=quiet)
						end # end remotecall_fetch()
					end # end while
				end # end @async
			end # end if
		end # end for
	end # end @sync

	# return reduction (row-wise sum) over results
	return reduce(+, results[1], results) ./ numfolds
end



"""
    cv_exlstsq(x, y, path_length, q [, compute_model=false]) -> mses [, b, bidx]

This function will perform `q`-fold crossvalidation for the best model size in the exchange algorithm for least squares regression.
Each path is asynchronously spawned using any available processor.
The function to compute each path, `one_fold()`, will return a vector of mean out-of-sample errors (MSEs).
`cv_exlstsq` reduces across the `q` folds yield averaged MSEs for each model size.

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `path_length` is an `Int` to specify the length of the regularization path to compute.
- `q` is the number of folds to compute.

Optional Arguments:

- `nrmsq` is the vector to store the squared norms of the columns of `x`. Defaults to `vec(sumabs2(x,1))`.
- `folds` is the `Int` array that indicates which data to hold out for testing. Defaults to `cv_get_folds(sdata(y),q)`.
- `n` is the number of cases. Defaults to `length(y)`.
- `p` is the number of predictors. Defaults to `size(x,2)`.
- `folds` is the partition of the data. Defaults to a random partition into `q` disjoint sets.
- `tol` is the convergence tolerance to pass to the path computations. Defaults to `1e-6`.
- `max_iter` caps the number of permissible iterations in the exchange algorithm. Defaults to `100`.
- `quiet` is a `Bool` to activate output. Defaults to `true` (no output).
   *NOTA BENE*: each processor outputs feed to the console without regard to the others,
   so setting `quiet=false` can yield very messy output!
- `logreg` is a `Bool` to indicate whether or not to perform logistic regression. Defaults to `false` (do linear regression).
- `compute_model` is a `Bool` to indicate whether or not to recompute the best model. Defaults to `false` (do not recompute).

Output:

- `mses` is a vector of averaged MSEs across all folds, with one component per model computed.

If `compute_model = true`, then for the best model size `k_star` `cv_exlstsq` will also return

- `b`, a vector of `k_star` components with the computed effect sizes
- `bidx`, an `Int` vector of `k_star` components indicating the support of `b` at `k_star`.
"""
function cv_exlstsq(
    x             :: DenseMatrix{Float64},
    y             :: DenseVector{Float64},
    path_length   :: Int,
    q             :: Int;
	pids          :: DenseVector{Int}     = procs(),
    nrmsq         :: DenseVector{Float64} = vec(sumabs2(x,1)),
    folds         :: DenseVector{Int}     = cv_get_folds(sdata(y),q),
    tol           :: Float64 = 1e-6,
    n             :: Int     = length(y),
    p             :: Int     = size(x,2),
    max_iter      :: Int     = 100,
    window        :: Int     = 20,
    compute_model :: Bool    = false,
    quiet         :: Bool    = true
)

    0 <= path_length <= p || throw(ArgumentError("Path length must be positive and cannot exceed number of predictors"))

#    # preallocate vectors used in xval
#    mses = zeros(Float64, path_length)   # vector to save mean squared errors
#
#    # want to compute a path for each fold
#    # the folds are computed asynchronously
#    # the @sync macro ensures that we wait for all of them to finish before proceeding
#    @sync @inbounds for i = 1:q
#
#        # one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression)
#        mses[i] = @fetch(one_fold(x, y, path_length, folds, i, max_iter=max_iter, quiet=quiet, window=window, p=p, nrmsq=nrmsq, tol=tol))
#    end
#
#    # average mses
#    mses ./= q
    mses = pfold(x, y, path_length, folds, q, n=n, p=p, pids=pids, nrmsq=nrmsq, tol=tol, max_iter=max_iter, quiet=quiet, refit=compute_model)

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
        bp = zeros(Float64,p)
        perm = collect(1:p)

        # first use exchange algorithm to extract model
        exchange_leastsq!(bp, x, y, perm, k, max_iter=max_iter, quiet=quiet, n=n, p=p, tol=tol, nrmsq=nrmsq, window=k)

        # which components of beta are nonzero?
        # cannot use binary indices here since we need to return Int indices
        bidx = find( function f(x) x.!= zero(Float64); end, bp)

        # allocate the submatrix of x corresponding to the inferred model
        x_inferred = x[:,bidx]

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
        # return it with the vector of MSEs
        Xty = BLAS.gemv('T', one(Float64), x_inferred, y)
        XtX = BLAS.gemm('T', 'N', one(Float64), x_inferred, x_inferred)
        try
            b = XtX \ Xty
        catch e
            warn("caught error: ", e, "\nSetting returned values of b to Inf")
            b = zeros(Float64, length(bidx))
            fill!(b, Inf)
        end
        return mses, b, bidx
    end

    return mses
end
