"""
    one_fold(x,y,path_length,folds,fold) -> Vector{Float}

For a regularization path given by `models`,
this function computes an out-of-sample error for the fold given by `fold` in a `q`-fold crossvalidation scheme.
The folds are indexed by the `Int` vector `folds`.

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.
- `models` is an `Int` vector containing the regularization path to compute.
- `folds` is the `Int` array that indicates which data to hold out for testing.
- `fold` indexes the current fold.

Optional Arguments:

- `tol` is the convergence tolerance to pass to the path computations. Defaults to `1e-6`.
- `max_iter` caps the number of permissible iterations in the algorithm. Defaults to `100`.
- `window` is an `Int` to dictate the dimension of the search window for potentially exchanging predictors.
   Defaults to `max(20, min(maximum(models), size(x,2)))` or the greater of `20` and `r`, where `r` is the lesser of the maximum model size and the number of predictors.
   Decreasing this quantity tells the algorithm to search through fewer current active predictors,
   which can decrease compute time but can also degrade model recovery performance.
- `quiet` is a Boolean to activate output. Defaults to `true` (no output).
"""
function one_fold{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    models   :: DenseVector{Int},
    folds    :: DenseVector{Int},
    fold     :: Int;
    tol      :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    window   :: Int  = max(20, min(maximum(models), size(x,2))),
    quiet    :: Bool = true
)

    # find testing indices and size of test set
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # allocate the arrays for the training set
    x_train = x[train_idx,:]
    y_train = y[train_idx]

    # will return a sparse matrix of betas
    betas = exlstsq(x_train, y_train, models=models, window=window, max_iter=max_iter, tol=tol, quiet=quiet)

    # compute the mean out-of-sample error for the TEST set
    errors = vec(sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ (2*test_size)

    return errors :: Vector{T}
end

"""
    pfold(x, y, path, folds, nfolds) -> Vector

This function is the parallel execution kernel in `cv_exlstsq`. It is not meant to be called outside of `cv_exlstsq`.
It will distribute `nfolds` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold` for each fold.
Each fold will compute a regularization path given by the `Int` vector `models`.
`pfold` collects the vectors of MSEs from each process, sum-reduces them, and returns their average across all folds.
"""
function pfold{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
    models   :: DenseVector{Int},
    folds    :: DenseVector{Int},
    nfolds   :: Int;
    pids     :: DenseVector{Int} = procs(x),
    tol      :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    window   :: Int  = max(20, min(maximum(models), size(x,2))),
    quiet    :: Bool = true,
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
    results = cell(nfolds)

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
                        current_fold > nfolds && break

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
                        results[current_fold] = remotecall_fetch(worker) do
                            one_fold(x, y, models, folds, current_fold, tol=tol, max_iter=max_iter, window=window, quiet=quiet)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (reduce(+, results[1], results) ./ nfolds) :: Vector{T}
end



"""
    cv_exlstsq(x, y, models, q) -> ELSQResults

This function will perform `q`-fold crossvalidation for the best model size in the exchange algorithm for least squares regression.
Each path is asynchronously spawned using any available indexed processor.
The function to compute each path, `one_fold`, will return a vector of mean out-of-sample errors (MSEs).
`cv_exlstsq` reduces across the `q` folds yield averaged MSEs for each model size.

Arguments:

- `x` is the `n` x `p` design matrix.
- `y` is the `n`-vector of responses.

Optional Arguments:

- `models` is an integer vector to specify the the regularization path to compute.
- `q` is the number of folds to compute. Defaults to `max(3, min(CPU_CORES, 5))`.
- `folds` is the `Int` array that indicates which data to hold out for testing. Defaults to `cv_get_folds(sdata(y),q)`.
- `folds` is the partition of the data. Defaults to a random partition into `q` disjoint sets.
- `tol` is the convergence tolerance to pass to the path computations. Defaults to `1e-6`.
- `max_iter` caps the number of permissible iterations in the exchange algorithm. Defaults to `100`.
- `quiet` is a `Bool` to activate output. Defaults to `true` (no output).
   *NOTA BENE*: each processor outputs feed to the console without regard to the others,
   so setting `quiet=false` can yield very messy output!

Output:

An `ELSQResults` container object with the following fields:
- `mses` is a vector of averaged MSEs across all folds, with one component per model computed.
- `k` is the best crossvalidated model size. Here model size refers to the number of nonzero regression coefficients.
- `b`, a vector of `k` components with the computed effect sizes of the best model β.
- `bidx`, an `Int` vector of `k` components indicating the support of `b` at `k`.
"""
function cv_exlstsq{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T};
    models   :: DenseVector{Int} = collect(1:min(20,size(x,2))),
    q        :: Int              = max(3, min(CPU_CORES, 5)),
    pids     :: DenseVector{Int} = procs(),
    folds    :: DenseVector{Int} = cv_get_folds(sdata(y),q),
    tol      :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    window   :: Int  = max(20, min(maximum(models), size(x,2))),
    quiet    :: Bool = true,
)

#    1 <= minimum(models) <= maximum(models) <= size(x,2) || throw(ArgumentError("Model sizes must be positive and cannot exceed number of predictors"))

    mses = pfold(x, y, models, folds, q, pids=pids, tol=tol, max_iter=max_iter, quiet=quiet, window=window)

    # what is the best model size?
    k = convert(Int, floor(mean(models[mses .== minimum(mses)])))

    # print results
    quiet || print_cv_results(mses, models, k)

    # refit coefficients
    b, bidx = refit_exlstsq(x, y, k, models=models, tol=tol, max_iter=max_iter, window=window, quiet=quiet)

    return ELSQCrossvalidationResults(mses, b, bidx, k, sdata(models))
end
