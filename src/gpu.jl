"""
    exchange_leastsq!(v, x::BEDFile, y, k) -> ELSQResults

    Execute the exchange algorithm using a `BEDFile` object `x` for one model size `k`.
"""
function exchange_leastsq!{T <: Float}(
    v        :: ELSQVariables{T},
    x        :: BEDFile{T},
    y        :: SharedVector{T},
    kernfile :: ASCIIString,
    k        :: Int,
    w        :: PlinkGPUVariables{T};
    pids     :: DenseVector{Int} = procs(x),
    n        :: Int  = length(y),
    p        :: Int  = size(x,2),
    window   :: Int  = k,
    max_iter :: Int  = 100,
    tol      :: T    = convert(T, 1e-6),
    quiet    :: Bool = false
)
    # initial value for previous residual sum of squares
    old_rss = oftype(tol,Inf) 

    # obtain top r components of bvec in magnitude
    selectperm!(v.perm, v.b, k, by=abs, rev=true, initialized=true)

    # update estimated response
    update_indices!(v.idx, v.b)
    A_mul_B!(v.xb, x, v.b, v.idx, k, v.mask_n)

    # update residuals
    difference!(v.r, y, v.xb)
    mask!(v.r, v.mask_n, 0, zero(T))

    # save value of RSS before starting algorithm
    rss = sumabs2(v.r) / 2

    # compute inner products of x and residuals
    # this is basically the negative gradient
    At_mul_B!(v.df, x, v.r, v.mask_n, w)

    # outer loop controls number of total iterations for algorithm run on one r
    for iter = 1:(max_iter)

        # output algorithm progress to console
        quiet || println("\titer = ", iter, ", RSS = ", rss)

        # middle loop tests each of top r parameters (by magnitude?)
        for i = abs(k-window+1):k

            # save information for current value of i
            l     = v.perm[i]
            betal = v.b[l]
            decompress_genotypes!(v.tempn, x, l) # tempn now holds X[:,l]

            # if necessary, compute inner products of current predictor against all other predictors
            # store this information in Dict inner
            # for current index, hold dot products in memory for duration of inner loop
            # the if/else statement below is the same as but faster than
            # > dotprods = get!(inner, l, BLAS.gemv('T', one(T), X, tempn))
            if !haskey(v.inner, l)
#                v.inner[l] = At_mul_B(x, v.tempn, w, mask_n=v.mask_n) 
                At_mul_B!(v.tempp, x, v.tempn, v.mask_n, w)
                v.inner[l] = copy(v.tempp) 
            end
            copy!(v.dotprods, v.inner[l])

#            # save values to determine best estimate for current predictor
#            b   = v.nrmsq[l]
#            a   = v.df[l] + betal*b
#            adb = a / b
#            r   = i
#
#            # inner loop compares current predictor j against all remaining predictors j+1,...,p
#            for j = (k+1):p
#                idx = v.perm[j]
#                c   = v.df[idx] + betal*v.dotprods[idx]
#                d   = v.nrmsq[idx]
#
#                # if current inactive predictor beats current active predictor,
#                # then save info for swapping
#                if c*c/d > a*adb + tol
#                    a   = c
#                    b   = d
#                    r   = j
#                    adb = a / b
#                end
#            end # end inner loop over remaining predictor set

            # subroutine compares current predictor i against all predictors k+1, k+2, ..., p
            # these predictors are candidates for inclusion in set
            # _exlstsq_innerloop! find best new predictor r
            a, b, r, adb = _exlstsq_innerloop!(v, k, i, p, tol)

            # now want to update residuals with current best predictor
            m = v.perm[r]
            decompress_genotypes!(v.tempn2, x, m) # tempn now holds x[:,m]
            mask!(v.tempn2, v.mask_n, 0, zero(T))
            axpymbz!(v.r, betal, v.tempn, adb, v.tempn2)
            mask!(v.r, v.mask_n, 0, zero(T))

            # if necessary, compute inner product of current predictor against all other predictors
            # save in our Dict for future reference
            # compare in performance to
            # > tempp = get!(inner, m, BLAS.gemv('T', one(T), X, tempn2))
            if !haskey(v.inner, m)
#                v.inner[m] = At_mul_B(x, v.tempn2, w, mask_n=v.mask_n) 
                At_mul_B!(v.tempp, x, v.tempn2, v.mask_n, w)
                v.inner[m] = copy(v.tempp)
            end
            copy!(v.tempp, v.inner[m])

            # also update df
            axpymbz!(v.df, betal, v.dotprods, adb, v.tempp)

#            # now swap best predictor with current predictor
#            j         = v.perm[i]
#            v.perm[i] = v.perm[r]
#            v.perm[r] = j
#            v.b[m]    = adb
#            if r != i
#                v.b[j] = zero(T)
#            end
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
    exlstsq(x::BEDFile, y, models) -> ELSQResults

    Execute the exchange algorithm with a `BEDFile` object `x` over a vector `models` of model sizes.
"""
function exlstsq{T <: Float}(
    x        :: BEDFile{T},
    y        :: SharedVector{T},
    kernfile :: ASCIIString;
    models   :: DenseVector{Int} = collect(1:min(20,size(x,2))),
    pids     :: DenseVector{Int} = procs(x), 
    v        :: ELSQVariables{T} = ELSQVariables(x, y), 
    w        :: PlinkGPUVariables{T} = PlinkGPUVariables(v.df, x, y, kernfile, v.mask_n),
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
        exchange_leastsq!(v, x, y, kernfile, i, w,  window=min(i,window), max_iter=max_iter, tol=tol, quiet=quiet, n=n, p=p, pids=pids)
        betas[:,i] = sparse(sdata(v.b))
    end

    # return matrix of betas
    return betas 
end

"""
    one_fold(x::BEDFile ,y, moels, folds, fold) -> Vector{Float}

    Compute one crossvalidation fold with the exchange algorithm using a `BEDFile` object `x`.
"""
function one_fold{T <: Float}(
    x        :: BEDFile{T},
    y        :: SharedVector{T},
    models   :: DenseVector{Int},
    kernfile :: ASCIIString,
    folds    :: DenseVector{Int},
    fold     :: Int;
    tol      :: T    = convert(T, 1e-6),
    max_iter :: Int  = 100,
    window   :: Int  = size(x,2),
    quiet    :: Bool = true
)

    # find testing indices
    # how big is training sample?
    test_idx = folds .== fold
    test_size = sum(test_idx)

    # preallocate vector for output
    myerrors = zeros(T, sum(test_idx))

    # train_idx is the vector that indexes the TRAINING set
    train_idx = convert(Vector{Int}, !test_idx)
    test_idx  = convert(Vector{Int}, test_idx)


    # preallocate temporary variables
    v = ELSQVariables(x, y, train_idx)
    w = PlinkGPUVariables(v.df, x, y, kernfile, v.mask_n)

    # will return a sparse matrix of betas
    betas = exlstsq(x, y, kernfile, models=models, v=v, w=w, window=window, max_iter=max_iter, tol=tol, quiet=quiet) 

    # compute the mean out-of-sample error for the TEST set
    mses = zeros(T, length(models))

    for i in eachindex(models) 
        # set b 
        copy!(v.b, sub(betas, :, i)) 

        # update indices of current b
        update_indices!(v.idx, v.b)

        # compute estimated response with current b
        A_mul_B!(v.xb, x, v.b, v.idx, models[i], test_idx)

        # compute residuals
        difference!(v.r, y, v.xb)
        mask!(v.r, test_idx, 0, zero(T))

        # compute out-of-sample error as squared residual averaged over size of test set
        mses[i] = sumabs2(v.r) / (2*test_size)
    end

    return mses :: Vector{T}
end

"""
    pfold(x, y, path, folds, q) -> Vector

This function is the parallel execution kernel in `cv_exlstsq`. It is not meant to be called outside of `cv_exlstsq`.
It will distribute `q` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold` for each fold.
Each fold will compute a regularization path `1:path`.
`pfold` collects the vectors of MSEs returned by calling `one_fold` for each process, reduces them, and returns their average across all folds.
"""
function pfold(
    T        :: Type,
    xfile    :: ASCIIString,
    xtfile   :: ASCIIString,
    x2file   :: ASCIIString,
    yfile    :: ASCIIString,
    meanfile :: ASCIIString,
    precfile :: ASCIIString,
    models   :: DenseVector{Int},
    kernfile :: ASCIIString,
    folds    :: DenseVector{Int},
    q        :: Int;
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-6),
    max_iter :: Int   = 100,
    window   :: Int   = 20, 
    quiet    :: Bool  = true,
    header   :: Bool  = false,
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
    results = cell(q)

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
                        current_fold > q && break

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
                        results[current_fold] = remotecall_fetch(worker) do
                            pids = [worker]
                            x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, pids=pids, header=header)
                            y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids)
                            one_fold(x, y, models, kernfile, folds, current_fold, max_iter=max_iter, window=window, tol=tol, quiet=quiet)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (reduce(+, results[1], results) ./ q) :: Vector{T}
end



"""
    cv_exlstsq(x::BEDFile, y, models, q) -> ELSQCrossvalidationResults 

Perofrm `q`-fold crossvalidation for the best model size in the exchange algorithm for a `BEDFile` object `x`, response vector `y`, and model sizes specified in `models`. 
"""
function cv_exlstsq(
    T        :: Type,
    xfile    :: ASCIIString,
    xtfile   :: ASCIIString,
    x2file   :: ASCIIString,
    yfile    :: ASCIIString,
    meanfile :: ASCIIString,
    precfile :: ASCIIString,
    kernfile :: ASCIIString;
    q        :: Int = max(3, min(CPU_CORES, 5)),
    models   :: DenseVector{Int} = begin
           # find p from the corresponding BIM file, then make path 
            bimfile = xfile[1:(endof(xfile)-3)] * "bim"
            p       = countlines(bimfile)
            collect(1:min(20,p))
            end,
    folds    :: DenseVector{Int} = begin
           # find n from the corresponding FAM file, then make folds
            famfile = xfile[1:(endof(xfile)-3)] * "fam"
            n       = countlines(famfile)
            cv_get_folds(n, q)
            end,
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-6),
    max_iter :: Int   = 100,
    window   :: Int   = 20,
    wg_size  :: Int   = 512,
    quiet    :: Bool  = true,
    header   :: Bool  = false,
)

#    1 <= minimum(models) <= maximum(models) <= size(x,2) || throw(ArgumentError("Model sizes must be positive and cannot exceed number of predictors"))

    mses = pfold(T, xfile, xtfile, x2file, yfile, meanfile, precfile, models, kernfile, folds, q, max_iter=max_iter, quiet=quiet, pids=pids, header=header, window=window)

    # what is the best model size?
    k = convert(Int, floor(mean(models[mses .== minimum(mses)])))

    # print results
    quiet || print_cv_results(mses, models, k)

    # recompute ideal model
    # initialize all variables 
    x = BEDFile(T, xfile, xtfile, x2file, meanfile, precfile, pids=pids, header=header)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}
    v = ELSQVariables(x, y, ones(Int, length(y)))
    w = PlinkGPUVariables(v.df, x, y, kernfile, v.mask_n)

    # first use exchange algorithm to extract model
    exchange_leastsq!(v, x, y, kernfile, k, w, max_iter=max_iter, quiet=quiet, tol=tol, window=k, pids=pids)

    # which components of beta are nonzero?
    inferred_model = v.b .!= zero(T)
    bidx = find(inferred_model)
    
    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model) 

    # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
    # return it with the vector of MSEs
    xty = BLAS.gemv('T', one(T), x_inferred, y)
    xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
    b = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

    bids = prednames(x)[bidx]
    return ELSQCrossvalidationResults{T}(mses, b, bidx, k, sdata(models), bids)
end

# default type for cv_exlstsq is Float64
cv_exlstsq(xfile::ASCIIString, xtfile::ASCIIString, x2file::ASCIIString, yfile::ASCIIString, meanfile::ASCIIString, precfile::ASCIIString, kernfile::ASCIIString; q::Int = max(3, min(CPU_CORES, 5)), models::DenseVector{Int} = begin bimfile = xfile[1:(endof(xfile)-3)] * "bim"; p = countlines(bimfile); collect(1:min(20,p)) end, folds::DenseVector{Int} = begin famfile = xfile[1:(endof(xfile)-3)] * "fam"; n = countlines(famfile); cv_get_folds(n, q) end, pids::DenseVector{Int} = procs(), tol::Float64 = 1e-6, max_iter::Int  = 100, window::Int  = 20, quiet::Bool = true, header::Bool = false,
) = cv_exlstsq(Float64, xfile, xtfile, x2file, yfile, meanfile, precfile, kernfile, folds=folds, q=q, models=models, pids=pids, tol=tol, max_iter=max_iter, window=window, quiet=quiet, header=header)


function pfold(
    T          :: Type,
    xfile      :: ASCIIString,
    x2file     :: ASCIIString,
    yfile      :: ASCIIString,
    models     :: DenseVector{Int},
    kernfile   :: ASCIIString,
    folds      :: DenseVector{Int},
    q          :: Int;
    pids       :: DenseVector{Int} = procs(),
    max_iter   :: Int  = 100,
    quiet      :: Bool = true,
    header     :: Bool = false
)

    # ensure correct type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # how many CPU processes can pfold use?
    np = length(pids)

    # report on CPU processes
    quiet || println("pfold: np = ", np)
    quiet || println("pids = ", pids)

    # set up function to share state (indices of folds)
    i = 1
    nextidx() = (idx=i; i+=1; idx)

    # preallocate cell array for results
    results = cell(q)

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
                        current_fold > q && break

                        # report distribution of fold to worker
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker.\n\n")

                        # launch job on worker
                        # worker loads data from file paths and then computes the errors in one fold
                        results[current_fold] = remotecall_fetch(worker) do
                                pids = [worker]
                                x = BEDFile(T, xfile, x2file, pids=pids, header=header)
                                y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids)
                                one_fold(x, y, models, kernfile, folds, current_fold, max_iter=max_iter, window=window, quiet=quiet, pids=pids)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return (reduce(+, results[1], results) ./ q) :: Vector{T}
end


function cv_exlstsq(
    T        :: Type,
    xfile    :: ASCIIString,
    x2file   :: ASCIIString,
    yfile    :: ASCIIString,
    kernfile :: ASCIIString;
    q        :: Int = max(3, min(CPU_CORES, 5)),
    models   :: DenseVector{Int} = begin
           # find p from the corresponding BIM file, then make path 
            bimfile = xfile[1:(endof(xfile)-3)] * "bim"
            p       = countlines(bimfile)
            collect(1:min(20,p))
            end,
    folds    :: DenseVector{Int} = begin
           # find n from the corresponding FAM file, then make folds
            famfile = xfile[1:(endof(xfile)-3)] * "fam"
            n       = countlines(famfile)
            cv_get_folds(n, q)
            end,
    pids     :: DenseVector{Int} = procs(),
    tol      :: Float = convert(T, 1e-4),
    max_iter :: Int   = 100,
    wg_size  :: Int   = 512,
    quiet    :: Bool  = true,
    header   :: Bool  = false
)
    # enforce type
    T <: Float || throw(ArgumentError("Argument T must be either Float32 or Float64"))

    # how many elements are in the path?
    num_models = length(models)

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # only use the worker processes
    mses = pfold(T, xfile, x2file, yfile, models, kernfile, folds, q, max_iter=max_iter, quiet=quiet, pids=pids, header=header, window=window)

    # what is the best model size?
    k = convert(Int, floor(mean(models[mses .== minimum(mses)])))

    # print results
    !quiet && print_cv_results(mses, models, k)

    # recompute ideal model
    # initialize all variables 
    x = BEDFile(T, xfile, x2file, pids=pids, header=header)
    y = SharedArray(abspath(yfile), T, (x.geno.n,), pids=pids) :: SharedVector{T}
    v = ELSQVariables(x, y, ones(Int, length(y)))
    w = PlinkGPUVariables(v.df, x, y, kernfile, v.mask_n)

    # first use exchange algorithm to extract model
    exchange_leastsq!(v, x, y, kernfile, k, w, max_iter=max_iter, quiet=quiet, tol=tol, window=k, pids=pids)

    # which components of beta are nonzero?
    inferred_model = output.beta .!= zero(T)
    bidx = find(inferred_model)

    # allocate the submatrix of x corresponding to the inferred model
    x_inferred = zeros(T, x.geno.n, sum(inferred_model))
    decompress_genotypes!(x_inferred, x, inferred_model)

    # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y
    xty = BLAS.gemv('T', one(T), x_inferred, y)
    xtx = BLAS.gemm('T', 'N', one(T), x_inferred, x_inferred)
    b   = zeros(T, length(bidx))
    try
        b = (xtx \ xty) :: Vector{T}
    catch e
        warn("in refit, caught error: ", e, "\nSetting returned values of b to -Inf")
        fill!(b, -Inf)
    end

    bids = prednames(x)[bidx]
    return ELSQCrossvalidationResults{T}(mses, sdata(models), b, bidx, k)
end

# default type for cv_iht is Float64
cv_exlstsq(xfile::ASCIIString, x2file::ASCIIString, yfile::ASCIIString, kernfile::ASCIIString; q::Int = max(3, min(CPU_CORES, 5)), models::DenseVector{Int} = begin bimfile=xfile[1:(endof(xfile)-3)] * "bim"; p=countlines(bimfile); collect(1:min(20,p)) end, folds::DenseVector{Int} = begin famfile=xfile[1:(endof(xfile)-3)] * "fam"; n=countlines(famfile); cv_get_folds(n, q) end, pids::DenseVector{Int}=procs(), tol::Float64=1e-4, max_iter::Int=100, quiet::Bool=true, header::Bool=false) = cv_iht(Float64, xfile, x2file, yfile, kernfile, models=models, folds=folds, q=q, pids=pids, tol=tol, max_iter=max_iter, quiet=quiet, header=header)
