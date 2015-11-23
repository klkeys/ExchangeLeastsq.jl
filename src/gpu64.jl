"An alias for the `OpenCL` module name"
cl = OpenCL


"""
    exchange_leastsq!(b, x::BEDFile, y, perm, r, kernfile)

If supplied a `BEDFile` `x` and an OpenCL kernel file `kernfile` as an ASCIIString, then `exchange_leastsq!()` will attempt to accelerate the calculation of the dense gradient `x' * (y - x*b)` with a GPU device. This variant introduces a host of extra arguments for the GPU. Most of these arguments are only meant to facilitate the calculation of a regularization path. The optional arguments that a user will most likely wish to manipulate are:

- `device`, an `OpenCL.Device` object indicating the device to use in computations. Defaults to `last(OpenCL.devices(:gpu))`.
- `mask_n`, an `Int` vector of `0`s and `1`s indexing the rows of `x` and `y` that should be included or masked in the analysis. Defaults to `ones(Int,n)`, which includes all data.
- `wg_size` is the desired workgroup size for the GPU. Defaults to `512`.
"""
function exchange_leastsq!(
    b        :: DenseVector{Float64}, 
    x           :: BEDFile, 
    y           :: DenseVector{Float64}, 
    perm        :: DenseVector{Int}, 
    r           :: Int,
    kernfile    :: ASCIIString; 
    inner       :: Dict{Int,DenseVector{Float64}} = Dict{Int,DenseVector{Float64}}(), 
    pids        :: DenseVector{Int}     = procs(),
    means       :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
    invstds     :: DenseVector{Float64} = invstd(x, means, shared=true, pids=pids),
    n           :: Int                  = length(y), 
    p           :: Int                  = size(x,2), 
    df          :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), 
    dotprods    :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), 
    tempp       :: DenseVector{Float64} = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), 
    Xb          :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), 
    res         :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), 
    tempn       :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), 
    tempn2      :: DenseVector{Float64} = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), 
    mask_n      :: DenseVector{Int}     = ones(Int,n),
    indices     :: BitArray{1}          = falses(p), 
    window      :: Int                  = r, 
    max_iter    :: Int                  = 100, 
    tol         :: Float64              = 1e-4, 
    n64         :: Float64              = convert(Float64, n), 
    quiet       :: Bool                 = false,
    wg_size     :: Int                  = 512,
    y_chunks    :: Int                  = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int                  = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0), 
    r_chunks    :: Int                  = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0), 
    device      :: cl.Device            = last(cl.devices(:gpu)),
    ctx         :: cl.Context           = cl.Context(device), 
    queue       :: cl.CmdQueue          = cl.CmdQueue(ctx),
    x_buff      :: cl.Buffer            = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(x.x)),
    y_buff      :: cl.Buffer            = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(res)),
    m_buff      :: cl.Buffer            = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(means)),
    p_buff      :: cl.Buffer            = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
    df_buff     :: cl.Buffer            = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf = sdata(df)),
    red_buff    :: cl.Buffer            = cl.Buffer(Float64, ctx, (:rw), p * y_chunks),
    xty_buff    :: cl.Buffer            = cl.Buffer(Float64, ctx, (:rw), p),
    mask_buff   :: cl.Buffer            = cl.Buffer(Int,     ctx, (:r,  :copy), hostbuf = sdata(mask_n)),
    genofloat   :: cl.LocalMem          = cl.LocalMem(Float64, wg_size),
    program     :: cl.Program           = cl.Program(ctx, source=kernfile) |> cl.build!,
    xtyk        :: cl.Kernel            = cl.Kernel(program, "compute_xt_times_vector"),
    rxtyk       :: cl.Kernel            = cl.Kernel(program, "reduce_xt_vec_chunks"),
    reset_x     :: cl.Kernel            = cl.Kernel(program, "reset_x"),
    wg_size32   :: Int32                = convert(Int32, wg_size),
    n32         :: Int32                = convert(Int32, n),
    p32         :: Int32                = convert(Int32, p),
    y_chunks32  :: Int32                = convert(Int32, y_chunks),
    y_blocks32  :: Int32                = convert(Int32, y_blocks),
    blocksize32 :: Int32                = convert(Int32, x.blocksize),
    r_length32  :: Int32                = convert(Int32, p*y_chunks)
)

    # error checking
    n == length(tempn)    || throw(DimensionMismatch("length(y) != length(tempn)"))
    n == length(tempn2)   || throw(DimensionMismatch("length(y) != length(tempn2)"))
    n == length(res)      || throw(DimensionMismatch("length(y) != length(res)"))
    p == length(b)     || throw(DimensionMismatch("Number of predictors != length(b)"))
    p == length(df)       || throw(DimensionMismatch("length(b) != length(df)"))
    p == length(tempp)    || throw(DimensionMismatch("length(b) != length(tempp)"))
    p == length(dotprods) || throw(DimensionMismatch("length(b) != length(dotprods)"))
    p == length(perm)     || throw(DimensionMismatch("length(b) != length(perm)"))
    0 <= r <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(b)"))
    tol >= eps(Float64)   || throw(ArgumentError("Global tolerance must exceed machine precision"))
    max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
    0 <= window <= r      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))

    # declare algorithm variables
    i       = 0                          # used for iterations
    iter    = 0                          # used for outermost loop
    j       = 0                          # used for iterations
    k       = 0                          # used for indexing
    l       = 0                          # used for indexing
    m       = 0                          # used for indexing
    idx     = 0                          # used for indexing
    a       = zero(Float64)
    adb     = zero(Float64)              # = a / b
    c       = zero(Float64)
    betal   = zero(Float64)              # store lth component of b 
    rss     = zero(Float64)              # residual sum of squares || Y - XB ||^2
    old_rss = oftype(zero(Float64), Inf) # previous residual sum of squares 

    # obtain top r components of b in magnitude
    selectperm!(perm, sdata(b), r, by=abs, rev=true, initialized=true)
    update_indices!(indices, b, p=p)

    # update X*b
    xb!(Xb,x,b,indices,r,mask_n, means=means, invstds=invstds, pids=pids)

    # update residuals based on Xb 
    difference!(res, y, Xb, n=n)
    mask!(res, mask_n, 0, zero(Float64), n=n)

    # save value of RSS before starting algorithm
    rss = 0.5*sumabs2(res)

    # compute inner products of x and residuals 
    # this is basically the negative gradient
    xty!(df, df_buff, x, x_buff, res, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)

    # outer loop controls number of total iterations for algorithm run on one r
    for iter = 1:(max_iter)

        # output algorithm progress to console
        quiet || println("\titer = ", iter, ", RSS = ", rss)

        # middle loop tests each of top r parameters (by magnitude?)
        for i = abs(r-window+1):r

            # save information for current value of i
            l     = perm[i]
            betal = b[l]
            decompress_genotypes!(tempn, x, l, means, invstds) # tempn now holds x[:,l]
            mask!(tempn, mask_n, 0, zero(Float64), n=n)

            # if necessary, compute inner products of current predictor against all other predictors
            # store this information in Dict inner
            # for current index, hold dot products in memory for duration of inner loop
            # the if/else statement below is the same as but faster than
            # > dotprods = get!(inner, l, BLAS.gemv('T', 1.0, x, tempn))
            if !haskey(inner, l)
                inner[l] = xty(x, tempn, kernfile, mask_n, pids=pids, means=means, invstds=invstds, n=x.n, p=x.p, p2=x.p2, wg_size=wg_size, y_chunks=y_chunks, y_blocks=y_blocks, r_chunks=r_chunks, wg_size32=wg_size32, n32=n32, p32=p32, y_chunks32=y_chunks32, y_blocks32=y_blocks32, blocksize32=blocksize32, device=device, ctx=ctx, queue=queue, program=program, xtyk=xtyk, rxtyk=rxtyk, x_buff=x_buff, y_buff=y_buff, m_buff=m_buff, p_buff=p_buff, red_buff=red_buff, xty_buff=xty_buff, genofloat=genofloat, mask_buff=mask_buff, reset_x=reset_x, r_length32=r_length32)
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
            mask!(tempn2, mask_n, 0, zero(Float64), n=n)
            axpymbz!(res, betal, tempn, adb, tempn2, p=n)
            mask!(res, mask_n, 0, zero(Float64), n=n)

            # if necessary, compute inner product of current predictor against all other predictors
            # save in our Dict for future reference
            if !haskey(inner, m)
                inner[m] = xty(x, tempn2, kernfile, mask_n, pids=pids, means=means, invstds=invstds, n=x.n, p=x.p, p2=x.p2, wg_size=wg_size, y_chunks=y_chunks, y_blocks=y_blocks, r_chunks=r_chunks, wg_size32=wg_size32, n32=n32, p32=p32, y_chunks32=y_chunks32, y_blocks32=y_blocks32, blocksize32=blocksize32, device=device, ctx=ctx, queue=queue, program=program, xtyk=xtyk, rxtyk=rxtyk, x_buff=x_buff, y_buff=y_buff, m_buff=m_buff, p_buff=p_buff, red_buff=red_buff, xty_buff=xty_buff, genofloat=genofloat, mask_buff=mask_buff, reset_x=reset_x, r_length32=r_length32)
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
                b[j] = zero(Float64)
            end

        end # end middle loop over predictors 

        # need to mask residuals that are not in analysis
        mask!(res, mask_n, 0, zero(Float64), n=n)

        # update residual sum of squares
        rss = 0.5*sumabs2(res)

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
    one_fold(x::BEDFile, y, path_length, kernfile, folds, fold)

If supplied a `BEDFile` `x` and an OpenCL kernel file `kernfile` as an ASCIIString, then `one_fold` will attempt to accelerate the calculation of the dense gradient `x' * (y - x*b)` in `exchange_leastsq!()` with a GPU device. The new optional arguments include:

- `devidx`, an index indicating the GPU device to use in computations. The device is chosen  as `OpenCL.devices(:gpu)[devidx]`. Defaults to `1` (choose the first GPU device)
- `wg_size` is the desired workgroup size for the GPU. Defaults to `512`.
- `header` is a `Bool` to feed to `readdlm` when loading the nongenetic covariates `x.x2`. Defaults to `false` (no header). 
"""
function one_fold(
    x           :: BEDFile, 
    y           :: DenseVector{Float64}, 
    path_length :: Int, 
    kernfile    :: ASCIIString,
    folds       :: DenseVector{Int}, 
    fold        :: Int; 
    pids        :: DenseVector{Int}     = procs(),
    means       :: DenseVector{Float64} = mean(Float64, x, shared=true, pids=pids), 
    invstds     :: DenseVector{Float64} = invstd(x, y=means, shared=true, pids=pids), 
    tol         :: Float64              = 1e-6,
    max_iter    :: Int                  = 100, 
    window      :: Int                  = 20, 
    n           :: Int                  = length(y),
    p           :: Int                  = size(x,2),
    wg_size     :: Int                  = 512,
    devidx      :: Int                  = 1,
    header      :: Bool                 = false,
    quiet       :: Bool                 = true 
)
    # get list of available GPU devices
    # var device gets pointer to device indexed by variable devidx 
    device = cl.devices(:gpu)[devidx]

    # find testing indices
    test_idx = folds .== fold

    # preallocate vector for output
    myerrors = zeros(Float64, path_length) 

    # train_idx is the vector that indexes the TRAINING set
    train_idx = !test_idx

    # how big is training sample?
    train_size = length(train_idx)
    test_size  = length(test_idx)

    # GPU code requires Int variant of training indices, so do explicit conversion
    train_idx = convert(Vector{Int}, train_idx)
    test_idx  = convert(Vector{Int}, test_idx)

    # declare sparse matrix for output
    betas   = spzeros(Float64, p, path_length)

    # declare all temporary arrays
    b        = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)
    perm     = SharedArray(Int,     p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids)
    df       = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)     # (negative) gradient 
    tempp    = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)     # temporary array of length p
    dotprods = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)     # hold in memory the dot products for current index
    bout     = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)     # output array for beta
    tempn    = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)     # temporary array of n floats 
    tempn2   = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)     # temporary array of n floats 
    res      = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)     # for || Y - XB ||_2^2
    Xb       = SharedArray(Float64, n, init = S -> S[localindexes(S)] = zero(Float64),   pids=pids)
    indices  = falses(p)
    inner    = Dict{Int,DenseVector{Float64}}()
    n64      = convert(Float64, n)

    # allocate GPU variables
    y_chunks    = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0)
    y_blocks    = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0)
    r_chunks    = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0)
    ctx         = cl.Context(device)
    queue       = cl.CmdQueue(ctx)
    program     = cl.Program(ctx, source=kernfile) |> cl.build!
    xtyk        = cl.Kernel(program, "compute_xt_times_vector")
    rxtyk       = cl.Kernel(program, "reduce_xt_vec_chunks")
    reset_x     = cl.Kernel(program, "reset_x")
    wg_size32   = convert(Int32, wg_size)
    n32         = convert(Int32, n)
    p32         = convert(Int32, p)
    y_chunks32  = convert(Int32, y_chunks)
    y_blocks32  = convert(Int32, y_blocks)
    blocksize32 = convert(Int32, x.blocksize)
    r_length32  = convert(Int32, p*y_chunks)
    x_buff      = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(x.x))
    m_buff      = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(means))
    p_buff      = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(invstds))
    y_buff      = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(res))
    df_buff     = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf = sdata(df))
    red_buff    = cl.Buffer(Float64, ctx, (:rw),        p * y_chunks)
    mask_buff   = cl.Buffer(Int,     ctx, (:rw, :copy), hostbuf = sdata(train_idx))
    xty_buff    = cl.Buffer(Float64, ctx, (:rw), p)
    genofloat   = cl.LocalMem(Float64, wg_size)


    # loop over each element of path
    @inbounds for i = 1:path_length
        # compute the regularization path on the training set
        exchange_leastsq!(b, x, y, perm, i, kernfile, inner=inner, max_iter=max_iter, quiet=quiet, n=n, p=p, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window = min(window, i), device=device, wg_size=wg_size, tol=tol, mask_n=train_idx, y_chunks=y_chunks, y_blocks=y_blocks, r_chunks=r_chunks, device=device, ctx=ctx, queue=queue, x_buff=x_buff, y_buff=y_buff, m_buff=m_buff, p_buff=p_buff, df_buff=df_buff, red_buff=red_buff, genofloat=genofloat, program=program, xtyk=xtyk, rxtyk=rxtyk, reset_x=reset_x, wg_size32=wg_size32, n32=n32, p32=p32, y_chunks32=y_chunks32, y_blocks32=y_blocks32, blocksize32=blocksize32, r_length32=r_length32, mask_buff=mask_buff, pids=pids, means=means, invstds=invstds, indices=indices, Xb=Xb, xty_buff=xty_buff, n64=n64) 

        # indices stores Boolean indexes of nonzeroes in b
        update_indices!(indices, b, p=p)

        # recompute estimated response 
        xb!(tempn,x,b,indices,i,test_idx, means=means, invstds=invstds, pids=pids)

        # recompute residuals
        difference!(res,y,Xb)

        # mask data from training set
        mask!(res, test_idx, 0, zero(Float64), n=n) 

        # compute out-of-sample error as squared residual averaged over size of test set
        myerrors[i] = 0.5*sumabs2(res) / test_size
    end

    return myerrors
end



"""
    pfold(xfile, xtfile, x2file,yfile, meanfile, invstdfile,pathlength,kernfile,folds,numfolds[, pids=procs(), devindices=ones(Int,numfolds])

This function is the parallel execution kernel in `cv_iht()`. It is not meant to be called outside of `cv_iht()`.
It will distribute `numfolds` crossvalidation folds across the processes supplied by the optional argument `pids` and call `one_fold()` for each fold.
Each fold will use the GPU device indexed by its corresponding component of the optional argument `devindices` to compute a regularization path of length `pathlength`.
`pfold()` collects the vectors of MSEs returned by calling `one_fold()` for each process, reduces them, and returns their average across all folds.
"""
function pfold(
    xfile      :: ASCIIString, 
    xtfile     :: ASCIIString, 
    x2file     :: ASCIIString, 
    yfile      :: ASCIIString, 
    meanfile   :: ASCIIString, 
    invstdfile :: ASCIIString, 
    pathlength :: Int, 
    kernfile   :: ASCIIString, 
    folds      :: DenseVector{Int},
    numfolds   :: Int;
    devindices :: DenseVector{Int} = ones(Int,numfolds), 
    pids       :: DenseVector{Int} = procs(),
    tol        :: Float64 = 1e-4,
    max_iter   :: Int     = 100, 
    window     :: Int     = 20,
    wg_size    :: Int     = 512,
    quiet      :: Bool    = true, 
    header     :: Bool    = false
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

                        # grab index of GPU device
                        devidx = devindices[current_fold]

                        # report distribution of fold to worker and device
                        quiet || print_with_color(:blue, "Computing fold $current_fold on worker $worker and device $devidx.\n\n")
                        
                        # launch job on worker 
                        # worker loads data from file paths and then computes the errors in one fold
                        results[current_fold] = remotecall_fetch(worker) do 
                                pids    = [worker]
                                x       = BEDFile(Float64, xfile, xtfile, x2file, pids=pids, header=header)
                                n       = x.n
                                p       = size(x,2)
                                y       = SharedArray(abspath(yfile), Float64, (n,), pids=pids)
                                means   = SharedArray(abspath(meanfile), Float64, (p,), pids=pids)
                                invstds = SharedArray(abspath(invstdfile), Float64, (p,), pids=pids)

                                one_fold(x, y, pathlength, kernfile, folds, current_fold, max_iter=max_iter, quiet=quiet, means=means, invstds=invstds, devidx=devidx, pids=pids, n=n, p=p, header=header, window=window, tol=tol, wg_size=wg_size)
                        end # end remotecall_fetch()
                    end # end while
                end # end @async
            end # end if
        end # end for
    end # end @sync

    # return reduction (row-wise sum) over results
    return reduce(+, results[1], results) 
end



"""
    cv_iht(xfile,xtfile,x2file,yfile,meanfile,invstdfile,path,kernfile,folds,numfolds [, pids=procs(), wg_size=512])

This variant of `cv_iht()` performs `numfolds` crossvalidation with a `BEDFile` object loaded by `xfile`, `xtfile`, and `x2file`,
with column means stored in `meanfile` and column precisions stored in `invstdfile`.
The continuous response is stored in `yfile` with data particioned by the `Int` vector `folds`.
The calculations employ GPU acceleration by calling OpenCL kernels from `kernfile` with workgroup size `wg_size`.
The folds are distributed across the processes given by `pids`.
"""
function cv_exlstsq(
#   x             :: BEDFile,
#   y             :: DenseVector{Float64}, 
#   path_length   :: Int, 
#   numfolds      :: Int; 
#   nrmsq         :: DenseVector{Float64} = sumsq(x, shared=false, means=means, invstds=invstds), 
#   means         :: DenseVector{Float64} = mean(Float64, x),
#   invstds       :: DenseVector{Float64} = invstd(x, y=means),
#   folds         :: DenseVector{Int}     = cv_get_folds(y,numfolds), 
#   tol           :: Float64 = 1e-4, 
#   n             :: Int     = length(y),
#   p             :: Int     = size(x,2), 
#   max_iter      :: Int     = 1000, 
#   window        :: Int     = 20,
#   compute_model :: Bool    = false,
#   quiet         :: Bool    = true
    xfile         :: ASCIIString,
    xtfile        :: ASCIIString,
    x2file        :: ASCIIString,
    yfile         :: ASCIIString,
    meanfile      :: ASCIIString,
    invstdfile    :: ASCIIString,
#   norm2file     :: ASCIIString,
#   path          :: DenseVector{Int}, 
    path_length   :: Int, 
    kernfile      :: ASCIIString,
    folds         :: DenseVector{Int},
    numfolds      :: Int; 
    pids          :: DenseVector{Int} = procs(),
    tol           :: Float64          = 1e-4, 
    max_iter      :: Int              = 100, 
    wg_size       :: Int              = 512,
    quiet         :: Bool             = true, 
    compute_model :: Bool             = false,
    header        :: Bool             = false
) 

    # how many GPU devices are available to us?
    devs = cl.devices(:gpu)
    ndev = length(devs)

    # how many folds can we fit on a GPU at once?
    # count one less per GPU device, just in case
#   max_folds = zeros(Int, ndev)
#   for i = 1:ndev
#       max_folds[i] = max(compute_max_gpu_load(x, wg_size, devs[i], prec64 = true) - 1, 0) 
#   end

    # how many rounds of folds do we need to schedule?
#   fold_rounds = zeros(Int, ndev)
#   for i = 1:ndev
#       fold_rounds[i] = div(numfolds, max_folds[i]) + (numfolds % max_folds[i] != 0 ? 1 : 0)
#   end

    # assign index of a GPU device for each fold    
    # default is first GPU device (devidx = 1)
    devindices = ones(Int, numfolds)
#   for i = 1:numfolds
#       devindices[i] += i % ndev
#   end

    # want to compute a path for each fold
    # the folds are computed asynchronously
    # only use the worker processes
    mses = pfold(xfile, xtfile, x2file, yfile, meanfile, invstdfile, path_length, kernfile, folds, numfolds, max_iter=max_iter, quiet=quiet, devindices=devindices, pids=pids, header=header)

    # average mses
    mses ./= numfolds

    # what is the best model size?
    k = convert(Int, floor(mean(collect(1:path_length)[mses .== minimum(mses)])))

    # print results
    quiet || begin
        println("\n\nCrossvalidation Results:")
        println("k\tMSE")
        for i = 1:length(mses)
            println(i, "\t", mses[i])
        end
        println("\nThe lowest MSE is achieved at k = ", k) 
    end

    # recompute ideal model
    if compute_model
        
        # initialize beta vector
        bp   = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)
        perm = SharedArray(Float64, p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids)
        x_inferred = zeros(Float64, n, k) 

        # first use exchange algorithm to extract model
        exchange_leastsq!(bp, x, y, perm, k, max_iter=max_iter, quiet=quiet, p=p, means=means, invstds=invstds) 

        # which components of beta are nonzero?
        # cannot use binary indices here since we need to return Int indices
        inferred_model = perm[1:k] 

        # allocate the submatrix of x corresponding to the inferred model
        decompress_genotypes!(x_inferred, x, inferred_model, means=means, invstds=invstds, pids=pids)

        # now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
        # return it with the vector of MSEs
        Xty = BLAS.gemv('T', one(Float64), x_inferred, y)   
        XtX = BLAS.gemm('T', 'N', one(Float64), x_inferred, x_inferred)
        b   = XtX \ Xty
        return mses, b, inferred_model
    end

    return mses
end
