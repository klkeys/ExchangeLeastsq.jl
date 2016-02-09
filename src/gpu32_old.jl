# alias for the OpenCL module name
cl = OpenCL

# EXCHANGE ALGORITHM FOR L0-PENALIZED LEAST SQUARES REGRESSION OVER ENTIRE GWAS 
# 
# This function minimizes the residual sum of squares
# 
# RSS = 0.5*|| Y - Xbeta ||_2^2
#
# subject to beta having no more than r nonzero components. The function will compute a B for a given value of r.
# For optimal accuracy and performance, this function should be run for multiple values of r over a path.
# In doing so, one should reuse the arguments beta, perm, and inner.
#
# This function is designed to operate on compressed genotypes in PLINK BED format. It requires the
# PLINK module to handle the decompression and linear algebra routines. Due to the compression,
# it is IMPERATIVE that the user supply the correct number of SNPs p. Also, this function will *NOT*
# work with missing data; if genotype calls are missing, then the user *MUST* impute them before using this algorithm. 
#
# Arguments:
# -- bvec is the p-dimensional warm-start for the iterate.
# -- X a BEDFile object that stores the compressed n x p statistical design matrix as an array of Int8 numbers.
# -- Y is the n-dimensional response vector.
# -- perm is a p-dimensional array of integers that sort beta in descending order by magnitude.
# -- r is the desired number of nonzero components in beta.
# -- p is the number of predictors in the model.
# 
# Optional Arguments:
# -- nrmsq is the vector to store the squared norms of the columns of X. Defaults to PLINK.sumsq(x) 
# -- n is the number of cases in the Model. Defaults to length(Y). 
# -- max_iter is the maximum permissible number of iterations. Defaults to 100.
# -- tol is the convergence tolerance. Defaults to 1e-6.
# -- "inner" is Dict for storing inner products. We fill inner dynamically as needed instead of computing X'X.
#    Defaults to an empty dict with typeasserts Int64 for the keys and Array{Float32,1} for the values.
# -- quiet is a boolean to control output. Defaults to false (full output).
# -- res is the temporary array to store the vector of RESiduals. Defaults to zeros(n).
# -- df is the temporary array to store the gradient. Defaults to zeros(p)
# -- tempn is a temporary array of length n. Defaults to zeros(n).
# -- tempn2 is another temporary array of length n. Defaults to copy(tempn).
# -- dotprods is the temporary array to store the current column of dot products from Dict "inner". Defaults to zeros(p)
# -- window is an Int variable to dictate the dimension of the search window for potentially exchanging predictors. 
#    Defaults to r (potentially exchange all current predictors). Decreasing this quantity tells the algorithm to search through 
#    fewer current active predictors, which can decrease compute time but can also degrade model recovery performance. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function exchange_leastsq!(
	bvec        :: DenseVector{Float32}, 
	X           :: BEDFile, 
	Y           :: DenseVector{Float32}, 
	perm        :: DenseVector{Int}, 
	r           :: Int,
	kernfile    :: ASCIIString; 
	inner       :: Dict{Int,DenseVector{Float32}} = Dict{Int,DenseVector{Float32}}(), 
	pids        :: DenseVector{Int}     = procs(),
	means       :: DenseVector{Float32} = mean(Float32,x, shared=true, pids=pids), 
	invstds     :: DenseVector{Float32} = invstd(x, means, shared=true, pids=pids),
#	nrmsq       :: DenseVector{Float32} = sumsq(x, shared=false, means=means, invstds=invstds, pids=pids), 
	n           :: Int = length(Y), 
	p           :: Int = size(X,2), 
	df          :: DenseVector{Float32} = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), 
	dotprods    :: DenseVector{Float32} = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), 
	tempp       :: DenseVector{Float32} = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), 
	Xb          :: DenseVector{Float32} = SharedArray(Float32, n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), 
	res         :: DenseVector{Float32} = SharedArray(Float32, n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), 
	tempn       :: DenseVector{Float32} = SharedArray(Float32, n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), 
	tempn2      :: DenseVector{Float32} = SharedArray(Float32, n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), 
	mask_n      :: DenseVector{Int}     = ones(Int,n),
	indices     :: BitArray{1}          = falses(p), 
	window      :: Int                  = r, 
	max_iter    :: Int                  = 100, 
	tol         :: Float32              = 1f-4, 
	quiet       :: Bool                 = false,
	wg_size     :: Int                  = 512,
	y_chunks    :: Int                  = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int                  = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0), 
	device      :: cl.Device            = last(cl.devices(:gpu)),
	ctx         :: cl.Context           = cl.Context(device), 
	queue       :: cl.CmdQueue          = cl.CmdQueue(ctx),
	x_buff      :: cl.Buffer            = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
	y_buff      :: cl.Buffer            = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(res)),
	m_buff      :: cl.Buffer            = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(means)),
	p_buff      :: cl.Buffer            = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
	df_buff     :: cl.Buffer            = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf = sdata(df)),
	red_buff    :: cl.Buffer            = cl.Buffer(Float32, ctx, (:rw), p * y_chunks),
	xty_buff    :: cl.Buffer            = cl.Buffer(Float32, ctx, (:rw), p),
	mask_buff   :: cl.Buffer            = cl.Buffer(Int,     ctx, (:r,  :copy), hostbuf = sdata(mask_n)),
	genofloat   :: cl.LocalMem          = cl.LocalMem(Float32, wg_size),
	program     :: cl.Program           = cl.Program(ctx, source=kernfile) |> cl.build!,
	xtyk        :: cl.Kernel            = cl.Kernel(program, "compute_xt_times_vector"),
	rxtyk       :: cl.Kernel            = cl.Kernel(program, "reduce_xt_vec_chunks"),
	wg_size32   :: Int32                = convert(Int32, wg_size),
	n32         :: Int32                = convert(Int32, n),
	p32         :: Int32                = convert(Int32, p),
	y_chunks32  :: Int32                = convert(Int32, y_chunks),
	y_blocks32  :: Int32                = convert(Int32, y_blocks),
	blocksize32 :: Int32                = convert(Int32, X.blocksize)
)

	# error checking
	n == length(tempn)    || throw(DimensionMismatch("length(Y) != length(tempn)"))
	n == length(tempn2)   || throw(DimensionMismatch("length(Y) != length(tempn2)"))
	n == length(res)      || throw(DimensionMismatch("length(Y) != length(res)"))
	p == length(bvec)     || throw(DimensionMismatch("Number of predictors != length(bvec)"))
	p == length(df)       || throw(DimensionMismatch("length(bvec) != length(df)"))
	p == length(tempp)    || throw(DimensionMismatch("length(bvec) != length(tempp)"))
	p == length(dotprods) || throw(DimensionMismatch("length(bvec) != length(dotprods)"))
#	p == length(nrmsq)    || throw(DimensionMismatch("length(bvec) != length(nrmsq)"))
	p == length(perm)     || throw(DimensionMismatch("length(bvec) != length(perm)"))
	0 <= r <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(bvec)"))
	tol >= eps(Float32)   || throw(ArgumentError("Global tolerance must exceed machine precision"))
	max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
	0 <= window <= r      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))

	# declare all integers 
	i    = 0	# used for iterations
	iter = 0	# used for outermost loop
	j    = 0	# used for iterations
	k    = 0	# used for indexing
	l    = 0	# used for indexing
	m    = 0	# used for indexing 
	idx  = 0	# used for indexing

	# declare all floats
	a       = zero(Float32)
	b       = zero(Float32)
	adb     = zero(Float32)	# = a / b
	c       = zero(Float32)
	d       = zero(Float32)
	betal   = zero(Float32)	# store lth component of bvec 
	rss     = oftype(zero(Float32),Inf)			# residual sum of squares || Y - XB ||^2
	old_rss = oftype(zero(Float32),Inf)			# previous residual sum of squares 

	# obtain top r components of bvec in magnitude
	selectperm!(perm, sdata(bvec), k, by=abs, rev=true, initialized=true)
	update_indices!(indices, bvec, p=p)

	# update X*b
	xb!(Xb,X,bvec,indices,r, means=means, invstds=invstds, pids=pids)

	# update residuals based on Xb 
	difference!(res, Y, Xb, n=n)

	# save value of RSS before starting algorithm
	rss = sumabs2(res)

	# compute inner products of X and residuals 
	# this is basically the negative gradient
#	xty!(df, X, res, means=means, invstds=invstds)
	xty!(df, df_buff, X, x_buff, res, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, wg_size, y_chunks, n, p, X.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, genofloat)

	# outer loop controls number of total iterations for algorithm run on one r
	for iter = 1:(max_iter)

		# output algorithm progress to console
		quiet || println("\titer = ", iter, ", RSS = ", rss)

		# middle loop tests each of top r parameters (by magnitude?)
		for i = abs(r-window+1):r

			# save information for current value of i
			l     = perm[i]
			betal = bvec[l]
			decompress_genotypes!(tempn, X, l, means, invstds) # tempn now holds X[:,l]

			# if necessary, compute inner products of current predictor against all other predictors
			# store this information in Dict inner
			# for current index, hold dot products in memory for duration of inner loop
			# the if/else statement below is the same as but faster than
			# > dotprods = get!(inner, l, BLAS.gemv('T', 1.0, X, tempn))
			if !haskey(inner, l)
				inner[l] = xty(X, tempn, kernfile, mask_n, means=means, invstds=invstds,
	n=X.n, 
	p=X.p, 
	p2=X.p2, 
	wg_size=wg_size,
	y_chunks=y_chunks,
    y_blocks=y_blocks,
	wg_size32=wg_size32,
	n32=n32,
	p32=p32,
	y_chunks32=y_chunks32,
	y_blocks32=y_blocks32,
	blocksize32=blocksize32,
	device=device,
	ctx=ctx,
	queue=queue,
	program=program,
	xtyk=xtyk,
	rxtyk=rxtyk,
	x_buff=x_buff,
	y_buff=y_buff,
	m_buff=m_buff,
	p_buff=p_buff,
	df_buff=df_buff,
	red_buff=red_buff,
	xty_buff=xty_buff,
	genofloat=genofloat,
	pids=pids,
	mask_buff=mask_buff)
			end
			copy!(dotprods,inner[l])

			# save values to determine best estimate for current predictor
#			b   = nrmsq[l]
			b   = n	# for normalized columns sumabs2() = n 
			a   = df[l] + betal*b
			adb = a / b
			k   = i

			# inner loop compares current predictor j against all remaining predictors j+1,...,p
			for j = (r+1):p
				idx = perm[j]
				c   = df[idx] + betal*dotprods[idx]
#				d   = nrmsq[idx]
				d   = nrmsq[idx]	# for normalized columns sumabs2() = n 

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
			decompress_genotypes!(tempn2, X, m, means, invstds) # tempn now holds X[:,l]
			axpymbz!(res, betal, tempn, adb, tempn2, p=n)

			# if necessary, compute inner product of current predictor against all other predictors
			# save in our Dict for future reference
			if !haskey(inner, m)
				inner[m] = xty(X, tempn2, kernfile, mask_n, means=means, invstds=invstds,
	n=X.n, 
	p=X.p, 
	p2=X.p2, 
	wg_size=wg_size,
	y_chunks=y_chunks,
    y_blocks=y_blocks,
	wg_size32=wg_size32,
	n32=n32,
	p32=p32,
	y_chunks32=y_chunks32,
	y_blocks32=y_blocks32,
	blocksize32=blocksize32,
	device=device,
	ctx=ctx,
	queue=queue,
	program=program,
	xtyk=xtyk,
	rxtyk=rxtyk,
	x_buff=x_buff,
	y_buff=y_buff,
	m_buff=m_buff,
	p_buff=p_buff,
	df_buff=df_buff,
	red_buff=red_buff,
	xty_buff=xty_buff,
	genofloat=genofloat,
	pids=pids,
	mask_buff=mask_buff)
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
		rss = sumabs2(res)

		# test for numerical instability
		isnan(rss) && throw(error("Objective function is NaN!"))
		isinf(rss) && throw(error("Objective function is Inf!"))

		# test for descent failure 
		# if no descent failure, then test for convergence
		# if not converged, then save RSS and continue
		ascent    = rss > old_rss + tol
		converged = abs(old_rss - rss) / abs(old_rss + 1) < tol 

		ascent && throw(error("Descent error detected at iteration $(iter)!\nOld RSS: $(old_rss)\nRSS: $(rss)")) 
		(converged || ascent) && return bvec
		old_rss = rss

	end # end outer iteration loop

	# at this point, maximum iterations reached
	# warn and return bvec
	warn("Maximum iterations $(max_iter) reached! Return value may not be correct.\n")
	return bvec

end # end exchange_leastsq


# COMPUTE ONE FOLD IN A CROSSVALIDATION SCHEME FOR A REGULARIZATION PATH FOR WHOLE GWAS
#
# For a regularization path given by the vector "path", 
# this function computes an out-of-sample error based on the indices given in the vector "test_idx". 
# The vector test_idx indicates the portion of the data to use for testing.
# The remaining data are used for training the model.
#
# This variant operates on compressed PLINK genotype matrices for GWAS analysis.
#
# Arguments:
# -- x is the nxp design matrix.
# -- y is the n-vector of responses.
# -- path is an Int to determine how many elements of the path to compute. 
# -- test_idx is the Int array that indicates which data to hold out for testing.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1f-4.
# -- max_iter caps the number of permissible iterations in the algorithm. Defaults to 100.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
function one_fold(
	x           :: BEDFile, 
	y           :: DenseVector{Float32}, 
	path_length :: Int, 
	folds       :: DenseVector{Int}, 
	fold        :: Int; 
	pids        :: DenseVector{Int}     = procs(),
	means       :: DenseVector{Float32} = mean(Float32, x, shared=true, pids=pids), 
	invstds     :: DenseVector{Float32} = invstd(x, y=means, shared=true, pids=pids), 
#	nrmsq       :: DenseVector{Float32} = sumsq(x, shared=true, means=means, invstds=invstds, pids=pids), 
	tol         :: Float32              = 1f-4,
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
	myerrors = zeros(Float32, sum(test_idx))

	# train_idx is the vector that indexes the TRAINING set
	train_idx = !test_idx

	# how big is training sample?
	train_size = length(train_idx)
	test_size  = length(test_idx)

	# GPU code requires Int variant of training indices, so do explicit conversion
	train_idx = convert(Vector{Int}, train_idx)
	test_idx  = convert(Vector{Int}, test_idx)

	# declare sparse matrix for output
	betas   = spzeros(Float32, p, path_length)

	# declare all temporary arrays
	b        = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)
	perm     = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids)
	inner    = Dict{Int,DenseVector{Float32}}()
	df       = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)		# (negative) gradient 
	tempp    = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)		# temporary array of length p
	dotprods = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)	# hold in memory the dot products for current index
	bout     = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)	# output array for beta
	tempn    = SharedArray(Float32, n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)	   	# temporary array of n floats 
	tempn2   = SharedArray(Float32, n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)	   	# temporary array of n floats 
	res      = SharedArray(Float32, n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)		# for || Y - XB ||_2^2
	bnonzeroes = falses(p)	        # indicate nonzero components of beta

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
	m_buff      = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(means))
	p_buff      = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(invstds))
	y_buff      = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(res))
	df_buff     = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf = sdata(df))
	red_buff    = cl.Buffer(Float32, ctx, (:rw),        p * y_chunks)
	mask_buff   = cl.Buffer(Int,     ctx, (:rw, :copy), hostbuf = sdata(train_idx))
	genofloat   = cl.LocalMem(Float32, wg_size)


	# loop over each element of path
	@inbounds for i = 1:path_length

		# compute the regularization path on the training set
#		exchange_leastsq!(b, x_train, y_train, perm, i, inner=inner, max_iter=max_iter, quiet=quiet, n=n, p=p, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window = min(window, i), device=device, wg_size=wg_size, tol=tol, mask_n=train_idx, y_chunks=y_chunks, y_blocks=y_blocks, r_chunks=r_chunks, device=device, ctx=ctx, queue=queue, x_buff=x_buff, y_buff=y_buff, m_buff=m_buff, p_buff=p_buff, df_buff=df_buff, red_buff=red_buff, genofloat=genofloat, program=program, xtyk=xtyk, rxtyk=rxtyk, reset_x=reset_x, wg_size32=wg_size32, n32=n32, p32=p32, y_chunks32=y_chunks32, y_blocks32=y_blocks32, blocksize32=blocksize32, r_length32=r_length32, mask_n=train_idx, mask_buff=mask_buff, pids=pids) 
		exchange_leastsq!(b, x_train, y_train, perm, i, inner=inner, max_iter=max_iter, quiet=quiet, n=n, p=p, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window = min(window, i), device=device, wg_size=wg_size, tol=tol, mask_n=train_idx, y_chunks=y_chunks, y_blocks=y_blocks, r_chunks=r_chunks, device=device, ctx=ctx, queue=queue, x_buff=x_buff, y_buff=y_buff, m_buff=m_buff, p_buff=p_buff, df_buff=df_buff, red_buff=red_buff, genofloat=genofloat, program=program, xtyk=xtyk, rxtyk=rxtyk, reset_x=reset_x, wg_size32=wg_size32, n32=n32, p32=p32, y_chunks32=y_chunks32, y_blocks32=y_blocks32, blocksize32=blocksize32, r_length32=r_length32, mask_n=train_idx, mask_buff=mask_buff, pids=pids) 

		# indices stores Boolean indexes of nonzeroes in b
		update_indices!(indices, b, p=p)

		# recompute estimated response 
		xb!(tempn,x,b,indices,i,test_idx, means=means, invstds=invstds, pids=pids)

		# recompute residuals
		difference!(r,y,Xb)

		# mask data from training set
#		res[folds .!= fold] = zero(Float32) 
		mask!(res, test_idx, 0, zero(Float32), n=n) 

		# compute out-of-sample error as squared residual averaged over size of test set
		myerrors[i] = sumabs2(r) / test_size
	end

	return myerrors
end



function pfold(
	xfile      :: ASCIIString, 
	xtfile     :: ASCIIString, 
	x2file     :: ASCIIString, 
	yfile      :: ASCIIString, 
	meanfile   :: ASCIIString, 
	invstdfile :: ASCIIString, 
	norm2file  :: ASCIIString,
	pathlength :: DenseVector{Int}, 
	kernfile   :: ASCIIString, 
	folds      :: DenseVector{Int},
	numfolds   :: Int;
	devindices :: DenseVector{Int} = ones(Int,numfolds), 
	pids       :: DenseVector{Int} = procs(),
	tol        :: Float32 = 1f-4,
	max_iter   :: Int     = 100, 
	max_step   :: Int     = 50, 
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
								x       = BEDFile(Float32, xfile, xtfile, x2file, pids=pids, header=header)
								n       = x.n
								p       = size(x,2)
								y       = SharedArray(abspath(yfile),      Float32, (n,), pids=pids)
								means   = SharedArray(abspath(meanfile),   Float32, (p,), pids=pids)
								invstds = SharedArray(abspath(invstdfile), Float32, (p,), pids=pids)
#								nrmsq   = SharedArray(abspath(norm2file),  Float32, (p,), pids=pids)

#								one_fold(x, y, pathlength, kernfile, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, devidx=devidx, pids=pids, nrmsq=nrmsq, n=n, p=p, header=header, window=window, tol=tol, wg_size=wg_size)
								one_fold(x, y, pathlength, kernfile, folds, current_fold, max_iter=max_iter, max_step=max_step, quiet=quiet, means=means, invstds=invstds, devidx=devidx, pids=pids, n=n, p=p, header=header, window=window, tol=tol, wg_size=wg_size)
						end # end remotecall_fetch()
					end # end while
				end # end @async
			end # end if
		end # end for
	end # end @sync

	# return reduction (row-wise sum) over results
	return reduce(+, results[1], results) 
end



# PARALLEL CROSSVALIDATION ROUTINE FOR EXCHANGE ALGORITHM USING PLINK FILES 
#
# This function will perform n-fold cross validation for the ideal model size in the exchange algorithm for least squares regression.
# It computes several paths as specified in the "paths" argument using the design matrix x and the response vector y.
# Each path is asynchronously spawned using any available processor.
# For each path, one fold is held out of the analysis for testing, while the rest of the data are used for training.
# The function to compute each path, "one_fold()", will return a vector of out-of-sample errors (MSEs).
# After all paths are computed, this function queries the RemoteRefs corresponding to these returned vectors.
# It then "reduces" all components along each path to yield averaged MSEs for each model size.
#
# This variant operates on compressed PLINK matrices for GWAS analysis.
#
# Arguments:
# -- x is the BEDFile object with the compressed PLINK matrices 
# -- y is the n-vector of responses.
# -- path is an Int to specify the length of the regularization path to compute 
# -- nfolds is the number of folds to compute.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- p is the number of predictors. Defaults to size(x,2).
# -- folds is the partition of the data. Defaults to a random partition into "nfolds" disjoint sets.
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1f-4.
# -- max_iter caps the number of permissible iterations in the IHT algorithm. Defaults to 100.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
#    NOTA BENE: each processor outputs feed to the console without regard to the others,
#    so setting quiet=true can yield very messy output!
# -- logreg is a Boolean to indicate whether or not to perform logistic regression. Defaults to false (do linear regression).
# -- compute_model is a Boolean to indicate whether or not to recompute the best model. Defaults to false (do not recompute). 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
function cv_exlstsq(
#	x             :: BEDFile,
#	y             :: DenseVector{Float32}, 
#	path_length   :: Int, 
#	numfolds      :: Int; 
#	nrmsq         :: DenseVector{Float32} = sumsq(x, shared=false, means=means, invstds=invstds), 
#	means         :: DenseVector{Float32} = mean(Float32, x),
#	invstds       :: DenseVector{Float32} = invstd(x, y=means),
#	folds         :: DenseVector{Int}     = cv_get_folds(y,numfolds), 
#	tol           :: Float32 = 1f-4, 
#	n             :: Int     = length(y),
#	p             :: Int     = size(x,2), 
#	max_iter      :: Int     = 100, 
#	window        :: Int     = 20,
#	compute_model :: Bool    = false,
#	quiet         :: Bool    = true
	xfile         :: ASCIIString,
	xtfile        :: ASCIIString,
	x2file        :: ASCIIString,
	yfile         :: ASCIIString,
	meanfile      :: ASCIIString,
	invstdfile    :: ASCIIString,
	norm2file     :: ASCIIString,
	path          :: DenseVector{Int}, 
	kernfile      :: ASCIIString,
	folds         :: DenseVector{Int},
	numfolds      :: Int; 
	pids          :: DenseVector{Int} = procs(),
	tol           :: Float32          = 1f-4, 
	max_iter      :: Int              = 100, 
	max_step      :: Int              = 50, 
	wg_size       :: Int              = 512,
	quiet         :: Bool             = true, 
	compute_model :: Bool             = false,
	header        :: Bool             = false
) 
	0 <= path_length <= p || throw(ArgumentError("Path length must be positive and cannot exceed number of predictors"))

	# how many elements are in the path?
	num_models = length(path)

	# how many GPU devices are available to us?
	devs = cl.devices(:gpu)
	ndev = length(devs)

	# how many folds can we fit on a GPU at once?
	# count one less per GPU device, just in case
#	max_folds = zeros(Int, ndev)
#	for i = 1:ndev
#		max_folds[i] = max(compute_max_gpu_load(x, wg_size, devs[i], prec64 = true) - 1, 0) 
#	end

	# how many rounds of folds do we need to schedule?
#	fold_rounds = zeros(Int, ndev)
#	for i = 1:ndev
#		fold_rounds[i] = div(numfolds, max_folds[i]) + (numfolds % max_folds[i] != 0 ? 1 : 0)
#	end

	# assign index of a GPU device for each fold	
	# default is first GPU device (devidx = 1)
	devindices = ones(Int, numfolds)
#	for i = 1:numfolds
#		devindices[i] += i % ndev
#	end

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# only use the worker processes
	mses = pfold(xfile, xtfile, x2file, yfile, meanfile, invstdfile, path, kernfile, folds, numfolds, max_iter=max_iter, max_step=max_step, quiet=quiet, devindices=devindices, pids=pids, header=header)

	# average mses
	mses ./= numfolds

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
	if compute_model
		
		# initialize beta vector
		bp   = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)
		perm = SharedArray(Float32, p, init = S -> S[localindexes(S)] = localindexes(S), pids=pids)
		x_inferred = zeros(Float32, n, k)

		# first use exchange algorithm to extract model
		exchange_leastsq!(bp, x, y, perm, k, max_iter=max_iter, quiet=quiet, p=p, means=means, invstds=invstds)

		# which components of beta are nonzero?
		# cannot use binary indices here since we need to return Int indices
		inferred_model = perm[1:k]

		# allocate the submatrix of x corresponding to the inferred model
		decompress_genotypes!(x_inferred, x, inferred_model, means=means, invstds=invstds, pids=pids)

		# now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
		# return it with the vector of MSEs
		Xty = BLAS.gemv('T', one(Float32), x_inferred, y)
		XtX = BLAS.gemm('T', 'N', one(Float32), x_inferred, x_inferred)
		b   = XtX \ Xty
		return mses, b, inferred_model
	end

	return mses
end
