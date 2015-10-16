########################
### TESTING ROUTINES ###
########################

# TEST THE EXCHANGE ALGORITHM ON RANDOM DATA
#
# This function simulates random uncorrelated Gaussian data to test the exchange algorithm.
#
# Arguments:
# -- n is the desired number of samples.
# -- p is the desired number of predictors.
# -- r is the desired number of true nonzero components, or the size of the true model.
# -- extra is the amount of extra model sizes to compute beyond r. Thus, the length of the computed
#    regularization path is r + extra.
#
# Optional Arguments:
# -- max_iter caps the number of outer iterations in the exchange algorithm. Defaults to 100.
# -- quiet is a Bool to control algorithm output to the console. Defaults to false (print output).
# -- window is the number of predictors to consider for swapping. Defaults to r + extra, which tells the algorithm 
#    to consider all active predictors for swapping. 
# -- seed sets the random seed for the data generation. Defaults to 2015.
# -- noise indicates the magnitude of the random noise added to the response y. Defaults to 0.5, which indicates
#    noise drawn from the N(0,0.25) distribution
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function test_exleastsq(n::Int, p::Int, r::Int, extra::Int; max_iter::Int = 100, quiet::Bool = false, window::Int = r+extra, seed::Int = 2015, noise::Float64 = 0.5) 
	tic()

	# testing data
	x         = randn(n,p)
	y         = randn(n)
	b         = zeros(p)
	perm      = collect(1:p)
	inner     = Dict{Int,DenseArray{Float64,1}}()

	# declare all temporary arrays
	res        = zeros(n)	# Y - Xbeta
	df         = zeros(p)	# X'(Y - Xbeta)
	tempn      = zeros(n)	# temporary array of length n 
	tempn2     = zeros(n)	# temporary array of length n 
	tempp      = zeros(p)	# temporary array of length p
	dotprods   = zeros(p)	# hold in memory the dot products for current index

	# precompute sum of squares for each column of x
	const nrmsq = vec(sumabs2(x,1))

	# declare return array
	betas = zeros(p,r+extra)

	# use a random permutation of indices for true model
	true_model = sample(perm, r)
	b[true_model] = randn(r)
	b_true = copy(b)
	BLAS.gemv!('N', 1.0, x, b, 1.0, y)	# y = xb + noise
	fill!(b, 0.0)
	iter = 0

	# print time spent on parameter initialization
	quiet || println("Parameter initialization took ", toq(), " seconds.")

	# reset timer and compute path
	tic()
	@time begin
		for k = 1:(r+extra)
			b = exchange_leastsq!(b, x, y, perm, k, inner=inner, max_iter=max_iter, quiet=quiet, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window=min(k,window))
			rss = sumabs2(res)

			# print results from this step of regularization path
			if !quiet
				println("#pred = ", k, ", rss = ", rss, ", #truepos = ", countnz(b[true_model])) 
				print("\t")
				print("\n")
			end
			betas[:,k] = b
			update_col!(betas, b, k, n=p, p=r+extra, a=1.0) 
		end
	end	

	# print time spent on computation of regularization path
	quiet || println("Path from 1 to ", r+extra, " took ", toq(), " seconds to compute.")

	# return sparsified betas 
	return b_true, sparse(betas) 
end




# TEST THE EXCHANGE ALGORITHM FOR LEAST SQUARES REGRESSION
# This subroutine runs the IHT on a dataset of chromosome 1 data from the WTCCC.
function test_exchangeleastsq(x_path::ASCIIString = "/Users/kkeys/Downloads/wtccc-n2000p32307.txt", y_path::ASCIIString = "/Users/kkeys/Downloads/withnoiselevelsd0_1/Y.100.1", b_path::ASCIIString = "/Users/kkeys/Downloads/withnoiselevelsd0_1/causal.100.1", r::Int = 100, tol::Float64 = 1e-6, max_iter::Int = 100, quiet::Bool = false, extra::Int = 100, window::Int = r) 

    # repeat input options
    println("This function will test the exchange algorithm.")
    println("Given options:")
    println("\tPath to X  = ", x_path)
    println("\tPath to Y  = ", y_path)
    println("\tPath to B  = ", b_path)
    println("\tModel size = ", r)
    println("\ttolerance  = ", tol, "\n\tmax_iter   = ", max_iter)

    # precompile @time macro for later use
    println("\nCompiling @time macro...")
    @time 1+1; 

    # now load data
    println("\nLoading data...")
    tic()

    # load design matrix
    x = readdlm(x_path)
    const (n,p) = size(x)

    # load response vector
    y = readdlm(y_path)
    y = vec(y)  # need y to be 1D

    # load model
    B = readdlm(b_path)

    # need indices for true model
    # add 1 since Julia uses 1-indexing
    bidx = convert(Array{Int,1}, B[:,2]) + 1

    # need components of true model
    be = convert(Array{Float64,1}, B[:,3])

    # how long did it take to load files?
	file_time = toq()
    quiet || println("Files took ", file_time, " seconds to load.")

	# load parameters
	println("Allocating temporary arrays...")
	tic()

	# declare all temporary arrays
    b          = zeros(p)	# parameter vector
	res        = zeros(n)	# Y - Xbeta
	df         = zeros(p)	# X'(Y - Xbeta)
	tempn      = zeros(n)	# temporary array of length n 
	tempn2     = zeros(n)	# temporary array of length n 
	tempp      = zeros(p)	# temporary array of length p
	dotprods   = zeros(p)	# hold in memory the dot products for current index
	perm       = collect(1:p) # hold permutation vector of b

	# declare associative array for storing inner products
	inner = Dict{Int,DenseArray{Float64,1}}()

	# how long did the allocation take?
	alloc_time = toq()
	quiet || println("Temporary arrays allocated in ", alloc_time, " seconds.")

	# precompute sum of squares for each column of x
	quiet || println("Precomputing squared Euclidean norms of matrix columns...")
	tic()
	const nrmsq = vec(sumabs2(x,1))
	norm_time = toq()
	quiet || println("Column norms took ", norm_time, " seconds to compute")

	# declare any return values
	iter = 0

	# print time spent on parameter initialization
	parameter_time = norm_time + alloc_time
	quiet || println("Parameter initialization took ", parameter_time, " seconds.")

    # run exchange 
    println("\nRunning exchange algorithm...")

	# reset timer and compute path
	tic()
	@time begin
		for i = 1:(r+extra)
			b = exchange_leastsq!(b, x, y, perm, i, inner=inner, max_iter=max_iter, quiet=quiet, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window=min(i,window))
			rss = sumabs2(res)

			# print results from this step of regularization path
			quiet || println("#pred = ", i, ", rss = ", rss, ", #truepos = ", countnz(b[bidx])) 
		end
	end	

	# print time spent on computation of regularization path
	quiet || println("Path from 1 to ", r+extra, " took ", toq(), " seconds to compute.")


    # recover vector from output
    bk = b[bidx]

    # evaluate model
    println("\nTrue positives: ", countnz(bk), "/", length(be), ".")
    println("Distances to true model:")
    println("Chebyshev (L-Inf) = ", chebyshev(bk, be), ".")
    println("Euclidean (L-2)   = ", euclidean(bk, be), ".")
end







# TEST THE EXCHANGE ALGORITHM FOR LEAST SQUARES REGRESSION OVER PLINK BINARY DATA
# This subroutine runs the exchange algorithm on a GWAS dataset from the WTCCC.
function test_exchangeleastsq_plink(x_path::ASCIIString = "/Users/kkeys/Downloads/wtccc_full/wtccc-n2k_chr1_clean.bed", xt_path::ASCIIString = "/Users/kkeys/Downloads/wtccc_full/wtccc-n2k_chr1_clean_t.bed", y_path::ASCIIString = "/Users/kkeys/Downloads/withnoiselevelsd0_1/Y.100.1", b_path::ASCIIString = "/Users/kkeys/Downloads/withnoiselevelsd0_1/causal.100.1", r::Int = 10, tol::Float64 = 1e-4, max_iter::Int = 100, quiet::Bool = false, extra::Int = 10, window::Int = 10) 

    # repeat input options
	quiet || begin
		println("This function will test the IHT algorithm.")
		println("Given options:")
		println("\tPath to X  = ", x_path)
		println("\tPath to X' = ", xt_path)
		println("\tPath to Y  = ", y_path)
		println("\tPath to B  = ", b_path)
		println("\tModel size = ", r)
		println("\ttolerance  = ", tol, "\n\tmax_iter   = ", max_iter)
	end

    # precompile @time macro for later use
    quiet || println("\nCompiling @time macro...")
    @time 1+1; 

    # now load data
    quiet || println("\nLoading data...")
    tic()

    # load design matrix
	x = BEDFile(x_path, xt_path)
	const (n,p) = size(x)

    # load response vector
    y = readdlm(y_path)
	y = vec(y) # need y to be 1D
	const y = convert(SharedArray{Float64,1}, y)

	const means   = PLINK.mean(Float64, x, shared=true) 
	const invstds = PLINK.invstd(x, means, shared=true)

    # load model
    B = readdlm(b_path)

    # need indices for true model
    # add 1 since Julia uses 1-indexing
	const bidx = convert(Array{Int,1}, B[:,2]) + 1

    # need components of true model
	const be = convert(Array{Float64,1}, B[:,3])

    # how long did it take to load files?
	file_time = toq()
    quiet || println("\nFiles took ", file_time, " seconds to load.")

	# discard B to recover some memory
	quiet || println("Object B is no longer needed, recovering memory...")
	tic()
	B = false
	gc()
	B_time = toq()
	quiet || println("Discarding B took ", B_time, " seconds.")

	# load parameters
	quiet || println("\nAllocating temporary arrays...")
	tic()

	# declare all temporary arrays
	b        = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# parameter vector
	Xb       = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# Xbeta
	res      = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# Y - Xbeta
	df       = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# X'(Y - Xbeta)
	tempn    = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# temporary array of length n 
	tempn2   = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# temporary array of length n 
	tempp    = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# temporary array of length p
	dotprods = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# hold in memory the dot products for current index
	indices  = falses(p)

	# permutation vector of b has a more complicated initialization
	perm = SharedArray(Int, p, init = S -> S[localindexes(S)] = localindexes(S))

	# declare associative array for storing inner products
	inner = Dict{Int,SharedArray{Float64,1}}()

	# how long did the allocation take?
	alloc_time = toq()
	quiet || println("Temporary arrays allocated in ", alloc_time, " seconds.")

	# precompute sum of squares for each column of x
	quiet || println("\nPrecomputing squared Euclidean norms of matrix columns...")
	tic()
	const nrmsq = sumsq(Float64, x, shared=true, means=means, invstds=invstds)
	norm_time = toq()
	quiet || println("Column norms took ", norm_time, " seconds to compute")

	# print time spent on parameter initialization
	parameter_time = norm_time + alloc_time + file_time + B_time
	quiet || println("Parameter initialization took ", parameter_time, " seconds.")

    # run exchange 
    quiet || println("\nRunning exchange algorithm...")

	# reset timer and compute path
	tic()
	@time begin
		for i = 2:(r+extra)
			exchange_leastsq!(b, x, y, perm, i, p=p, n=n, inner=inner, max_iter=max_iter, quiet=quiet, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window=min(i,window), Xb=Xb, means=means, invstds=invstds, indices=indices)
			rss = sumabs2(res)

			# print results from this step of regularization path
			quiet || println("#pred = ", i, ", rss = ", rss, ", #truepos = ", countnz(b[bidx])) 
		end
	end	

	# print time spent on computation of regularization path
	quiet || println("Path from 1 to ", r+extra, " took ", toq(), " seconds to compute.")

    # recover vector from output
    bk = b[bidx]

    # evaluate model
	quiet || begin
		println("\nTrue positives: ", countnz(bk), "/", length(be), ".")
		println("Distances to true model:")
		println("Chebyshev (L-Inf) = ", chebyshev(bk, be), ".")
		println("Euclidean (L-2)   = ", euclidean(bk, be), ".")
	end

	return bk, bidx
end

test_exchangeleastsq_plink(r::Int, extra::Int, window::Int) = test_exchangeleastsq_plink("/Users/kkeys/Downloads/wtccc_full/wtccc-n2k_chr1_clean.bed", "/Users/kkeys/Downloads/wtccc_full/wtccc-n2k_chr1_clean_t.bed", "/Users/kkeys/Downloads/withnoiselevelsd0_1/Y.100.1", "/Users/kkeys/Downloads/withnoiselevelsd0_1/causal.100.1", r, 1e-4, 100, false, extra, window) 


# CROSSVALIDATION ROUTINE FOR EXCHANGE ALGORITHM 
function test_cv_exlstsq(kend,numfolds; tol::Float64 = 1e-4, max_iter::Int = 1000, quiet::Bool=true, compute_model::Bool = false)

	# notify of number of folds
	quiet || println("Testing ", numfolds, "-fold crossvalidation from k = 1 to k = ", kend, "...")

	# start timing
	tic()

	# file paths for data, response, model
	x_path = "/Users/kkeys/Downloads/wtccc-n2000p32307.txt"
	y_path = "/Users/kkeys/Downloads/withnoiselevelsd0_1/Y.100.1"
	b_path = "/Users/kkeys/Downloads/withnoiselevelsd0_1/causal.100.1"

	# load design matrix
	x = readdlm(x_path)
	const (n,p) = size(x)
	const x = convert(SharedArray, x)

	# load response vector
	y = readdlm(y_path)
	y = vec(y)	# need y to be 1D
	const y = convert(SharedArray,y)

	# load model
	B = readdlm(b_path)

	# need indices for true model
	# add 1 since Julia uses 1-indexing
	const bidx = convert(Array{Int,1}, B[:,2]) + 1

	# need components of true model
	const be = convert(Array{Float64,1}, B[:,3])
	
	parameter_time = toq()

	# inform console that the data are loaded
	quiet || println("Data loaded in ", parameter_time, " seconds.")
	
	# obtain a partition of the sample set
	myfolds = cv_get_folds(y,numfolds)

	# inform console that the folds are computed
	quiet || println("Folds computed.")

	# run cv_iht to crossvalidate dataset
	# be careful! mses will be a tuple if compute_model=true
	tic()
	if compute_model
		mses, b = cv_exlstsq(x,y,kend,numfolds, p=p, tol=tol, max_iter=max_iter, quiet=quiet, folds=myfolds, compute_model=compute_model) 
	else
		mses = cv_exlstsq(x,y,kend,numfolds, p=p, tol=tol, max_iter=max_iter, quiet=quiet, folds=myfolds, compute_model=compute_model) 
	end


	cv_time = toq()
	quiet || println("CV computed in ", cv_time, " seconds.") 

	compute_model && return mses, b
	return mses
end


function compare_exlstsq(x::Matrix, kend, numfolds; tol::Float64 = 1e-4, max_iter::Int = 1000, quiet::Bool=true, compute_model::Bool = false, y_path::ASCIIString = "/Users/kkeys/Desktop/Thesis/Exchange/Y_1_same_1.0_2.0.txt", b_path::ASCIIString = "/Users/kkeys/Desktop/Thesis/Exchange/b_1_same_1.0_2.0.txt", bidx_path::ASCIIString = "/Users/kkeys/Desktop/Thesis/Exchange/bidx_1_same_1.0_2.0.txt")

	# notify of number of folds
	quiet || println("Testing ", numfolds, "-fold crossvalidation from k = 1 to k = ", kend, "...")

	# start timing
	tic()

	# dimensions of design matrix
	const n,p = size(x)
	x = convert(SharedArray{Float64,2}, x)

	# load response vector
	y = readdlm(y_path)
	y = vec(y)	# need y to be 1D
#	y = convert(Array{Float64,1}, y)
	y = convert(SharedArray{Float64,1}, y)

	# load model
	be = readdlm(b_path)
	be = vec(be)
	const be = convert(Array{Float64,1}, be)

	# need indices for true model
	bidx = readdlm(bidx_path)
	bidx = vec(bidx)
	const bidx = convert(Array{Int,1}, bidx) 

	parameter_time = toq()

	# inform console that the data are loaded
	quiet || println("Data loaded in ", parameter_time, " seconds.")
	
	# start with exchange algorithm
	quiet || println("Running crossvalidation with exchange algorithm...")

	# obtain a partition of the sample set
	myfolds = cv_get_folds(y,numfolds)

	# inform console that the folds are computed
	quiet || println("Folds computed.")

	# run cv_iht to crossvalidate dataset
	# be careful! mses will be a tuple if compute_model=true
	tic()
#	mses = cv_exlstsq(x,y,kend,numfolds, p=p, tol=tol, max_iter=max_iter, quiet=quiet, folds=myfolds, compute_model=compute_model) 
	if compute_model
		mses, b, ex_model = cv_exlstsq(x,y,kend,numfolds, p=p, tol=tol, max_iter=max_iter, quiet=quiet, folds=myfolds, compute_model=compute_model) 
	else
		mses = cv_exlstsq(x,y,kend,numfolds, p=p, tol=tol, max_iter=max_iter, quiet=quiet, folds=myfolds, compute_model=compute_model) 
	end


	cv_time = toq()
	quiet || println("CV computed in ", cv_time, " seconds.") 

	# now compute LASSO model
	quiet || println("\nComputing LASSO model...")

	# translate folds

	# run glmnet
	tic()
	cv = glmnetcv(sdata(x), sdata(y), dfmax=kend, nlambda=kend, nfolds=5, parallel=true, folds=my_folds, tol=tol)

	# extract best beta from lasso
	lasso_btemp = cv.path.betas[:,indmin(cv.meanloss)]

	# inform console that lasso computation is complete
	lasso_time = toq()
	quiet || println("LASSO model crossvalidated in ", lasso_time, " seconds.")

	if compute_model
#		lasso_bidx = lasso_btemp .!= 0.0
		lasso_bidx = find( function f(x) x .!= 0.0; end, lasso_btemp) 
		x_lasso = x[:,lasso_bidx]
		lasso_b = (x_lasso'*x_lasso) \ (x_lasso'*y)
	end


	# return values based on compute_model 
	compute_model && return mses, cv.meanloss, b, lasso_b, be, ex_model, lasso_bidx, bidx
	return mses, cv.meanloss
end

function compare_exlstsq(x::Matrix, y::Vector, be::Vector, bidx::Vector, kend::Int, numfolds::Int; tol::Float64 = 1e-4, max_iter::Int = 10000, quiet::Bool=true, compute_model::Bool = false, n::Int = length(y), p::Int = length(be), window::Int = 20, nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1))) 
	
	isequal((n,p),size(x)) || throw(DimensionMismatch("(n,p) = ($n,$p) != ($(size(x)))"))
	0 <= window <= p       || throw(error("For current X, must have window in (0,$p)"))
	0 <= kend <= p         || throw(error("Path legnth must be positive and cannot exceed the number of predictors"))

	# notify of number of folds
	quiet || println("Testing ", numfolds, "-fold crossvalidation from k = 1 to k = ", kend, "...")

	# start with exchange algorithm
	quiet || println("Running crossvalidation with exchange algorithm...")

	# obtain a partition of the sample set
	myfolds = cv_get_folds(y,numfolds)

	# inform console that the folds are computed
	quiet || println("Folds computed.")

	# run cv_iht to crossvalidate dataset
	tic()
	if compute_model
		mses, b, ex_model = cv_exlstsq(x,y,kend,numfolds, p=p, tol=tol, max_iter=max_iter, quiet=quiet, folds=myfolds, compute_model=compute_model, window=window) 
	else
		mses = cv_exlstsq(x,y,kend,numfolds, p=p, tol=tol, max_iter=max_iter, quiet=quiet, folds=myfolds, compute_model=compute_model, window=window) 
	end
	cv_time = toq()
	quiet || println("CV computed in ", cv_time, " seconds.") 

	# now compute LASSO model
	quiet || println("\nComputing LASSO model...")

	# run glmnet
	tic()
#	cv = glmnetcv(x,y, dfmax=kend, nlambda=kend, nfolds=numfolds, parallel=true, folds=my_folds, tol=tol)

	# extract best beta from lasso
#	lasso_btemp = cv.path.betas[:,indmin(cv.meanloss)]
	lasso_btemp = zeros(p)
	lasso_btemp[1] = 1.0

	# inform console that lasso computation is complete
	lasso_time = toq()
	quiet || println("LASSO model crossvalidated in ", lasso_time, " seconds.")

	if compute_model
#		lasso_bidx = lasso_btemp .!= 0.0
		lasso_bidx = find( function f(x) x .!= 0.0; end, lasso_btemp) 
		x_lasso = x[:,lasso_bidx]
		lasso_b = (x_lasso'*x_lasso) \ (x_lasso'*y)
	end


	# return values based on compute_model 
	compute_model && return mses, zeros(kend), b, lasso_b, be, ex_model, lasso_bidx, bidx, lasso_time, cv_time
	return mses, cv.meanloss, cv_time, lasso_time
end

