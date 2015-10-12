# COMPUTE ONE FOLD IN A CROSSVALIDATION SCHEME FOR A REGULARIZATION PATH
#
# For a regularization path given by the vector "path", 
# this function computes an out-of-sample error based on the indices given in the vector "test_idx". 
# The vector test_idx indicates the portion of the data to use for testing.
# The remaining data are used for training the model.
#
# Arguments:
# -- x is the nxp design matrix.
# -- y is the n-vector of responses.
# -- path is an Int to determine how many elements of the path to compute. 
# -- test_idx is the Int array that indicates which data to hold out for testing.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1e-4.
# -- max_iter caps the number of permissible iterations in the algorithm. Defaults to 1000.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
function one_fold(
	x           :: DenseArray{Float64,2}, 
	y           :: DenseArray{Float64,1}, 
	path_length :: Int, 
	folds       :: DenseArray{Int,1}, 
	fold        :: Int; 
	nrmsq       :: DenseArray{Float64,1} = vec(sumsq(x,1)),
	p           :: Int  = size(x,2), 
	max_iter    :: Int  = 1000, 
	window      :: Int  = 20, 
	quiet       :: Bool = true 
) 

	# find testing indices
	test_idx = folds .== fold

	# preallocate vector for output
	myerrors = zeros(sum(test_idx))

	# train_idx is the vector that indexes the TRAINING set
	train_idx = !test_idx

	# how big is training sample?
	const n = length(train_idx)

	# allocate the arrays for the training set
	x_train   = x[train_idx,:]
	y_train   = y[train_idx] 
	b         = zeros(Float64, p)
	betas     = zeros(Float64, p,path_length)
	perm      = collect(1:p)

	# allocate Dict to store inner prodcuts
	@compat inner = Dict{Int,DenseArray{Float64,1}}()

	# declare all temporary arrays
	res        = zeros(Float64, n)	# Y - Xbeta
	df         = zeros(Float64, p)	# X'(Y - Xbeta)
	tempn      = zeros(Float64, n)	# temporary array of length n 
	tempn2     = zeros(Float64, n)	# temporary array of length n 
	tempp      = zeros(Float64, p)	# temporary array of length p
	dotprods   = zeros(Float64, p)	# hold in memory the dot products for current index
	bnonzeroes = falses(p)			# indicate nonzero components of beta
	bout       = zeros(Float64, p)	# output array for beta

	# loop over each element of path
	for i = 1:path_length

		# compute the regularization path on the training set
		bout = exchange_leastsq!(b, x_train, y_train, perm, i, inner=inner, max_iter=max_iter, quiet=quiet, n=n, p=p, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window = min(window, i), nrmsq=nrmsq) 

		# find the support of bout
		update_indices!(bnonzeroes, bout, p=p)

		# subset training indices of x with support
		x_refit    = x_train[:,bnonzeroes]

		# perform ordinary least squares to refit support of bout
		Xty        = BLAS.gemv('T', 1.0, x_refit, y_train)
		XtX        = BLAS.gemm('T', 'N', 1.0, x_refit, x_refit)
		b_refit    = XtX \ Xty 

		# put refitted values back in bout
		bout[bnonzeroes] = b_refit

		# copy bout back to b 
		copy!(b, bout)

		# store b
		update_col!(betas, b, i, n=p, p=path_length, a=1.0) 
	end

	# sparsify the betas
	betas = sparse(betas)

	# compute the mean out-of-sample error for the TEST set 
	myerrors  = vec(sumabs2(broadcast(-, y[test_idx], x[test_idx,:] * betas), 1)) ./ length(test_idx)

	return myerrors
end




# CREATE UNSTRATIFIED CROSSVALIDATION PARTITION
# This function will partition n indices into k disjoint sets for k-fold crossvalidation
# Arguments:
# -- n is the dimension of the data set to partition.
# -- k is the number of disjoint sets in the partition.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function cv_get_folds(y::Vector, nfolds::Int)
	n, r = divrem(length(y), nfolds)
	shuffle!([repmat(1:nfolds, n), 1:r])
end





# PARALLEL CROSSVALIDATION ROUTINE FOR EXCHANGE ALGORITHM 
#
# This function will perform n-fold cross validation for the ideal model size in the exchange algorithm for least squares regression.
# It computes several paths as specified in the "paths" argument using the design matrix x and the response vector y.
# Each path is asynchronously spawned using any available processor.
# For each path, one fold is held out of the analysis for testing, while the rest of the data are used for training.
# The function to compute each path, "one_fold()", will return a vector of out-of-sample errors (MSEs).
# After all paths are computed, this function queries the RemoteRefs corresponding to these returned vectors.
# It then "reduces" all components along each path to yield averaged MSEs for each model size.
#
# Arguments:
# -- x is the nxp design matrix.
# -- y is the n-vector of responses.
# -- path is an Int to specify the length of the regularization path to compute 
# -- nfolds is the number of folds to compute.
#
# Optional Arguments:
# -- n is the number of samples. Defaults to length(y).
# -- p is the number of predictors. Defaults to size(x,2).
# -- folds is the partition of the data. Defaults to a random partition into "nfolds" disjoint sets.
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1e-4.
# -- max_iter caps the number of permissible iterations in the IHT algorithm. Defaults to 1000.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
#    NOTA BENE: each processor outputs feed to the console without regard to the others,
#    so setting quiet=true can yield very messy output!
# -- logreg is a Boolean to indicate whether or not to perform logistic regression. Defaults to false (do linear regression).
# -- compute_model is a Boolean to indicate whether or not to recompute the best model. Defaults to false (do not recompute). 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
@compat function cv_exlstsq(
	x             :: DenseArray{Float64,2}, 
	y             :: DenseArray{Float64,1}, 
	path_length   :: Int, 
	numfolds      :: Int; 
	nrmsq         :: DenseArray{Float64,1} = vec(sumabs2(x,1)), 
	folds         :: DenseArray{Int,1}     = cv_get_folds(y,numfolds), 
	tol           :: Float64 = 1e-4, 
	n             :: Int     = length(y),
	p             :: Int     = size(x,2), 
	max_iter      :: Int     = 1000, 
	window        :: Int     = 20,
	compute_model :: Bool    = false,
	quiet         :: Bool    = true
)

	0 <= path_length <= p || throw(ArgumentError("Path length must be positive and cannot exceed number of predictors"))

	# preallocate vectors used in xval	
	mses    = zeros(Float64, path_length)	# vector to save mean squared errors
	my_refs = cell(numfolds)		# cell array to store RemoteRefs

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# the @sync macro ensures that we wait for all of them to finish before proceeding 
	@sync for i = 1:numfolds

		# one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression) 
		# @spawn(one_fold(...)) returns a RemoteRef to the result
		# store that RemoteRef so that we can query the result later 
		my_refs[i] = @spawn(one_fold(x, y, path_length, folds, i, max_iter=max_iter, quiet=quiet, window=window, n=n, p=p, nrmsq=nrmsq)) 
	end
	
	# recover MSEs on each worker
	for i = 1:numfolds
		mses += fetch(my_refs[i])
	end

	# average mses
	mses ./= numfolds

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
	if compute_model
		
		# initialize beta vector
		bp = zeros(Float64,p)
		perm = collect(1:p)

		# first use exchange algorithm to extract model
		bp = exchange_leastsq!(bp, x, y, perm, k, max_iter=max_iter, quiet=quiet, p=p) 

		# which components of beta are nonzero?
		# cannot use binary indices here since we need to return Int indices
		inferred_model = find( function f(x) x.!= 0.0; end, bp)

		# allocate the submatrix of x corresponding to the inferred model
		x_inferred = x[:,inferred_model]

		# now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
		# return it with the vector of MSEs
		Xty = BLAS.gemv('T', 1.0, x_inferred, y)	
		XtX = BLAS.gemm('T', 'N', 1.0, x_inferred, x_inferred)
		b = XtX \ Xty
		return mses, b, inferred_model
	end

	return mses
end
