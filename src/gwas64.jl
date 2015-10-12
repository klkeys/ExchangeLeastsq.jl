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
#    Defaults to an empty dict with typeasserts Int64 for the keys and Array{Float64,1} for the values.
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
@compat function exchange_leastsq!(
	bvec     :: DenseArray{Float64,1}, 
	X        :: BEDFile, 
	Y        :: DenseArray{Float64,1}, 
	perm     :: DenseArray{Int,1}, 
	r        :: Int; 
	inner    :: Dict{Int,DenseArray{Float64,1}} = Dict{Int,DenseArray{Float64,1}}(), 
	means    :: DenseArray{Float64,1} = mean(Float64,x), 
	invstds  :: DenseArray{Float64,1} = invstd(x, means),
	nrmsq    :: DenseArray{Float64,1} = sumsq(x, shared=false, means=means, invstds=invstds), 
	n        :: Int = length(Y), 
	p        :: Int = size(X,2), 
	Xb       :: DenseArray{Float64,1} = zeros(Float64, n), 
	res      :: DenseArray{Float64,1} = zeros(Float64, n), 
	df       :: DenseArray{Float64,1} = zeros(Float64, p), 
	tempp    :: DenseArray{Float64,1} = zeros(Float64, p), 
	tempn    :: DenseArray{Float64,1} = zeros(Float64, n), 
	tempn2   :: DenseArray{Float64,1} = zeros(Float64, n), 
	dotprods :: DenseArray{Float64,1} = zeros(Float64, p), 
	indices  :: BitArray{1} = falses(p), 
	window   :: Int     = r, 
	max_iter :: Int     = 10000, 
	tol      :: Float64 = 1e-4, 
	quiet    :: Bool    = false
)

	# error checking
	n == length(tempn)    || throw(DimensionMismatch("length(Y) != length(tempn)"))
	n == length(tempn2)   || throw(DimensionMismatch("length(Y) != length(tempn2)"))
	n == length(res)      || throw(DimensionMismatch("length(Y) != length(res)"))
	p == length(bvec)     || throw(DimensionMismatch("Number of predictors != length(bvec)"))
	p == length(df)       || throw(DimensionMismatch("length(bvec) != length(df)"))
	p == length(tempp)    || throw(DimensionMismatch("length(bvec) != length(tempp)"))
	p == length(dotprods) || throw(DimensionMismatch("length(bvec) != length(dotprods)"))
	p == length(nrmsq)    || throw(DimensionMismatch("length(bvec) != length(nrmsq)"))
	p == length(perm)     || throw(DimensionMismatch("length(bvec) != length(perm)"))
	0 <= r <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(bvec)"))
	tol >= eps()          || throw(ArgumentError("Global tolerance must exceed machine precision"))
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
	a       = 0.0
	b       = 0.0
	adb     = 0.0	# = a / b
	c       = 0.0
	d       = 0.0
	betal   = 0.0	# store lth component of bvec 
	rss     = Inf	# residual sum of squares || Y - XB ||^2
	old_rss = Inf	# previous residual sum of squares 

	# obtain top r components of bvec in magnitude
	selectpermk!(perm,bvec, r, p=p)
	update_indices!(indices, bvec, p=p)

	# update X*b
	xb!(Xb,X,bvec,indices,r, means=means, invstds=invstds)

	# update residuals based on Xb 
	difference!(res, Y, Xb, n=n)

	# save value of RSS before starting algorithm
	rss = sumabs2(res)

	# compute inner products of X and residuals 
	# this is basically the negative gradient
	xty!(df, X, res, means=means, invstds=invstds)

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
				inner[l] = xty(X, tempn, means=means, invstds=invstds)
			end
			copy!(dotprods,inner[l])

			# save values to determine best estimate for current predictor
			b   = nrmsq[l]
			a   = df[l] + betal*b
			adb = a / b
			k   = i

			# inner loop compares current predictor j against all remaining predictors j+1,...,p
			for j = (r+1):p
				idx = perm[j]
				c   = df[idx] + betal*dotprods[idx]
				d   = nrmsq[idx]

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
				inner[m] = xty(X, tempn2, means=means, invstds=invstds)
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
				bvec[j] = 0.0
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
	throw(error("Maximum iterations $(max_iter) reached! Return value may not be correct.\n"))
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
# -- tol is the convergence tolerance to pass to the path computations. Defaults to 1e-4.
# -- max_iter caps the number of permissible iterations in the algorithm. Defaults to 1000.
# -- quiet is a Boolean to activate output. Defaults to true (no output).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu 
@compat function one_fold(
	x           :: BEDFile, 
	y           :: DenseArray{Float64,1}, 
	path_length :: Int, 
	folds       :: DenseArray{Int,1}, 
	fold        :: Int; 
	means       :: DenseArray{Float64,1} = mean(Float64, x), 
	invstds     :: DenseArray{Float64,1} = invstd(x, y=means), 
	nrmsq       :: DenseArray{Float64,1} = sumsq(x, shared=false, means=means, invstds=invstds), 
	p           :: Int  = size(x,2), 
	max_iter    :: Int  = 1000, 
	window      :: Int  = 20, 
	quiet       :: Bool = true 
)

	# find testing indices
	test_idx = folds .== fold

	# preallocate vector for output
	myerrors = zeros(Float64, sum(test_idx))

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
	inner     = Dict{Int,DenseArray{Float64,1}}()

	# declare all temporary arrays
	df         = zeros(Float64, p)	# X'(Y - Xbeta)
	tempp      = zeros(Float64, p)	# temporary array of length p
	dotprods   = zeros(Float64, p)	# hold in memory the dot products for current index
	bout       = zeros(Float64, p)	# output array for beta
	tempn      = zeros(Float64, n)	# temporary array of length n 
	tempn2     = zeros(Float64, n)	# temporary array of length n 
	res        = zeros(Float64, n)	# Y - Xbeta
	bnonzeroes = falses(p)	        # indicate nonzero components of beta

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
	x             :: BEDFile,
	y             :: DenseArray{Float64,1}, 
	path_length   :: Int, 
	numfolds      :: Int; 
	nrmsq         :: DenseArray{Float64,1} = sumsq(x, shared=false, means=means, invstds=invstds), 
	means         :: DenseArray{Float64,1} = mean(Float64, x),
	invstds       :: DenseArray{Float64,1} = invstd(x, y=means),
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
		my_refs[i] = @spawn(one_fold(x, y, path_length, folds, i, max_iter=max_iter, quiet=quiet, window=window, n=n, p=p, nrmsq=nrmsq, means=means, invstds=invstds)) 
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
		bp = zeros(Float64, p)
		perm = collect(1:p)
		x_inferred = zeros(Float64, n, k)

		# first use exchange algorithm to extract model
		bp = exchange_leastsq!(bp, x, y, perm, k, max_iter=max_iter, quiet=quiet, p=p, means=means, invstds=invstds) 

		# which components of beta are nonzero?
		# cannot use binary indices here since we need to return Int indices
		inferred_model = find( function f(x) x.!= 0.0; end, bp)

		# allocate the submatrix of x corresponding to the inferred model
		decompress_genotypes!(x_inferred, x, inferred_model, means=means, invstds=invstds)

		# now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
		# return it with the vector of MSEs
		Xty = BLAS.gemv('T', 1.0, x_inferred, y)	
		XtX = BLAS.gemm('T', 'N', 1.0, x_inferred, x_inferred)
		b   = XtX \ Xty
		return mses, b, inferred_model
	end

	return mses
end
