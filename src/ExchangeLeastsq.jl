module ExchangeLeastsq

using GLMNet
using RegressionTools
using PLINK
using Distances: euclidean, chebyshev
using StatsBase: sample
import Base.LinAlg.BLAS.gemv
import Base.LinAlg.BLAS.gemv!
import Base.LinAlg.BLAS.axpy!

export exchange_leastsq
export exchange_leastsq!
export cv_exlstsq
export test_exchangeleastsq
export test_exleastsq
export test_cv_exlstsq
export compare_exlstsq
export test_exchangeleastsq_plink



###################
### SUBROUTINES ###
###################

## PARTIAL PERMUTATION SORT ON INDICES OF A VECTOR
## This subroutine replaces sortperm to get the top k components of a vector in magnitude.
## By performing only a partial sort, it saves in compute time and memory.
## Feed selectperm() a preallocated vector z of indices for optimal performance.
##function selectperm(x,k::Int; p::Int = length(x), z::Array{Int,1} = [1:p])
#function selectperm!(z::DenseArray{Int,1}, x::DenseArray{Float64,1}, k::Int; p::Int = length(x)) 
#	k <= p                 || throw(ArgumentError("selectperm: k cannot exceed length of x!"))
#	length(z) == length(x) || throw(DimensionMismatch("Arguments z and x do not have the same length")) 
#	return select!(z, 1:k, by = (i)->abs(x[i]), rev = true)
#end 


# calculate residuals (Y - XB)  piecemeal
# first line does residuals = - X * x_mm
# next line does residuals = residuals + Y = Y - X*x_mm
function update_residuals!(r::Array{Float64,1}, X::Array{Float64,2}, Y::Array{Float64,1}, b::Array{Float64,1})

	# ensure conformable arguments
	length(r) == length(Y) || DimensionMismatch("update_residuals!: output vector must have same length as Y")
	length(b) == size(X,2) || DimensionMismatch("update_residuals!: X and beta not conformable")

	copy!(r, Y)
	BLAS.gemv!('N', -1.0, X, b, 1.0, r)
end


# PERFORM A*X + Y - B*Z 
#
# The silly name is based on BLAS axpy (A*X Plus Y), except that this function performs A*X Plus Y Minus B*Z.
# The idea behind axpymz!() is to perform the computation in one pass over the arrays. The output is the same as 
# > @devec y = y + a*x - b*z
function axpymbz!(j::Int, y::DenseArray{Float64,1}, a::Float64, x::DenseArray{Float64,}, b::Float64, z::DenseArray{Float64,1})
	y[j] + a*x[j] - b*z[j]
end

function axpymbz!(y::Array{Float64,1}, a, x::Array{Float64,1}, b, z::Array{Float64,1}; p::Int = length(y)) 
	@inbounds @simd for i = 1:p
		y[i] = axpymbz!(i, y, a, x, b, z) 
	end
	return y
end


function axpymbz!(y::SharedArray{Float64,1}, a, x::SharedArray{Float64,1}, b, z::SharedArray{Float64,1}; p::Int = length(y)) 
	@sync @inbounds @parallel for i = 1:p
		y[i] = y[i] + a*x[i] - b*z[i]
	end
	return y
end





######################
### MAIN FUNCTIONS ###
######################


# EXCHANGE ALGORITHM FOR L0-PENALIZED LEAST SQUARES REGRESSION 
# 
# This function minimizes the residual sum of squares
# 
# RSS = 0.5*|| Y - Xbeta ||_2^2
#
# subject to beta having no more than r nonzero components. The function will compute a B for a given value of r.
# For optimal accuracy and performance, this function should be run for multiple values of r over a path.
# In doing so, one should reuse the arguments beta, perm, and inner.
#
# Arguments:
# -- beta is the p-dimensional warm-start for the iterate
# -- X is the n x p statistical design matrix
# -- Y is the n-dimensional response vector
# -- perm is a p-dimensional array of integers that sort beta in descending order by magnitude
# -- r is the desired number of nonzero components in beta
# 
# Optional Arguments:
# -- nrmsq is the vector to store the squared norms of the columns of X. Defaults to vec(sumsq(X,1))
# -- n and p are the dimensions of X; the former actually defaults to length(Y) while the latter defaults to size(X,2).
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
function exchange_leastsq!(betavec::DenseArray{Float64,1}, X::DenseArray{Float64,2}, Y::DenseArray{Float64,1}, perm::DenseArray{Int,1}, r::Int; nrmsq::DenseArray{Float64,1} = vec(sumsq(X,1)), n::Int = length(Y), p::Int = size(X,2), max_iter::Int = 10000, tol::Float64 = 1e-6, inner = Dict{Int,DenseArray{Float64,1}}(), quiet::Bool = false, res::DenseArray{Float64,1} = zeros(n), df::DenseArray{Float64,1} = zeros(p), tempn::DenseArray{Float64,1} = zeros(n), tempp::DenseArray{Float64,1} = zeros(p), tempn2::DenseArray{Float64,1} = copy(tempn), dotprods::DenseArray{Float64,1} = zeros(p), window::Int = r)

	# error checking
	n == size(X,1)        || throw(DimensionMismatch("length(Y) != size(X,1)"))
	n == length(tempn)    || throw(DimensionMismatch("length(Y) != length(tempn)"))
	n == length(tempn2)   || throw(DimensionMismatch("length(Y) != length(tempn2)"))
	n == length(res)      || throw(DimensionMismatch("length(Y) != length(res)"))
	p == length(betavec)  || throw(DimensionMismatch("length(betavec) != length(betavec)"))
	p == length(df)       || throw(DimensionMismatch("length(betavec) != length(df)"))
	p == length(tempp)    || throw(DimensionMismatch("length(betavec) != length(tempp)"))
	p == length(dotprods) || throw(DimensionMismatch("length(betavec) != length(dotprods)"))
	p == length(nrmsq)    || throw(DimensionMismatch("length(betavec) != length(nrmsq)"))
	p == length(perm)     || throw(DimensionMismatch("length(betavec) != length(perm)"))
	0 <= r <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(betavec)"))
	tol >= eps()          || throw(ArgumentError("Global tolerance must exceed machine precision"))
	max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
	0 <= window <= r      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))


	# declare all integers 
	i::Int    = 0	# used for iterations
	iter::Int = 0	# used for outermost loop
	j::Int    = 0	# used for iterations
	k::Int    = 0	# used for indexing
	l::Int    = 0	# used for indexing
	m::Int    = 0	# used for indexing 
	idx::Int  = 0	# used for indexing

	# declare all floats
	a::Float64       = 0.0
	b::Float64       = 0.0
	adb::Float64     = 0.0	# = a / b
	c::Float64       = 0.0
	d::Float64       = 0.0
	betal::Float64   = 0.0	# store lth component of betavec 
	rss::Float64     = 0.0	# residual sum of squares || Y - XB ||^2
	old_rss::Float64 = Inf	# previous residual sum of squares 

	# obtain top r components of betavec in magnitude
	selectperm!(perm,betavec, r, p=p)

	# compute partial residuals based on top r components of perm vector
	RegressionTools.update_partial_residuals!(res, Y, X, perm, betavec, r, n=n, p=p)

	# save value of RSS before starting algorithm
	rss = sumabs2(res)

	# compute inner products of X and residuals 
	# this is basically the negative gradient
	BLAS.gemv!('T', 1.0, X, res, 0.0, df)

	# outer loop controls number of total iterations for algorithm run on one r
	for iter = 1:(max_iter)

		# output algorithm progress to console
		quiet || println("\titer = ", iter, ", RSS = ", rss)

		# middle loop tests each of top r parameters (by magnitude?)
		for i = abs(r-window+1):r

			# save information for current value of i
			l     = perm[i]
			betal = betavec[l]
			update_col!(tempn, X, l, n=n, p=p)	# tempn now holds X[:,l]

			# if necessary, compute inner products of current predictor against all other predictors
			# store this information in Dict inner
			# for current index, hold dot products in memory for duration of inner loop
			# the if/else statement below is the same as but faster than
			# > dotprods = get!(inner, l, BLAS.gemv('T', 1.0, X, tempn))
			if !haskey(inner, l)
				inner[l] = BLAS.gemv('T', 1.0, X, tempn)
			end
			copy!(dotprods,inner[l])

			# save values to determine best estimate for current predictor
#			a   = fma(betal, nrmsq, df, l)	# a = df[l] + betal*b
			b   = nrmsq[l]
			a   = df[l] + betal*b
			adb = a / b
			k   = i

			# inner loop compares current predictor j against all remaining predictors j+1,...,p
			for j = (r+1):p
				idx = perm[j]
#				c   = fma(betal, dotprods, df, idx)	# c = df[idx] + betal*dotprods[idx]
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
			update_col!(tempn2, X, m, n=n, p=p)	# tempn2 now holds X[:,m]
			axpymbz!(res, betal, tempn, adb, tempn2, p=n)

			# if necessary, compute inner product of current predictor against all other predictors
			# save in our Dict for future reference
			# compare in performance to
			# > tempp = get!(inner, m, BLAS.gemv('T', 1.0, X, tempn2))
			if !haskey(inner, m)
				inner[m] = BLAS.gemv('T', X, tempn2)
			end
			copy!(tempp, inner[m])

			# also update df
			axpymbz!(df, betal, dotprods, adb, tempp, p=p)

			# now swap best predictor with current predictor
			j          = perm[i]
			perm[i]    = perm[k] 
			perm[k]    = j 
			betavec[m] = adb
			if k != i
				betavec[j] = 0.0
			end

		end # end middle loop over predictors 

		# update residual sum of squares
		rss = sumabs2(res)

		# test for descent failure 
		# if no descent failure, then test for convergence
		# if not converged, then save RSS and continue
		ascent    = rss > old_rss + tol
		converged = abs(old_rss - rss) / abs(old_rss + 1) < tol 

		ascent && throw(error("Descent error detected at iteration $(iter)!\nOld RSS: $(old_rss)\nRSS: $(rss)")) 
		(converged || ascent) && return betavec
		old_rss = rss
		isnan(rss) && throw(error("Objective function is NaN!"))
		isinf(rss) && throw(error("Objective function is Inf!"))

	end # end outer iteration loop

	# at this point, maximum iterations reached
	# warn and return betavec
	throw(error("Maximum iterations $(max_iter) reached! Return value may not be correct.\n"))
	return betavec

end # end exchange_leastsq



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
# -- betavec is the p-dimensional warm-start for the iterate.
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
function exchange_leastsq!(betavec::DenseArray{Float64,1}, X::BEDFile, Y::DenseArray{Float64,1}, perm::DenseArray{Int,1}, r::Int, p::Int; Xb::DenseArray{Float64,1} = zeros(p), nrmsq::DenseArray{Float64,1} = sumsq(x,n,p) , n::Int = length(Y), max_iter::Int = 10000, tol::Float64 = 1e-4, inner = Dict{Int,DenseArray{Float64,1}}(), quiet::Bool = false, res::DenseArray{Float64,1} = zeros(n), df::DenseArray{Float64,1} = zeros(p), tempn::DenseArray{Float64,1} = zeros(n), tempp::DenseArray{Float64,1} = zeros(p), tempn2::DenseArray{Float64,1} = copy(tempn), dotprods::DenseArray{Float64,1} = zeros(p), window::Int = r, indices::BitArray{1} = falses(p), means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means))

	# error checking
	n == length(tempn)    || throw(DimensionMismatch("length(Y) != length(tempn)"))
	n == length(tempn2)   || throw(DimensionMismatch("length(Y) != length(tempn2)"))
	n == length(res)      || throw(DimensionMismatch("length(Y) != length(res)"))
	p == length(betavec)  || throw(DimensionMismatch("Number of predictors != length(betavec)"))
	p == length(df)       || throw(DimensionMismatch("length(betavec) != length(df)"))
	p == length(tempp)    || throw(DimensionMismatch("length(betavec) != length(tempp)"))
	p == length(dotprods) || throw(DimensionMismatch("length(betavec) != length(dotprods)"))
	p == length(nrmsq)    || throw(DimensionMismatch("length(betavec) != length(nrmsq)"))
	p == length(perm)     || throw(DimensionMismatch("length(betavec) != length(perm)"))
	0 <= r <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(betavec)"))
	tol >= eps()          || throw(ArgumentError("Global tolerance must exceed machine precision"))
	max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
	0 <= window <= r      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))

	# declare all integers 
	i::Int    = 0	# used for iterations
	iter::Int = 0	# used for outermost loop
	j::Int    = 0	# used for iterations
	k::Int    = 0	# used for indexing
	l::Int    = 0	# used for indexing
	m::Int    = 0	# used for indexing 
	idx::Int  = 0	# used for indexing

	# declare all floats
	a::Float64       = 0.0
	b::Float64       = 0.0
	adb::Float64     = 0.0	# = a / b
	c::Float64       = 0.0
	d::Float64       = 0.0
	betal::Float64   = 0.0	# store lth component of betavec 
	rss::Float64     = Inf	# residual sum of squares || Y - XB ||^2
	old_rss::Float64 = Inf	# previous residual sum of squares 

	# obtain top r components of betavec in magnitude
#	println("selectperm!")
#	@time selectperm!(perm,betavec, r, p=p)
	selectperm!(perm,betavec, r, p=p)
	indices = falses(p)
	for i = 1:r
		indices[perm[i]] = true
	end

	# compute partial residuals based on top r components of perm vector
#	PLINK.update_partial_residuals!(res, Y, X, perm, betavec, r) 

#	println("xb!")
#	@time xb!(Xb,X,betavec,indices,r, means=means, invstds=invstds)
	xb!(Xb,X,betavec,indices,r, means=means, invstds=invstds)
#	println("update_partial_residuals!")
#	@time PLINK.update_partial_residuals!(res, Y, X, indices, betavec, r, Xb=Xb, means=means, invstds=invstds)
#	PLINK.update_partial_residuals!(res, Y, X, indices, betavec, r, Xb=Xb, means=means, invstds=invstds)
	difference!(res, Y, Xb, n=n)

	# save value of RSS before starting algorithm
	rss = sumabs2(res)

	# compute inner products of X and residuals 
	# this is basically the negative gradient
#	println("xty!")
#	@time xty!(df, X, res, means=means, invstds=invstds)
	xty!(df, X, res, means=means, invstds=invstds)

	# outer loop controls number of total iterations for algorithm run on one r
	for iter = 1:(max_iter)

		# output algorithm progress to console
		quiet || println("\titer = ", iter, ", RSS = ", rss)

		# middle loop tests each of top r parameters (by magnitude?)
		for i = abs(r-window+1):r

			# save information for current value of i
			l     = perm[i]
			betal = betavec[l]
#			println("decompress_genotypes!")
#			@time decompress_genotypes!(tempn, X, l, means, invstds) # tempn now holds X[:,l]
			decompress_genotypes!(tempn, X, l, means, invstds) # tempn now holds X[:,l]

			# if necessary, compute inner products of current predictor against all other predictors
			# store this information in Dict inner
			# for current index, hold dot products in memory for duration of inner loop
			# the if/else statement below is the same as but faster than
			# > dotprods = get!(inner, l, BLAS.gemv('T', 1.0, X, tempn))
			if !haskey(inner, l)
#				println("xty")
#				@time inner[l] = xty(X, tempn, means=means, invstds=invstds)
				inner[l] = xty(X, tempn, means=means, invstds=invstds)
			end
			copy!(dotprods,inner[l])

			# save values to determine best estimate for current predictor
#			a   = fma(betal, nrmsq, df, l)	# a = df[l] + betal*b
			b   = nrmsq[l]
			a   = df[l] + betal*b
			adb = a / b
			k   = i

			# inner loop compares current predictor j against all remaining predictors j+1,...,p
			for j = (r+1):p
				idx = perm[j]
#				c   = fma(betal, dotprods, df, idx)	# c = df[idx] + betal*dotprods[idx]
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
#			println("decompress_genotypes!")
#			@time decompress_genotypes!(tempn2, X, m, means, invstds) # tempn now holds X[:,l]
			decompress_genotypes!(tempn2, X, m, means, invstds) # tempn now holds X[:,l]
#			println("axpymbz!")
#			@time axpymbz!(res, betal, tempn, adb, tempn2, p=n)
			axpymbz!(res, betal, tempn, adb, tempn2, p=n)

			# if necessary, compute inner product of current predictor against all other predictors
			# save in our Dict for future reference
			# compare in performance to
			# > tempp = get!(inner, m, BLAS.gemv('T', 1.0, X, tempn2))
			if !haskey(inner, m)
#				println("xty")
#				@time inner[m] = xty(X, tempn2, means=means, invstds=invstds)
				inner[m] = xty(X, tempn2, means=means, invstds=invstds)
			end
			copy!(tempp, inner[m])

			# also update df
#			println("axpymbz!")
#			@time axpymbz!(df, betal, dotprods, adb, tempp, p=p)
			axpymbz!(df, betal, dotprods, adb, tempp, p=p)

			# now swap best predictor with current predictor
			j          = perm[i]
			perm[i]    = perm[k] 
			perm[k]    = j 
			betavec[m] = adb
			if k != i
				betavec[j] = 0.0
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
		(converged || ascent) && return betavec
		old_rss = rss

	end # end outer iteration loop

	# at this point, maximum iterations reached
	# warn and return betavec
	throw(error("Maximum iterations $(max_iter) reached! Return value may not be correct.\n"))
	return betavec

end # end exchange_leastsq





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
#function one_fold(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path_length::Int, test_idx::DenseArray{Int,1}; n::Int = length(y), p::Int = size(x,2), max_iter::Int = 1000, quiet::Bool = true, window::Int = 20, nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1))) 
#function one_fold(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path_length::Int, folds::DenseArray{Int,1}, fold::Int; n::Int = length(y), p::Int = size(x,2), max_iter::Int = 1000, quiet::Bool = true, window::Int = 20, nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1))) 
function one_fold(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path_length::Int, folds::DenseArray{Int,1}, fold::Int; p::Int = size(x,2), max_iter::Int = 1000, quiet::Bool = true, window::Int = 20, nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1))) 

	# find testing indices
	test_idx = folds .== fold

	# preallocate vector for output
	myerrors = zeros(sum(test_idx))

	# train_idx is the vector that indexes the TRAINING set
#	train_idx = setdiff(collect(1:n), test_idx)
	train_idx = !test_idx

	# how big is training sample?
	const n = length(train_idx)

	# allocate the arrays for the training set
	x_train   = x[train_idx,:]
	y_train   = y[train_idx] 
	b         = zeros(p)
	betas     = zeros(p,path_length)
	perm      = collect(1:p)
	inner     = Dict{Int,DenseArray{Float64,1}}()

	# declare all temporary arrays
	res::DenseArray{Float64,1}        = zeros(n)	# Y - Xbeta
	df::DenseArray{Float64,1}         = zeros(p)	# X'(Y - Xbeta)
	tempn::DenseArray{Float64,1}      = zeros(n)	# temporary array of length n 
	tempn2::DenseArray{Float64,1}     = zeros(n)	# temporary array of length n 
	tempp::DenseArray{Float64,1}      = zeros(p)	# temporary array of length p
	dotprods::DenseArray{Float64,1}   = zeros(p)	# hold in memory the dot products for current index
	bnonzeroes::BitArray{1}           = falses(p)	# indicate nonzero components of beta
	bout::DenseArray{Float64,1}       = zeros(p)	# output array for beta

	# loop over each element of path
	for i = 1:path_length
		# compute the regularization path on the training set
		bout = exchange_leastsq!(b, x_train, y_train, perm, i, inner=inner, max_iter=max_iter, quiet=quiet, n=n, p=p, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window = min(window, i), nrmsq=nrmsq) 

		# find the support of bout
		bnonzeroes = bout .!= 0.0

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
#function one_fold(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path_length::Int, test_idx::DenseArray{Int,1}; n::Int = length(y), p::Int = size(x,2), max_iter::Int = 1000, quiet::Bool = true, window::Int = 20, nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1))) 
#function one_fold(x::DenseArray{Float64,2}, y::DenseArray{Float64,1}, path_length::Int, folds::DenseArray{Int,1}, fold::Int; n::Int = length(y), p::Int = size(x,2), max_iter::Int = 1000, quiet::Bool = true, window::Int = 20, nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1))) 
function one_fold(x::BEDFile, y::DenseArray{Float64,1}, path_length::Int, folds::DenseArray{Int,1}, fold::Int; p::Int = size(x,2), max_iter::Int = 1000, quiet::Bool = true, window::Int = 20, nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1)), means::DenseArray{Float64,1} = mean(x), invstds::DenseArray{Float64,1} = invstd(x, y=means)) 

	# find testing indices
	test_idx = folds .== fold

	# preallocate vector for output
	myerrors = zeros(sum(test_idx))

	# train_idx is the vector that indexes the TRAINING set
#	train_idx = setdiff(collect(1:n), test_idx)
	train_idx = !test_idx

	# how big is training sample?
	const n = length(train_idx)

	# allocate the arrays for the training set
	x_train   = x[train_idx,:]
	y_train   = y[train_idx] 
	b         = zeros(p)
	betas     = zeros(p,path_length)
	perm      = collect(1:p)
	inner     = Dict{Int,DenseArray{Float64,1}}()

	# declare all temporary arrays
	res::DenseArray{Float64,1}        = zeros(n)	# Y - Xbeta
	df::DenseArray{Float64,1}         = zeros(p)	# X'(Y - Xbeta)
	tempn::DenseArray{Float64,1}      = zeros(n)	# temporary array of length n 
	tempn2::DenseArray{Float64,1}     = zeros(n)	# temporary array of length n 
	tempp::DenseArray{Float64,1}      = zeros(p)	# temporary array of length p
	dotprods::DenseArray{Float64,1}   = zeros(p)	# hold in memory the dot products for current index
	bnonzeroes::BitArray{1}           = falses(p)	# indicate nonzero components of beta
	bout::DenseArray{Float64,1}       = zeros(p)	# output array for beta

	# loop over each element of path
	for i = 1:path_length
		# compute the regularization path on the training set
		bout = exchange_leastsq!(b, x_train, y_train, perm, i, inner=inner, max_iter=max_iter, quiet=quiet, n=n, p=p, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window = min(window, i), nrmsq=nrmsq) 

		# find the support of bout
		bnonzeroes = bout .!= 0.0

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
function cv_exlstsq(x::DenseArray{Float64,2}, 
				 	y::DenseArray{Float64,1}, 
					path_length::Int, 
					numfolds::Int; 
					n::Int = length(y),
					p::Int = size(x,2), 
					tol::Float64 = 1e-4, 
					max_iter::Int = 1000, 
					quiet::Bool = true, 
					folds::DenseArray{Int,1} = cv_get_folds(y,numfolds), 
					compute_model::Bool = false,
					window::Int = 20,
					nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1))) 


	0 <= path_length <= p || throw(ArgumentError("Path length must be positive and cannot exceed number of predictors"))

	# preallocate vectors used in xval	
	mses    = zeros(path_length)	# vector to save mean squared errors
	my_refs = cell(numfolds)		# cell array to store RemoteRefs

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# the @sync macro ensures that we wait for all of them to finish before proceeding 
	@sync for i = 1:numfolds

		# test_idx saves the numerical identifier of the vector of indices corresponding to the ith fold 
		# this vector will indiate which part of the data to hold out for testing 
#		test_idx  = find( function f(x) x .== i; end, folds)

		# one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression) 
		# @spawn(one_fold(...)) returns a RemoteRef to the result
		# store that RemoteRef so that we can query the result later 
#		my_refs[i] = @spawn(one_fold(x, y, path_length, test_idx, max_iter=max_iter, quiet=quiet, window=window, p=p, nrmsq=nrmsq)) 
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
		bp = zeros(p)
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
function cv_exlstsq(x::BEDFile,
				 	y::DenseArray{Float64,1}, 
					path_length::Int, 
					numfolds::Int; 
					n::Int = length(y),
					p::Int = size(x,2), 
					tol::Float64 = 1e-4, 
					max_iter::Int = 1000, 
					quiet::Bool = true, 
					folds::DenseArray{Int,1} = cv_get_folds(y,numfolds), 
					compute_model::Bool = false,
					window::Int = 20,
					nrmsq::DenseArray{Float64,1} = vec(sumsq(x,1)),
					means::DenseArray{Float64,1} = mean(x),
					invstds::DenseArray{Float64,1} = invstd(x, y=means)) 


	0 <= path_length <= p || throw(ArgumentError("Path length must be positive and cannot exceed number of predictors"))

	# preallocate vectors used in xval	
	mses    = zeros(path_length)	# vector to save mean squared errors
	my_refs = cell(numfolds)		# cell array to store RemoteRefs

	# want to compute a path for each fold
	# the folds are computed asynchronously
	# the @sync macro ensures that we wait for all of them to finish before proceeding 
	@sync for i = 1:numfolds

		# test_idx saves the numerical identifier of the vector of indices corresponding to the ith fold 
		# this vector will indiate which part of the data to hold out for testing 
#		test_idx  = find( function f(x) x .== i; end, folds)

		# one_fold returns a vector of out-of-sample errors (MSE for linear regression, MCE for logistic regression) 
		# @spawn(one_fold(...)) returns a RemoteRef to the result
		# store that RemoteRef so that we can query the result later 
#		my_refs[i] = @spawn(one_fold(x, y, path_length, test_idx, max_iter=max_iter, quiet=quiet, window=window, p=p, nrmsq=nrmsq)) 
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
		bp = zeros(p)
		perm = collect(1:p)
		x_inferred = zeros(x.n, k)

		# first use exchange algorithm to extract model
		bp = exchange_leastsq!(bp, x, y, perm, k, max_iter=max_iter, quiet=quiet, p=p, means=means, invstds=invstds) 

		# which components of beta are nonzero?
		# cannot use binary indices here since we need to return Int indices
		inferred_model = find( function f(x) x.!= 0.0; end, bp)

		# allocate the submatrix of x corresponding to the inferred model
#		x_inferred = x[:,inferred_model]
		decompress_genotypes!(x_inferred, x, inferred_model, means=means, invstds=invstds)

		# now estimate b with the ordinary least squares estimator b = inv(x'x)x'y 
		# return it with the vector of MSEs
		Xty = BLAS.gemv('T', 1.0, x_inferred, y)	
		XtX = BLAS.gemm('T', 'N', 1.0, x_inferred, x_inferred)
		b = XtX \ Xty
		return mses, b, inferred_model
	end

	return mses
end





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
function test_exchangeleastsq(x_path::ASCIIString = "/Users/kkeys/Downloads/wtccc-n2000p32307.txt", y_path::ASCIIString = "/Users/kkeys/Downloads/chr1.Y.300", b_path::ASCIIString = "/Users/kkeys/Downloads/chr1.causal.300", r::Int = 300, tol::Float64 = 1e-6, max_iter::Int = 100, quiet::Bool = false, extra::Int = 100, window::Int = r) 

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
    println("\nTrue positives: ", countnz(bk), "/", 300, ".")
    println("Distances to true model:")
    println("Chebyshev (L-Inf) = ", chebyshev(bk, be), ".")
    println("Euclidean (L-2)   = ", euclidean(bk, be), ".")
end


## TEST THE EXCHANGE ALGORITHM FOR LEAST SQUARES REGRESSION OVER PLINK BINARY DATA
## This subroutine runs the exchange algorithm on a GWAS dataset from the WTCCC.
#function test_exchangeleastsq_plink(n::Int, p::Int; x_path::ASCIIString = "/Users/kkeys/Downloads/wtccc_full/hapmap_r23a/hapmap_r23a.bed", y_path::ASCIIString = "/Users/kkeys/Downloads/wtccc_full/hapmap_r23a/Y.hapmap.causal.11.noise05.txt", b_path::ASCIIString = "/Users/kkeys/Downloads/wtccc_full/hapmap_r23a/B.hapmap.causal.11.noise05.txt", r::Int = 11, tol::Float64 = 1e-4, max_iter::Int = 100, quiet::Bool = false, extra::Int = 10, window::Int = 10) 
#
#    # repeat input options
#	quiet || begin
#		println("This function will test the IHT algorithm.")
#		println("Given options:")
#		println("\tPath to X  = ", x_path)
#		println("\tPath to Y  = ", y_path)
#		println("\tPath to B  = ", b_path)
#		println("\tModel size = ", r)
#		println("\ttolerance  = ", tol, "\n\tmax_iter   = ", max_iter)
#	end
#
#    # precompile @time macro for later use
#    quiet || println("\nCompiling @time macro...")
#    @time 1+1; 
#
#    # now load data
#    quiet || println("\nLoading data...")
#    tic()
#
#    # load design matrix
##	x = read_bedfile(x_path)
#	x = BEDFile(x_path, shared=true)
#
#    # load response vector
#    y = readdlm(y_path)
#	y = vec(y) # need y to be 1D
#	const y = convert(SharedArray{Float64,1}, y)
#
#    # load model
#    B = readdlm(b_path)
#
#    # need indices for true model
#    # add 1 since Julia uses 1-indexing
#    const bidx = convert(Array{Int,1}, B[:,2])
#
#    # need components of true model
#    const be = convert(Array{Float64,1}, B[:,3])
#
#	quiet || begin 
#		println("True Model:\n\tIndex\tValue")
#		for i = 1:length(bidx)
#			println("\t", bidx[i], "\t", be[i])
#		end
#	end
#
#    # how long did it take to load files?
#	file_time = toq()
#    quiet || println("\nFiles took ", file_time, " seconds to load.")
#
#	# discard B to recover some memory
#	quiet || println("Object B is no longer needed, recovering memory...")
#	tic()
#	B = false
#	gc()
#	B_time = toq()
#	quiet || println("Discarding B took ", B_time, " seconds.")
#
#	# load parameters
#	quiet || println("\nAllocating temporary arrays...")
#	tic()
#
#	# declare all temporary arrays
#	b        = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# parameter vector
#	res      = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# Y - Xbeta
#	df       = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# X'(Y - Xbeta)
#	tempn    = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# temporary array of length n 
#	tempn2   = SharedArray(Float64, n, init = S -> S[localindexes(S)] = 0.0)		# temporary array of length n 
#	tempp    = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# temporary array of length p
#	dotprods = SharedArray(Float64, p, init = S -> S[localindexes(S)] = 0.0)		# hold in memory the dot products for current index
#
#	# permutation vector of b has a more complicated initialization
##	perm     = SharedArray(Int,     p, init = S -> for i = 1:length(S) S[i] = i; end)
#	perm     = SharedArray(Int,     p, init = S -> S[localindexes(S)] = localindexes(S))
#
#	# declare associative array for storing inner products
#	inner = Dict{Int,DenseArray{Float64,1}}()
#
#	# how long did the allocation take?
#	alloc_time = toq()
#	quiet || println("Temporary arrays allocated in ", alloc_time, " seconds.")
#
#	# precompute sum of squares for each column of x
#	quiet || println("\nPrecomputing squared Euclidean norms of matrix columns...")
#	tic()
#	const nrmsq = vec(sumsq(x, shared=true))
#	norm_time = toq()
#	quiet || println("Column norms took ", norm_time, " seconds to compute")
#
#	# declare any return values
#	iter::Int = 0
#
#	# print time spent on parameter initialization
#	parameter_time = norm_time + alloc_time + file_time + B_time
#	quiet || println("Parameter initialization took ", parameter_time, " seconds.")
#
#
#	# check all parameters for NaN
##	println("Any NaN in x? ", any(isnan(x)))
##	println("Any NaN in y? ", any(isnan(y)))
##	println("Any NaN in b? ", any(isnan(b)))
##	println("Any NaN in nrmsq? ", any(isnan(nrmsq)))
##	println("Any NaN in res? ", any(isnan(res)))
##	println("Any NaN in tempn? ", any(isnan(tempn)))
##	println("Any NaN in tempn2? ", any(isnan(tempn2)))
##	println("Any NaN in tempp? ", any(isnan(tempp)))
##	println("Any NaN in dotprods? ", any(isnan(tempp)))
#
#    # run exchange 
#    quiet || println("\nRunning exchange algorithm...")
#
#	# reset timer and compute path
#	tic()
#	@time begin
#		for i = 2:(r+extra)
#			b = exchange_leastsq!(b, x, y, perm, i, p, n=n, inner=inner, max_iter=max_iter, quiet=quiet, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window=min(i,window))
#			rss = sumsq(sdata(res))
#
#			# print results from this step of regularization path
#			quiet || println("#pred = ", i, ", rss = ", rss, ", #truepos = ", countnz(b[bidx])) 
#		end
#	end	
#
#	# print time spent on computation of regularization path
#	quiet || println("Path from 1 to ", r+extra, " took ", toq(), " seconds to compute.")
#
#    # recover vector from output
#    bk = b[bidx]
#
#    # evaluate model
#	quiet || begin
#		println("\nTrue positives: ", countnz(bk), "/", length(be), ".")
#		println("Distances to true model:")
#		println("Chebyshev (L-Inf) = ", chebyshev(bk, be), ".")
#		println("Euclidean (L-2)   = ", euclidean(bk, be), ".")
#	end
#end





# TEST THE EXCHANGE ALGORITHM FOR LEAST SQUARES REGRESSION OVER PLINK BINARY DATA
# This subroutine runs the exchange algorithm on a GWAS dataset from the WTCCC.
#function test_exchangeleastsq_plink(n::Int, p::Int; x_path::ASCIIString = "/Users/kkeys/Dropbox/STAMPEED/filtered2.bed", y_path::ASCIIString = "/Users/kkeys/Dropbox/STAMPEED/filtered.trait", r::Int = 300, tol::Float64 = 1e-4, max_iter::Int = 100, quiet::Bool = false, extra::Int = 10, window::Int = 10) 
function test_exchangeleastsq_plink(x_path::ASCIIString = "/Users/kkeys/Downloads/wtccc_full/wtccc-n2k_chr1_clean.bed", xt_path::ASCIIString = "/Users/kkeys/Downloads/wtccc_full/wtccc-n2k_chr1_clean_t.bed", y_path::ASCIIString = "/Users/kkeys/Downloads/withnoiselevelsd0_1/Y.100.1", b_path::ASCIIString = "/Users/kkeys/Downloads/withnoiselevelsd0_1/causal.100.1", r::Int = 100, tol::Float64 = 1e-4, max_iter::Int = 100, quiet::Bool = false, extra::Int = 10, window::Int = 10) 

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
#	x = read_bedfile(x_path)
	x = BEDFile(x_path, xt_path)
	const (n,p) = size(x)

    # load response vector
    y = readdlm(y_path)
	y = vec(y) # need y to be 1D
	y = (y - mean(y)) / std(y)
	const y = convert(SharedArray{Float64,1}, y)

	const means   = PLINK.mean(x) 
	const invstds = PLINK.invstd(x, y=means)

    # load model
    B = readdlm(b_path)

    # need indices for true model
    # add 1 since Julia uses 1-indexing
	const bidx = convert(Array{Int,1}, B[:,2]) + 1

    # need components of true model
	const be = convert(Array{Float64,1}, B[:,3])

#	quiet || begin 
#		println("True Model:\n\tIndex\tValue")
#		for i = 1:length(bidx)
#			println("\t", bidx[i], "\t", be[i])
#		end
#	end

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
	inner = Dict{Int,DenseArray{Float64,1}}()

	# how long did the allocation take?
	alloc_time = toq()
	quiet || println("Temporary arrays allocated in ", alloc_time, " seconds.")

	# precompute sum of squares for each column of x
	quiet || println("\nPrecomputing squared Euclidean norms of matrix columns...")
	tic()
	const nrmsq = vec(sumsq(x, shared=true, means=means, invstds=invstds))
	norm_time = toq()
	quiet || println("Column norms took ", norm_time, " seconds to compute")

	# declare any return values
	iter::Int = 0

	# print time spent on parameter initialization
	parameter_time = norm_time + alloc_time + file_time + B_time
	quiet || println("Parameter initialization took ", parameter_time, " seconds.")

    # run exchange 
    quiet || println("\nRunning exchange algorithm...")

	# reset timer and compute path
	tic()
	@time begin
		for i = 2:(r+extra)
			exchange_leastsq!(b, x, y, perm, i, p, n=n, inner=inner, max_iter=max_iter, quiet=quiet, nrmsq=nrmsq, res=res, df=df, tempn=tempn, tempn2=tempn2, tempp=tempp, dotprods=dotprods, window=min(i,window), Xb=Xb, means=means, invstds=invstds, indices=indices)
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
#	myfolds = cv_get_folds(n,numfolds)
	myfolds = cv_get_folds(y,numfolds)

	# inform console that the folds are computed
	quiet || println("Folds computed.")

	# run cv_iht to crossvalidate dataset
	# be careful! mses will be a tuple if compute_model=true
	tic()
#	mses = cv_exlstsq(x,y,kend,numfolds, p=p, tol=tol, max_iter=max_iter, quiet=quiet, folds=myfolds, compute_model=compute_model) 
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

end # end module ExchangeLeastsq
