###################
### SUBROUTINES ###
###################

# PERFORM A*X + Y - B*Z 
#
# The silly name is based on BLAS axpy (A*X Plus Y), except that this function performs A*X Plus Y Minus B*Z.
# The idea behind axpymz!() is to perform the computation in one pass over the arrays. The output is the same as 
# > @devec y = y + a*x - b*z
function axpymbz!(j::Int, y::DenseArray{Float64,1}, a::Float64, x::DenseArray{Float64,1}, b::Float64, z::DenseArray{Float64,1})
	y[j] + a*x[j] - b*z[j]
end

function axpymbz!(y::Array{Float64,1}, a::Float64, x::Array{Float64,1}, b::Float64, z::Array{Float64,1}; p::Int = length(y)) 
	@inbounds for i = 1:p
		y[i] = axpymbz!(i, y, a, x, b, z) 
	end
end


#function axpymbz!(y::SharedArray{Float64,1}, a::Float64, x::SharedArray{Float64,1}, b::Float64, z::SharedArray{Float64,1}; p::Int = length(y)) 
#	@sync @inbounds @parallel for i = 1:p
#		y[i] = y[i] + a*x[i] - b*z[i]
#	end
#end


function myrange(q::SharedArray{Float64,1})
    idx = indexpids(q)
    if idx == 0
        # This worker is not assigned a piece
        return 1:0, 1:0
    end
    nchunks = length(procs(q))
    splits = [int(round(s)) for s in linspace(0,length(q),nchunks+1)]
    return splits[idx]+1 : splits[idx+1]
end

function axpymbz_shared_chunk!(y::SharedArray{Float64,1}, a::Float64, x::SharedArray{Float64,1}, b::Float64, z::SharedArray{Float64,1}, irange::UnitRange{Int})
    @inbounds for i in irange
        y[i] = axpymbz!(i,y,a,x,b,z)
    end
end

axpymbz_shared!(y::SharedArray{Float64,1}, a::Float64, x::SharedArray{Float64,1}, b::Float64, z::SharedArray{Float64,1}) = axpymbz_shared_chunk!(y,a,x,b,z,myrange(y))

function axpymbz!(y::SharedArray{Float64,1}, a::Float64, x::SharedArray{Float64,1}, b::Float64, z::SharedArray{Float64,1}) 
    @sync begin
        for p in procs(y)
            @async remotecall_wait(p, axpymbz_shared!, y, a, x, b, z)
        end
    end
end

#####################
### MAIN FUNCTION ###
#####################


# EXCHANGE ALGORITHM FOR L0-PENALIZED LEAST SQUARES REGRESSION 
# 
# This function minimizes the residual sum of squares
# 
# RSS = 0.5*|| Y - XB ||_2^2
#
# subject to beta having no more than r nonzero components. The function will compute a B for a given value of r.
# For optimal accuracy, this function should be run for multiple values of r over a path.
# For optimal performance for regularization path computations, reuse the arguments "bvec", "perm", and "inner".
#
# Arguments:
# -- bvec is the p-dimensional warm-start for the iterate
# -- X is the n x p statistical design matrix
# -- Y is the n-dimensional response vector
# -- perm is a p-dimensional array of integers that sort beta in descending order by magnitude
# -- r is the desired number of nonzero components in beta
# 
# Optional Arguments:
# -- "inner" is Dict for storing inner products. We fill inner dynamically as needed instead of computing X'X.
# -- n and p are the dimensions of X; the former actually to length(Y) while the latter defaults to size(X,2).
# -- nrmsq is the vector to store the squared norms of the columns of X. Defaults to vec(sumabs2(X,1))
# -- df is the temporary array to store the gradient. Defaults to zeros(p)
# -- dotprods is the temporary array to store the current column of dot products from Dict "inner". Defaults to zeros(p)
# -- tempp is a temporary array of length p. Defaults to zeros(p).
# -- res is the temporary array to store the vector of RESiduals. Defaults to zeros(n).
# -- tempn is a temporary array of length n. Defaults to zeros(n).
# -- tempn2 is another temporary array of length n. Defaults to copy(tempn).
# -- window is an Int variable to dictate the dimension of the search window for potentially exchanging predictors. 
#    Defaults to r (potentially exchange all current predictors). Decreasing this quantity tells the algorithm to search through 
#    fewer current active predictors, which can decrease compute time but can also degrade model recovery performance. 
# -- max_iter is the maximum permissible number of iterations. Defaults to 100.
#    Defaults to an empty dict with typeasserts Int64 for the keys and Array{Float64,1} for the values.
# -- tol is the convergence tolerance. Defaults to 1e-6.
# -- quiet is a boolean to control output. Defaults to false (full output).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
@compat function exchange_leastsq!(
	bvec     :: DenseArray{Float64,1}, 
	X        :: DenseArray{Float64,2}, 
	Y        :: DenseArray{Float64,1}, 
	perm     :: DenseArray{Int,1}, 
	r        :: Int; 
	inner    :: Dict{Int, DenseArray{Float64,1}} = Dict{Int,DenseArray{Float64,1}}(), 
	n        :: Int = length(Y), 
	p        :: Int = size(X,2), 
	nrmsq    :: DenseArray{Float64,1} = vec(sumabs2(X,1)), 
	df       :: DenseArray{Float64,1} = zeros(Float64, p), 
	dotprods :: DenseArray{Float64,1} = zeros(Float64, p), 
	tempp    :: DenseArray{Float64,1} = zeros(Float64, p), 
	res      :: DenseArray{Float64,1} = zeros(Float64, n), 
	tempn    :: DenseArray{Float64,1} = zeros(Float64, n), 
	tempn2   :: DenseArray{Float64,1} = zeros(Float64, n), 
	window   :: Int     = r,
	max_iter :: Int     = 10000, 
	tol      :: Float64 = 1e-6, 
	quiet    :: Bool    = false
)
	# error checking
	n == size(X,1)        || throw(DimensionMismatch("length(Y) != size(X,1)"))
	n == length(tempn)    || throw(DimensionMismatch("length(Y) != length(tempn)"))
	n == length(tempn2)   || throw(DimensionMismatch("length(Y) != length(tempn2)"))
	n == length(res)      || throw(DimensionMismatch("length(Y) != length(res)"))
	p == length(bvec)     || throw(DimensionMismatch("length(bvec) != length(bvec)"))
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
	rss     = 0.0	# residual sum of squares || Y - XB ||^2
	old_rss = Inf	# previous residual sum of squares 

	# obtain top r components of bvec in magnitude
	selectpermk!(perm,bvec, r, p=p)

	# compute partial residuals based on top r components of perm vector
	RegressionTools.update_partial_residuals!(res, Y, X, perm, bvec, r, n=n, p=p)

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
			betal = bvec[l]
			update_col!(tempn, X, l, n=n, p=p, a=1.0)	# tempn now holds X[:,l]

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
			update_col!(tempn2, X, m, n=n, p=p, a=1.0)	# tempn2 now holds X[:,m]
			axpymbz!(res, betal, tempn, adb, tempn2, p=n)

			# if necessary, compute inner product of current predictor against all other predictors
			# save in our Dict for future reference
			# compare in performance to
			# > tempp = get!(inner, m, BLAS.gemv('T', 1.0, X, tempn2))
			if !haskey(inner, m)
				inner[m] = BLAS.gemv('T', 1.0, X, tempn2)
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

		# test for descent failure 
		# if no descent failure, then test for convergence
		# if not converged, then save RSS and continue
		ascent    = rss > old_rss + tol
		converged = abs(old_rss - rss) / abs(old_rss + 1) < tol 

		ascent && throw(error("Descent error detected at iteration $(iter)!\nOld RSS: $(old_rss)\nRSS: $(rss)")) 
		(converged || ascent) && return bvec
		old_rss = rss
		isnan(rss) && throw(error("Objective function is NaN!"))
		isinf(rss) && throw(error("Objective function is Inf!"))

	end # end outer iteration loop

	# at this point, maximum iterations reached
	# warn and return bvec
	throw(error("Maximum iterations $(max_iter) reached! Return value may not be correct.\n"))
	return bvec

end # end exchange_leastsq
