type ELSQVariables{T <: Float, V <: DenseVector}
    b        :: V
    nrmsq    :: V
    df       :: V
    dotprods :: V
    tempp    :: V
    r        :: V
    tempn    :: V
    tempn2   :: V
    perm     :: DenseVector{Int} 
    inner    :: Dict{Int, DenseVector{T}}

    ELSQVariables(b::DenseVector{T}, nrmsq::DenseVector{T}, df::DenseVector{T}, dotprods::DenseVector{T}, tempp::DenseVector{T}, r::DenseVector{T}, tempn::DenseVector{T}, tempn2::DenseVector{T}, perm::DenseVector{Int}, inner::Dict{Int, DenseVector{T}}) = new(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, perm, inner)
end

function ELSQVariables{T <: Float}(
    b        :: DenseVector{T},
    nrmsq    :: DenseVector{T},
    df       :: DenseVector{T},
    dotprods :: DenseVector{T},
    tempp    :: DenseVector{T},
    r        :: DenseVector{T},
    tempn    :: DenseVector{T},
    tempn2   :: DenseVector{T},
    perm     :: DenseVector{Int}, 
    inner    :: Dict{Int, DenseVector{T}}
)
    ELSQVariables{T, typeof(b)}(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, perm, inner)
end

function ELSQVariables{T <: Float}(
    x :: DenseMatrix{T},
    y :: DenseVector{T}
)
    # dimensions of arrays
    n,p = size(x)

    # form arrays
    b        = zeros(T, p) 
    nrmsq    = vec(sumabs2(x,1))
    df       = zeros(T, p)
    dotprods = zeros(T, p)
    tempp    = zeros(T, p)
    r        = zeros(T, n)
    tempn    = zeros(T, n)
    tempn2   = zeros(T, n)
    perm     = collect(1:p)

    # form dictionary
    inner = Dict{Int, DenseVector{T}}()

    # return container object
    ELSQVariables{T, DenseVector{T}}(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, perm, inner)
end

immutable ELSQCrossvalidationResults{T <: Float}
    mses :: Vector{T}
    b    :: Vector{T}
    bidx :: Vector{Int}
    k    :: Int
end

function ELSQCrossvalidationResults{T <: Float}(
    mses :: Vector{T},
    k    :: Int
)
    b    = zeros(T, 1)
    bidx = zeros(Int, 1)
    ELSQCrossvalidationResults{T}(mses, b, bidx, k)
end

function Base.display(x::ELSQCrossvalidationResults)
    println("Exchange Algorithm Crossvalidation Results:")
    println("Best model size is $k predictors")
    println("\tPredictor\tValue\tMSE")
    for i in eachindex(x.bidx)
        println("\t", x.bidx[i], "\t", x.b[i], "\t", x.mses[i])
    end
end

function check_finiteness{T <: Float}(x::T)
    isnan(x) && throw(error("Objective function is NaN, aborting..."))
    isinf(x) && throw(error("Objective function is Inf, aborting..."))
end

function print_descent_error{T <: Float}(iter::Int, loss::T, next_loss::T)
    print_with_color(:red, "\nExchange algorithm fails to descend!\n")
    print_with_color(:red, "Iteration: $(iter)\n")
    print_with_color(:red, "Current Objective: $(loss)\n")
    print_with_color(:red, "Next Objective: $(next_loss)\n")
    print_with_color(:red, "Difference in objectives: $(abs(next_loss - loss))\n")
    throw(error("Descent failure!"))
end

function print_maxiter{T <: Float}(max_iter::Int, loss::T)
    print_with_color(:red, "Exchange algorithm has hit maximum iterations $(max_iter)!\n")
    print_with_color(:red, "Return value may be incorrect\n")
    print_with_color(:red, "Current Loss: $(loss)\n")
end 

function errorcheck{T <: Float}(
    x        :: DenseMatrix{T},
    y        :: DenseVector{T},
#    k        :: Int,
    tol      :: T,
    max_iter :: Int,
    window   :: Int,
    p        :: Int = size(x,2)
)
#    0 <= k <= p           || throw(ArgumentError("Value of r must be nonnegative and cannot exceed length(bvec)"))
    tol >= eps()          || throw(ArgumentError("Global tolerance must exceed machine precision"))
    max_iter >= 1         || throw(ArgumentError("Maximum number of iterations must exceed 1"))
#    0 <= window <= k      || throw(ArgumentError("Value of selection window must be nonnegative and cannot exceed r"))
    return nothing
end

function print_cv_results{T <: Float}(errors::DenseVector{T}, path::DenseVector{Int}, k::Int)
    println("\n\nCrossvalidation Results:")
    println("k\tMSE")
    for i = 1:length(errors)
        println(path[i], "\t", errors[i])
    end
    println("\nThe lowest MSE is achieved at k = ", k)
end
