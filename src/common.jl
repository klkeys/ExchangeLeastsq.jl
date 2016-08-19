type ELSQVariables{T <: Float, V <: DenseVector}
    b        :: V
    nrmsq    :: Vector{T}
    df       :: V
    dotprods :: Vector{T}
#    tempp    :: Vector{T} 
    tempp    :: V 
    r        :: V
#    tempn    :: Vector{T} 
#    tempn2   :: Vector{T}
    tempn    :: V 
    tempn2   :: V
    xb       :: V
    perm     :: DenseVector{Int} 
    inner    :: Dict{Int, Vector{T}}
    mask_n   :: DenseVector{Int}
    idx      :: BitArray{1}

    ELSQVariables(b::DenseVector{T}, nrmsq::DenseVector{T}, df::DenseVector{T}, dotprods::DenseVector{T}, tempp::DenseVector{T}, r::DenseVector{T}, tempn::DenseVector{T}, tempn2::DenseVector{T}, xb::DenseVector{T}, perm::DenseVector{Int}, inner::Dict{Int, Vector{T}}, mask_n::DenseVector{Int}, idx::BitArray{1}) = new(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, xb, perm, inner, mask_n, idx)
end

function ELSQVariables{T <: Float}(
    b        :: DenseVector{T},
    nrmsq    :: Vector{T},
    df       :: DenseVector{T},
    dotprods :: Vector{T},
#    tempp    :: Vector{T},
    tempp    :: DenseVector{T},
    r        :: DenseVector{T},
#    tempn    :: Vector{T},
#    tempn2   :: Vector{T},
    tempn    :: DenseVector{T},
    tempn2   :: DenseVector{T},
    xb       :: DenseVector{T},
    perm     :: DenseVector{Int}, 
    inner    :: Dict{Int, Vector{T}},
    mask_n   :: DenseVector{Int},
    idx      :: BitArray{1}
)
    ELSQVariables{T, typeof(b)}(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, xb, perm, inner, mask_n, idx)
end

function ELSQVariables{T <: Float}(
    x :: DenseMatrix{T},
    y :: DenseVector{T}
)
    # dimensions of arrays
    n,p = size(x)

    # form arrays
    b        = zeros(T, p) 
    nrmsq    = vec(sumabs2(x,1)) :: Vector{T}
    df       = zeros(T, p)
    dotprods = zeros(T, p)
    tempp    = zeros(T, p)
    r        = zeros(T, n)
    tempn    = zeros(T, n)
    tempn2   = zeros(T, n)
    xb       = zeros(T, n) 
    perm     = collect(1:p)
#    mask_n   = zeros(Int, n)
    mask_n   = ones(Int, n)
    idx      = falses(p)

    # form dictionary
#    inner = Dict{Int, DenseVector{T}}()
    inner = Dict{Int, Vector{T}}()

    # return container object
    ELSQVariables{eltype(y), typeof(y)}(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, xb, perm, inner, mask_n, idx)
end


function ELSQVariables{T <: Float}(
    x :: BEDFile{T},
    y :: SharedVector{T},
#    z :: DenseVector{Int}
)
    # dimensions of arrays
    n,p = size(x)

    # process ids?
    pids = procs(x)

    # form arrays
    b        = SharedArray(T, (p,), pids=pids) :: typeof(y)
    nrmsq    = (length(y) - 1) * ones(T, p) 
    df       = SharedArray(T, (p,), pids=pids) :: typeof(y)
    dotprods = zeros(T, p) 
#    tempp    = zeros(T, p) 
    tempp    = SharedArray(T, (p,), pids=pids) :: typeof(y)
    r        = SharedArray(T, (n,), pids=pids) :: typeof(y)
#    tempn    = zeros(T, n) 
#    tempn2   = zeros(T, n) 
    tempn    = SharedArray(T, (n,), pids=pids) :: typeof(y)
    tempn2   = SharedArray(T, (n,), pids=pids) :: typeof(y)
    xb       = SharedArray(T, (n,), pids=pids) :: typeof(y)
    perm     = collect(1:p)
#    mask_n   = zeros(Int, n)
    mask_n   = ones(Int, n)
    idx      = falses(p)

    # form dictionary
    inner = Dict{Int, Vector{T}}()

    # return container object
    ELSQVariables{eltype(y), typeof(y)}(b, nrmsq, df, dotprods, tempp, r, tempn, tempn2, xb, perm, inner, mask_n, idx)
end

function ELSQVariables{T <: Float}(
    x :: BEDFile{T},
    y :: SharedVector{T},
    z :: DenseVector{Int} # <-- this should be the bitmask
)
    w = ELSQVariables(x,y)
    copy!(w.mask_n, z)
    return w
end

immutable ELSQCrossvalidationResults{T <: Float}
    mses :: Vector{T}
    b    :: Vector{T}
    bidx :: Vector{Int}
    k    :: Int
    path :: Vector{Int}
    bids :: Vector{UTF8String}
end

#function Base.display(x::ELSQCrossvalidationResults)
#    println("Crossvalidation Results:")
#    println("Best model size is $k predictors")
#    println("\tPredictor\tValue\tMSE")
#    for i in eachindex(x.bidx)
#        println("\t", x.bidx[i], "\t", x.b[i], "\t", x.mses[i])
#    end
#end

# constructor for when bids are not available
# simply makes vector of "V$i" where $i are drawn from bidx
function ELSQCrossvalidationResults{T <: Float}(
    mses :: Vector{T},
    b    :: Vector{T},
    bidx :: Vector{Int},
    k    :: Int,
    path :: Vector{Int},
)  
    bids = convert(Vector{UTF8String}, ["V" * "$i" for i in bidx]) :: Vector{UTF8String}
    ELSQCrossvalidationResults{eltype(mses)}(mses, b, bidx, k, path, bids)
end

# function to view an ELSQCrossvalidationResults object
function Base.show(io::IO, x::ELSQCrossvalidationResults)
    println(io, "Crossvalidation results:") 
    println(io, "Minimum MSE ", minimum(x.mses), " occurs at k = $(x.k).")
    println(io, "Best model β has the following nonzero coefficients:")
    println(io, DataFrame(Predictor=x.bidx, Name=x.bids, Estimated_β=x.b))
    return nothing
end

function Gadfly.plot(x::ELSQCrossvalidationResults)
    df = DataFrame(ModelSize=x.path, MSE=x.mses)
    plot(df, x="ModelSize", y="MSE", xintercept=[x.k], Geom.line, Geom.vline(color=colorant"red"), Guide.xlabel("Model size"), Guide.ylabel("MSE"), Guide.title("MSE versus model size"))
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

function errorcheck{T <: Float}(
    x        :: BEDFile{T},
    y        :: SharedVector{T},
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
