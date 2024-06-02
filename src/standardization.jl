module Standardization

export mean, std, standarize

"""
    mean(X::AbstractMatrix{T}) where T <: AbstractFloat

Calculate the sample mean of each column in the given matrix `X`.

# Arguments
- `X::AbstractMatrix{T}`: A matrix of floating-point numbers.

# Returns
- `AbstractMatrix{T}`: A row vector containing the sample mean of each column
in the input matrix `X`.

# Example
```julia
X = [4.0 2.0; 3.0 5.0; 6.0 1.0; 7.0 8.0]
mean_X = mean(X)
println(mean_X) # Output: [5.0 4.0]
```
"""
function mean(X::AbstractMatrix{T}) where T <: AbstractFloat
    n = size(X, 1)  # number of rows
    sum(X, dims=1) / T(n)
end

"""
    std(X::AbstractMatrix{T}, mean_X::AbstractMatrix{T}) where T <: AbstractFloat

Calculate the sample standard deviation of each column in the given matrix `X`.

# Arguments
- `X::AbstractMatrix{T}`: A matrix of floating-point numbers.
- `mean_X::AbstractMatrix{T}`: A row vector containing the sample mean of each column.

# Returns
- `AbstractMatrix{T}`: A row vector containing the sample standard deviation of each column.

# Example
```julia
X = [1.0 1.0; 1.0 -1.0]
std_X = std(X, mean(X))
println(std_X) # Output: [0.0, sqrt(2)]
```
"""
function std(X::AbstractMatrix{T},
        mean_X::AbstractMatrix{T}) where T <: AbstractFloat
        
    n = size(X, 1)  # number of rows
    deviations = (X .- mean_X) .^ 2
    variance = sum(deviations, dims=1) / T(n - 1)  # Bessel's correction
    sqrt.(variance)
end


"""
    standardize(X::AbstractMatrix{T}) where T <: AbstractFloat

Standardize the given matrix `X` by standardizing each column.
This process transforms the data to have zero mean and unit variance for
each column.

# Arguments
- `X::AbstractMatrix{T}`: A matrix of floating-point numbers.

# Returns
- `X::AbstractMatrix{T}`: A matrix where each column of the input matrix `X`
has been standardized.

# Example
```julia
X = [4.0 2.0; 3.0 5.0; 6.0 1.0; 7.0 8.0]
X_standardized = standardize(X)
println(X_standardized)
```
"""
function standardize(X::AbstractMatrix{T}) where T <: AbstractFloat
    mean_X = mean(X)
    std_X = std(X, mean_X)
    (X .- mean_X) ./ std_X
end

end