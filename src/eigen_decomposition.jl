module EigenDecomposition

function dot_product(x::AbstractVector{T}, y::AbstractVector{T}) where T <: AbstractFloat
    sum(x[i] * y[i] for i in eachindex(x))
end

function power_iteration(A::AbstractMatrix{T}, num_iter::Int) where T <: AbstractFloat

    n = size(A, 1)  # number of rows
    v = rand(T, n)  # create a random vector of length n
    v /= sqrt(sum(v .^ 2))  # normalize the vector
    λ = T(0)

    for _ in 1:num_iter
        v = A * v  # apply the linear transformation
        v /= sqrt(sum(v .^ 2))  # normalize the vector
        λ = dot_product(v, A * v)  # compute Rayleigh quotient
    end

    λ, v
end

function eigen_decomposition(A::AbstractMatrix{T}, num_iter::Int) where T <: AbstractFloat
    n = size(A, 1)  # number of rows
    Λ = zeros(T, n)  # storage for eigenvalues
    V = zeros(T, n, n)  # storage for eigenvectors
    A_copy = copy(A)  # copy the matrix to avoid modification of original

    for i in 1:n
        λ, v = power_iteration(A_copy, num_iter)
        Λ[i] = λ
        V[:, i] = v
        # perform rank-1 update to remove the contribution
        # of the found eigenvalue and eigenvector from the
        # matrix, allowing the next iteration to find
        # the next largest eigenvalue and its eigenvector
        A_copy -= λ * (v * v')
    end

    Λ, V
end

end
