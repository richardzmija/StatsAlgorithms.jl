module PCA

using ..Standardization: standardize
using ..CovarianceMatrix: covariance_matrix
using ..EigenDecomposition: eigen_decomposition

export pca

"""
    pca(X::AbstractMatrix{T}, k::Int, num_iter::Int = 100) where T <: AbstractFloat

Perform Principal Component Analysis (PCA) on the data matrix `X` and compute the
top `k` principal components.

# Arguments
- `X::AbstractMatrix{T}`: A matrix of floating-point numbers.
- `k::Int`: Number of principal components to retain.
- `num_iter::Int`: Number of iterations for precision of eigenvalues computation.

# Returns
- `AbstractMatrix{T}`: Matrix containing the top `k` principal components.
"""
function pca(X::AbstractMatrix{T}, k::Int, num_iter::Int = 100) where T <: AbstractFloat
    X_standardized = standardize(X)
    cov_matrix = covariance_matrix(X_standardized; standardize_matrix = false)
    Λ, V = eigen_decomposition(cov_matrix, num_iter)

    idx = sortperm(Λ, rev=true)
    Λ = Λ[idx]  # Get eigenvalues sorted is descending order
    V = V[:, idx]  # Sort the columns representing the eigenvectors

    # Eigenvectors of a symmetric matrix are orthogonal and eigen_decomposition
    # returns normalized eigenvectors which means that a list of eigenvectors
    # forms an orthonormal basis.
    # Construct a projection matrix
    W = V[:, 1:k]

    # Transform the original data into the new coordinate system
    # defined by the top k principal components.
    X_pca = X_standardized * W
    X_pca
end

end
