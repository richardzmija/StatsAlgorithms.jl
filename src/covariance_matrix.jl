module CovarianceMatrix

using ..Standardization: standardize, mean

export covariance_matrix

function covariance_matrix(X::AbstractMatrix{T};
        standardize_matrix::Bool = true) where T <: AbstractFloat

    n = size(X, 1)  # number of rows

    if standardize_matrix
        X = standardize(X)
        return (X' * X) / T(n - 1)
    end

    X_centered = X .- mean(X)
    (X_centered' * X_centered) / T(n - 1)
end

end
