module StatsAlgorithms

include("standardization.jl")
include("covariance_matrix.jl")
include("eigen_decomposition.jl")
include("pca.jl")

using .PCA

export pca

end
