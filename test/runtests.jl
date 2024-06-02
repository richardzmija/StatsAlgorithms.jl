include("../src/StatsAlgorithms.jl")

using .StatsAlgorithms
using Plots

X = [4.0 2.0; 3.0 5.0; 6.0 1.0; 7.0 8.0]
X_pca = pca(X, 2)

p1 = scatter(X[:, 1], X[:, 2], title = "Original Data", xlabel = "Feature 1",
ylabel = "Feature 2", legend = false)

p2 = scatter(X_pca[:, 1], X_pca[:, 2], title = "PCA Transformed Data",
xlabel = "PC1", ylabel = "PC2", legend = false)

plot(p1, p2, layout=(1, 2))
