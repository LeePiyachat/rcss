function StochasticGrid(Gridsize::Int64,Initpoint::Vector{Float64},Length::Int64,Pathnumber::Int64,X::rcss)
    return(StochasticGrid(Gridsize,Initpoint,Length,Pathnumber,X.disturb,X.Weight))
end
############################################################################
function StochasticGrid(Gridsize::Int64, Initpoint::Vector{Float64},Length::Int64, Pathnumber::Int64, disturb::Array{Float64,3}, Weight::Vector{Float64})
    path = SimulatePath(Initpoint,Length::Int64,Pathnumber::Int64,disturb,Weight)[1]
    Y = kmeans(path, Gridsize; maxiter = 200, display=:iter)
    return(transpose(Y.centers))
end
