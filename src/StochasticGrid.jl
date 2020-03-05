function stochasticgrid(gridsize::Int64,initpoint::Vector{Float64},length::Int64,pathnumber::Int64,x::rcss)
    return(stochasticgrid(gridsize,initpoint,length,pathnumber,x.disturb,x.weight))
end
############################################################################
function stochasticgrid(gridsize::Int64, initpoint::Vector{Float64},length::Int64, pathnumber::Int64, disturb::Array{Float64,3}, weight::Vector{Float64})
    path = simulatepath(initpoint,length::Int64,pathnumber::Int64,disturb,weight)[1]
    Y = kmeans(path, gridsize; maxiter = 200, display=:iter)
    return(transpose(Y.centers))
end
