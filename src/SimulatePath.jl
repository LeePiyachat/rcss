function simulatepath(initpoint::Vector{Float64},pathlength::Int64,pathnumber::Int64,disturb::Array{Float64,3},weight::Vector{Float64})
    distribution = Categorical(weight)
    #dnum=size(weight)[1]
    path = zero(Matrix{Float64}(undef,length(initpoint),pathlength * pathnumber))
    path_labels = zero(Matrix{Int64}(undef,1, pathlength * pathnumber))
    k = 1::Int64
    for l in 1:pathnumber
        point = initpoint
        label = 0
        for i in 1: pathlength
            path[:, k] = point
            path_labels[:, k] .= label
            k = k+1
            label = rand(distribution)
            point = disturb[:,:,label ] * point
        end
    end
    return(path, path_labels)
end

function simulatepath(initpoint::Vector{Float64},pathlength::Int64,pathnumber::Int64,x::rcss)
    return(simulatepath(initpoint,pathlength,pathnumber,x.disturb,x.weight))
end
