function SimulatePath(Initpoint::Vector{Float64},Pathlength::Int64,Pathnumber::Int64,disturb::Array{Float64,3},Weight::Vector{Float64})
    distribution = Categorical(Weight)
    #dnum=size(weight)[1]
    path = zero(Matrix{Float64}(undef, length(Initpoint), Pathlength * Pathnumber))
    path_labels = zero(Matrix{Int64}(undef, 1, Pathlength * Pathnumber))
    k = 1::Int64
    for l in 1:Pathnumber
        point = Initpoint
        label = 0
        for i in 1: Pathlength
            path[:, k] = point
            path_labels[:, k] .= label
            k = k+1
            label = rand(distribution)
            point = disturb[:,:,label ] * point
        end
    end
    return(path, path_labels)
end

function SimulatePath(Initpoint::Vector{Float64},Pathlength::Int64,Pathnumber::Int64,X::rcss)
    return(SimulatePath(Initpoint,Pathlength,Pathnumber,X.disturb,X.Weight))
end
