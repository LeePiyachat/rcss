function get_val(field::Array{Float64},argument::Vector{Float64},x::rcss,index_va::Int64)
    if index_va == 0
        outcome = sum(maximum(field[:,:] * argument))
    elseif -50 <= index_va < 0
        hosts, dists = knn(x.tree,argument,-index_va,true)
        hosts   = transpose(hcat(hosts...))
        outcome = sum(maximum(field[hosts[:,1],:] * argument))
    elseif 0 < index_va <= 50
        kweights    =   zero(Array{Float64}(undef,index_va))
        hosts, dists=   knn(x.tree,argument,index_va,true)
        hosts       =   transpose(hcat(hosts...))
        dists       =   transpose(hcat(dists...))
        kweights[:] =   kern(dists[:],x)
        result      =   zero(Vector{Float64}(undef,x.snum))
        for m in 1:index_va
            result += kweights[m] * field[hosts[m,1],:]
        end
        outcome     =   sum(result.* argument)
    else error("Wrong Neighbors number in value access")
    end
    return(outcome)
end

