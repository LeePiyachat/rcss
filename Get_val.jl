function Get_val(Field::Array{Float64},Argument::Vector{Float64},X::rcss,Index_va::Int64)
    if Index_va == 0
        outcome = sum(maximum(Field[:,:] * Argument))
    elseif -50 <= Index_va < 0
        hosts, dists = knn(X.tree,Argument,-Index_va,true)
        hosts   = transpose(hcat(hosts...))
        outcome = sum(maximum(Field[hosts[:,1],:] * Argument))
    elseif 0 < Index_va <= 50
        kweights    =   zero(Array{Float64}(undef,Index_va))
        hosts, dists=   knn(X.tree,Argument,Index_va,true)
        hosts       =   transpose(hcat(hosts...))
        dists       =   transpose(hcat(dists...))
        kweights[:] =   Kern(dists[:],X)
        result      =   zero(Vector{Float64}(undef,X.snum))
        for m in 1:Index_va
            result += kweights[m] * Field[hosts[m,1],:]
        end
        outcome     =   sum(result.* Argument)
    else error("Wrong Neighbors number in value access")
    end
    return(outcome)
end
