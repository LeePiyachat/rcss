function Get_ph(p::Int64,a::Int64,Value::Array{Float64},Evalue::Array{Float64},X::rcss,z::Array{Float64},z_labels::Array{Int64},t::Int64,Index_ph::Int64,Index_va::Int64)
    if Index_ph == 0
        output=0.0
        for pp in 1:X.pnum
            s = 0
            for k in 1:X.dnum
                s += X.Weight[k] * Get_val(Value[:,:, pp,t+1],X.disturb[:,:,k] * z[:, t],X,Index_va)
            end
            output +=  X.control[p,pp,a]*(s - Get_val(Value[:,:, pp,t+1], z[:,t+1], X,Index_va))
        end
    elseif 0 < Index_ph <= 50
        kweights = zero(Array{Float64}(Index_ph))
        hosts,dists = knn(X.tree,z[:, t],Index_ph,true)
        hosts = transpose(hcat(hosts...))
        dists = transpose(hcat(dists...))
        kweights[:] = kern(dists[:],X)
        output = 0.0
        for m in 1:Index_ph
            result = 0.0
            point = vec(X.Grid[hosts[m,1],:])
            for pp in 1:X.pnum
                s   = Get_val(Evalue[:,:, pp,t],point,X, Index_va)
                #s=0 #true martingale incremets, if access method in Bellman the same
                #for k in 1:X.dnum
                #s+=X.weight[k]*get_val(value[:,:, pp, t+1],  X.disturb[:,:, k]*point, X, index_va)
                #end
                result +=  X.control[p,pp,a] * (s-Get_val(Value[:,:,pp,t+1],X.disturb[:,:,z_labels[t+1]] * point,X,Index_va))
            end
            output += kweights[m]*result
        end
    else error("Wrong Neighbors number in Martingale correction")
    end
    return(output)
end
