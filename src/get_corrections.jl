function get_corrections(value::Array{Float64},evalue::Array{Float64},x::rcss,z::Array{Float64},z_labels::Array{Int64},t::Int64,index_ph::Int64,index_va::Int64)
    output  =   zero(Vector{Float64}(undef,x.pnum))
    if index_ph == 0
        for pp in 1:x.pnum
            s = 0
            for k in 1:x.dnum
                s += x.weight[k] * get_val(value[:,:,pp,t+1],x.disturb[:,:,k] * z[:,t],x,index_va)
            end
            output[pp] = (s - get_val(value[:,:,pp,t+1],z[:,t+1],x,index_va))
        end
    elseif   0 < index_ph <= 50
        kweights    = zero(Array{Float64}(undef,index_ph))
        hosts,dists = knn(x.tree,z[:,t],index_ph,true)
        hosts       = transpose(hcat(hosts...))
        dists       = transpose(hcat(dists...))
        kweights[:] = kern(dists[:], x)
        for m in 1:index_ph
            result  = zero(Vector{Float64}(undef,x.pnum))
            point   = vec(x.grid[hosts[m,1],:])
            for pp in 1:x.pnum
                s = get_val(evalue[:,:,pp,t],point,x,index_va)
                #s=0 #  true martingale incremets, if access method in Bellman the same
                #for k in 1:x.dnum
                #s+=x.weight[k]*get_val(value[:,:, pp, t+1],  x.disturb[:,:, k]*point, x, index_va)
                #end
                result[pp] = (s - get_val(value[:,:,pp,t+1], x.disturb[:,:,z_labels[t+1]] * point,x,index_va))
            end
            output += kweights[m]*result
        end
    else  error("Wrong Neighbors number in martingale correction")
    end
    return(output)
end
