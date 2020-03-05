function expectedslow(value::Matrix{Float64},x::rcss)
    #result = zero(Matrix{Float64}(x.gnum,x.snum))
    result = zero(Matrix{Float64}(undef,x.gnum,x.snum))
    if  x.nnum[1]== 0
        for k in 1:x.dnum
            subgradients = value * x.disturb[:,:,k] * x.weight[k]
            result += subgradients[indenlarge(subgradients,x),:]
        end
    else
        if (x.nnum[1] > 0)
            nnum = x.nnum[1]
            kweights = zero(Array{Float64}(undef,x.gnum,nnum))
        else
            nnum =- x.nnum[1]
            container = zero(Vector{Float64}(undef,nnum))
        end
        for k in 1:x.dnum
            subgradients = transpose(value * x.disturb[:,:,k] * x.weight[k])
            hosts,dists= knn(x.tree,x.disturb[:,:,k] * transpose(x.grid),nnum,true)
            hosts = transpose(hcat(hosts...))
            dists = transpose(hcat(dists...))
            if (x.nnum[1] > 0)
                for i in 1:x.gnum
                    kweights[i,:] = kern(dists[i,:],x)
                end
                for i in 1:x.gnum
                    for m in nnum
                        result[i,:] += kweights[i,m] * subgradients[:, hosts[i,m]]
                    end
                end
            else
                for i in 1:x.gnum
                    s = subgradients[:, hosts[i,:]]
                    #result[i,:]+= S[:, indmax(transpose(S) * x.grid[i,:])]
                    result[i,:]+= s[:, argmax(transpose(s) * x.grid[i,:])]
                end
            end
        end
    end
    return(result)
end
