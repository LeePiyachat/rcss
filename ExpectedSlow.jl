function ExpectedSlow(Value::Matrix{Float64},X::rcss)
    result = zero(Matrix{Float64}(X.gnum,X.snum))
    if  X.nnum[1]== 0
        for k in 1:X.dnum
            subgradients = Value * X.disturb[:,:,k] * X.Weight[k]
            result += subgradients[IndEnlarge(subgradients,X),:]
        end
    else
        if (X.nnum[1] > 0)
            nnum = X.nnum[1]
            kweights = zero(Array{Float64}(X.gnum,nnum))
        else
            nnum =- X.nnum[1]
            container = zero(Vector{Float64}(nnum))
        end
        for k in 1:X.dnum
            subgradients = transpose(Value * X.disturb[:,:,k] * X.Weight[k])
            hosts,dists= knn(X.tree,X.disturb[:,:,k] * transpose(X.Grid),nnum,true)
            hosts = transpose(hcat(hosts...))
            dists = transpose(hcat(dists...))
            if (X.nnum[1] > 0)
                for i in 1:X.gnum
                    kweights[i,:] = kern(dists[i,:], X)
                end
                for i in 1:X.gnum
                    for m in nnum
                        result[i,:] += kweights[i,m] * subgradients[:, hosts[i,m]]
                    end
                end
            else
                for i in 1:X.gnum
                    S = subgradients[:, hosts[i,:]]
                    result[i,:]+= S[:, indmax(transpose(S) * X.Grid[i,:])]
                end
            end
        end
    end
    return(result)
end
