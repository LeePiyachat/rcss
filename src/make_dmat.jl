function make_dmat(x::rcss)
    print(" Making dmat \n")
    dmats       =   zero(Array{Float64}(undef, x.gnum,x.gnum,x.rnum+1))
    kweights    =   zero(Array{Float64}(undef, x.gnum,x.nnum[1]))

    for k in 1:x.dnum
        hosts, dists = knn(x.tree,x.disturb[:,:,k] * transpose(x.grid),x.nnum[1],true)
        hosts = transpose(hcat(hosts...))
        dists = transpose(hcat(dists...))
        for i in 1:x.gnum
            kweights[i,:] = kern(dists[i,:], x)
        end
        for i in 1:x.gnum
            for m in 1:x.nnum[1]
                dmats[i,hosts[i,m],x.rnum+1] += kweights[i,m] * x.weight[k]
            end
            for l in 1:x.rnum
                for m in 1:x.nnum[1]
                    dmats[i,hosts[i,m],l] +=  kweights[i,m] * x.modif[l,k] * x.weight[k]
                end
            end
        end
    end
    println("Finished dmat \n")
    return(dmats)
end
