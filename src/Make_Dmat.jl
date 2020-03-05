function Make_Dmat(X::rcss)
    print(" Making Dmat \n")
    Dmats       =   zero(Array{Float64}(undef, X.gnum,X.gnum,X.rnum+1))
    kweights    =   zero(Array{Float64}(undef, X.gnum,X.nnum[1]))

    for k in 1:X.dnum
        hosts, dists = knn(X.tree,X.disturb[:,:,k] * transpose(X.Grid),X.nnum[1],true)
        hosts = transpose(hcat(hosts...))
        dists = transpose(hcat(dists...))
        for i in 1:X.gnum
            kweights[i,:] = Kern(dists[i,:], X)
        end
        for i in 1:X.gnum
            for m in 1:X.nnum[1]
                Dmats[i,hosts[i,m],X.rnum+1] += kweights[i,m] * X.Weight[k]
            end
            for l in 1:X.rnum
                for m in 1:X.nnum[1]
                    Dmats[i,hosts[i,m],l] +=  kweights[i,m] * X.Modif[l,k] * X.Weight[k]
                end
            end
        end
    end
    println("Finished Dmat \n")
    return(Dmats)
end
