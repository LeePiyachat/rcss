function PolicyRun(Evalue::Array{Float64},X::rcss,Initpoint::Vector{Float64},Initposition::Int64)
    tnum            = size(Evalue)[4]+1
    Evalues         = zero(Array{Float64}(undef,X.pnum))
    container       = zero(Array{Float64}(undef,X.pnum,X.anum))
    policy          = zero(Array{Int64}(undef,X.pnum, tnum-1))
    Pathsimulation  = SimulatePath(Initpoint,tnum,1,X)
    states          = Pathsimulation[1]
    Index_va        = X.nnum[1]
    #
    t = 1
    # determine control policy
    while t < tnum
        for p in 1:X.pnum
            Evalues[p]= Get_val(Evalue[:,:,p,t],states[:,t],X,Index_va)
        end
        for a in 1:X.anum
            container[:, a]= X.control[:,:,a] * Evalues
        end
        for p in 1:X.pnum
            for a in 1:X.anum
                container[p,a]+= X.Reward(t,states[:,t],p,a,X.mp, "scalar")
            end
        end
        for p in 1:X.pnum
            policy[p, t] = argmax(container[p,:])
        end
        t = t+1
    end

    #Calculate positions and actiions trajectory
    distributions = Array{Categorical}(undef,X.pnum, X.anum)
    for p in 1:X.pnum
        for a in 1:X.anum
            distributions[p, a] =   Categorical(X.control[p,:,a])
        end
    end
    positions   = zero(Vector{Int64}(undef,tnum))
    actions     = zero(Vector{Int64}(undef,tnum-1))
    positions[1]= Initposition

    t = 1
    while (t<tnum)
        actions[t] = policy[positions[t], t]
        positions[t+1] = rand(distributions[positions[t], actions[t]])
        t = t+1
    end
    result = Dict{String,Array{Real}}()
    result["Policy"] = policy
    result["States"] = states
    result["Positions"] = positions
    result["Actions"] = actions
    return(result)
end
