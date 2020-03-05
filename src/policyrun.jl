function policyrun(evalue::Array{Float64},x::rcss,initpoint::Vector{Float64},initposition::Int64)
    tnum            = size(evalue)[4]+1
    evalues         = zero(Array{Float64}(undef,x.pnum))
    container       = zero(Array{Float64}(undef,x.pnum,x.anum))
    policy          = zero(Array{Int64}(undef,x.pnum, tnum-1))
    pathsimulation  = simulatepath(initpoint,tnum,1,x)
    states          = pathsimulation[1]
    index_va        = x.nnum[1]
    #
    t = 1
    # determine control policy
    while t < tnum
        for p in 1:x.pnum
            evalues[p]= get_val(evalue[:,:,p,t],states[:,t],x,index_va)
        end
        for a in 1:x.anum
            container[:, a]= x.positioncontrol[:,:,a] * evalues
        end
        for p in 1:x.pnum
            for a in 1:x.anum
                container[p,a]+= x.reward(t,states[:,t],p,a,x.mp, "scalar")
            end
        end
        for p in 1:x.pnum
            policy[p, t] = argmax(container[p,:])
        end
        t = t+1
    end

    #Calculate positions and actiions trajectory
    distributions = Array{Categorical}(undef,x.pnum, x.anum)
    for p in 1:x.pnum
        for a in 1:x.anum
            distributions[p, a] =   Categorical(x.positioncontrol[p,:,a])
        end
    end
    positions   = zero(Vector{Int64}(undef,tnum))
    actions     = zero(Vector{Int64}(undef,tnum-1))
    positions[1]= initposition

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
