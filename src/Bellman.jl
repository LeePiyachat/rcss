function Bellman(tnum::Int64,x::rcss,index_be::Int64)
    if  -50 <= index_be <= 0
        condexpectation = expectedslow
        println("Bellman with Slow method")
        if x.nnum[1]!= index_be
            print("\n")
            print("Changing index to ", - index_be , "neighbors \n" )
            x.nnum[:].= index_be
        end
    elseif  0 < index_be <= 50
        condexpectation = expectedfast
        println("Bellman with Fast method")
        if x.nnum[1]!= index_be
            print("\n")
            print("Recalculating matrix to ", index_be, " neighbors \n" )
            x.nnum[:] .= index_be
            x.dmat[:,:,:] = make_dmat(x)
        end
    else error("No option for Bellman recursion")
    end
    #fields for value and expected value function
    value = zero(Array{Float64}(undef,x.gnum,x.snum,x.pnum,tnum))
    evalue= zero(Array{Float64}(undef,x.gnum,x.snum,x.pnum,tnum-1))
    #initialize backward induction

    t = tnum
    container = zero(Matrix{Float64}(undef,x.anum, x.snum))

    for p in 1: x.pnum
        for i in 1: x.gnum
            value[i,:,p,t] = x.scrap(t, x.grid[i,:],p, x.mp)
        end
    end

    t = tnum
    while (t > 1)
        t = t-1
        print(t)
        for p in 1: x.pnum
            evalue[:,:,p,t]= condexpectation(value[:,:,p, t+1],x)
            print(".")
        end
        #
        for i in 1: x.gnum
            for p in 1:x.pnum
                container = zero(container)
                for a in 1:x.anum
                    for pp in 1:x.pnum
                        container[a,:] += x.positioncontrol[p,pp,a] * evalue[i,:,pp,t]
                    end
                    container[a,:]+= x.reward(t, x.grid[i,:], p, a, x.mp)
                end
                #amax = indmax(container * x.grid[i,:])
                amax = argmax(container * x.grid[i,:])
                value[i,:,p,t] = container[amax,:]
            end
        end
    end
    return(value, evalue)
end
############################################################################
function Bellman(tnum::Int64,x::rcss,index_be::Int64,scalar::String)
    if  -50 <= index_be <= 0
        condexpectation = expectedslow
        println("Bellman with slow method")
        if x.nnum[1]!= index_be
            print("\n")
            print("Changing index to ", - index_be, " neighbors \n")
            x.nnum[:] .= index_be
        end
    elseif  0 < index_be <=50
        condexpectation = expectedfast
        println("Bellman with fast method")
        if x.nnum[1] != index_be
            print("\n")
            print("Recalculating matrix to ", index_be, " neighbors \n")
            x.nnum[:] .= index_be
            x.dmat[:,:,:] = make_dmat(x)
        end
    else error("No such option for Bellman recursion")
    end
    #fields for value and expected value function
    value   =   zero(Array{Float64}(undef, x.gnum,x.snum,x.pnum,tnum))
    evalue  =   zero(Array{Float64}(undef, x.gnum,x.snum,x.pnum,tnum-1))
    #initialize backward induction
    t = tnum
    container   = zero(Array{Float64}(undef, x.pnum,x.anum))
    evalues     = zero(Vector{Float64}(undef, x.pnum))

    for p in 1: x.pnum
        for i in 1: x.gnum
            value[i,:,p, t]= x.scrap(t, x.grid[i,:],p,x.mp)
        end
    end

    t = tnum
    while (t > 1)
        t = t - 1
        print(t)
        for p in 1: x.pnum
            evalue[:,:,p,t]= condexpectation(value[:,:,p, t+1],x)
            print(".")
        end

        for i in 1: x.gnum
            for pp in 1:x.pnum
                evalues[pp] = sum(evalue[i,:,pp, t].* x.grid[i,:])
            end
            for a in 1:x.anum
                container[:,a] = x.positioncontrol[:,:,a] * evalues
            end
            for p in 1:x.pnum
                for a in 1:x.anum
                    container[p,a] += x.reward(t,x.grid[i,:],p,a,x.mp,scalar)
                end
                amax = argmax(container[p,:])
                value[i,:,p,t] = x.reward(t,x.grid[i,:], p,amax,x.mp)
                for pp in 1:x.pnum
                    value[i,:,p,t]+= x.positioncontrol[p,pp,amax] * evalue[i,:,pp, t]
                end
            end
        end
    end
    return(value,evalue)
end
