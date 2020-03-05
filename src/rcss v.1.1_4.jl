using Distributions
using NearestNeighbors
using Clustering
using Printf
using Plots
using Random
########################
struct rcss
    ##########################################
    # dimensions
    #########################################

    gnum::Int64 # grid size
    snum::Int64 # state size
    dnum::Int64 # distribution size
    rnum::Int64 # number of random elements
    pnum::Int64 # number of positions
    anum::Int64 # number of actions
    nnum::Vector{Int64} # number of next neighbors

    #########################################
    # data fields
    #######################################

    grid::Matrix{Float64}           #(gnum,snum) grid
    w::Matrix{Float64}              #(snum,snum) disturbance skeleton
    r_index::Matrix{Int64}          #(rnum,2) rows and columns of radom elements
    modif::Matrix{Float64}          #(rnum,dnum)for each column = disturbance random elements
    weight::Vector{Float64}         #(dnum)weights of discrete distribution
    disturb::Array{Float64, 3}      #(snum,snum,dnum)all disturbances
    dmat::Array{Float64,3}          #(gnum,gnum,rnum+1)permutation matrices
    tree::NearestNeighbors.BallTree #{Float64,Distances.Euclidean}# Next Neighbors tree
    positioncontrol::Array{Float64,3}    #(pnum,pnum,anum)stochastic control of positions
    control::Function               #control function
    reward::Function                #reward functions
    scrap:: Function                #scrap function
    #ex::Vector{Int64}              #parameter for kernel function
    mp::Dict{String, Array{Real}}

    ########################################
    # inner constructor
    #######################################
    rcss(gnum,snum,dnum,rnum,pnum,anum,tree,control,reward,scrap,mp) = new(gnum,snum,dnum,rnum,pnum,anum,ones(Int64, 1),
                                     zero(Matrix{Float64}(undef, gnum,snum)),
                                     zero(Matrix{Float64}(undef, snum,snum)),
                                     zero(Matrix{Int64}(undef, rnum,2)),
                                     zero(Matrix{Float64}(undef, rnum,dnum)),
                                     zero(Vector{Float64}(undef, dnum)),
                                     zero(Array{Float64}(undef, snum,snum,dnum)),
                                     zero(Array{Float64}(undef, gnum, gnum,rnum+1)),
                                     tree,
                                     zero(Array{Float64}(undef, pnum, pnum, anum)),
                                     control,
                                     reward,
                                     scrap,
                                     #ones(Vector{Int64}(1)),
                                     mp
                                     )
                                 end

##########################
# outer constructors
#########################
function rcss(gridsize::Int64,initpoint::Array{Float64,1},
                                pathlength::Int64,
                                pathnumber::Int64,
                                w::Matrix{Float64},
                                r_index::Array{Int64,2},
                                modif::Matrix{Float64},
                                weight::Vector{Float64},
                                control::Function,
                                reward::Function,
                                scrap::Function,
                                mp::Dict{String,Array{Real}}
                                )
disturb     =   make_disturb(w, r_index, modif)  #generate stochastic grid
grid        =   stochasticgrid(gridsize,initpoint,pathlength,pathnumber,disturb,weight)

return(rcss(grid, w, r_index, modif, weight, control, reward, scrap, mp))

end
##########################################################################################
function rcss(grid::AbstractArray{Float64, 2},w::Matrix{Float64},r_index::Matrix{Int64},modif::Matrix{Float64},weight::Vector{Float64},control,reward,scrap,mp)
    gnum = size(grid)[1]
    snum = size(grid)[2]
    dnum = size(modif)[2]
    rnum = size(modif)[1]
    pnum = mp["pnum"][1]
    anum = mp["anum"][1]
    ################
    # disturbances
    ###############
    disturb =   make_disturb(w, r_index,modif)
    tree    =   BallTree(transpose(grid))

    result = rcss(gnum, snum, dnum, rnum, pnum, anum,tree,control,reward,scrap,mp)
    # call inner constructer to define fields
    result.grid[:,:] = grid               # fill fields
    result.w[:,:] = w
    result.r_index[:,:] = r_index
    result.modif[:,:] = modif
    result.weight[:] = weight

    ###################
    println("making control")
    ######################
    # control
    #####################
    positioncontrol =   zero(Array{Float64,3}(undef, pnum,pnum,anum))
    for p in 1:size(positioncontrol)[1]
        for pp in 1:size(positioncontrol)[2]
            for a in 1:size(positioncontrol)[3]
                positioncontrol[p,pp,a] = control(p,pp,a,mp)
            end
        end
    end
    ######################
    result.positioncontrol[:,:,:] = positioncontrol
    result.disturb[:,:,:] = disturb
    ################
    # Dmat
    ################
    result.dmat[:,:,:] = make_dmat(result)
    #################
    return(result)
end

################################################################################################
# function definitions
###############################################################################################
function make_disturb(w::Matrix{Float64},r_index::Matrix{Int64},modif::Matrix{Float64})
    snum = size(w)[1]
    dnum = size(modif)[2]
    rnum = size(modif)[1]
    disturb = zero(Array{Float64}(undef, snum, snum, dnum))
    for k in 1:dnum
        disturb[:,:,k] = w
        for i in 1:rnum
            disturb[r_index[i,1],r_index[i,2],k] = modif[i,k]
        end
    end
    return(disturb)
end
###############################################################################################
function enlarge(subgradients::Matrix{Float64},x::rcss)
    result = Matrix{Float64}(x.gnum,x.snum)
    for i in 1:x.gnum
        #result[i,:]=subgradients[indmax(subgradients*x.grid[i,:]),:]
        result[i,:]=subgradients[argmax(subgradients*x.grid[i,:]),:]
    end
    return(result)
end
###############################################################################################
function indenlarge(subgradients::Matrix{Float64},x::rcss)
    result = Vector{Int64}(undef,x.gnum)
    for i in 1:x.gnum
        #result[i]=indmax(subgradients * x.grid[i,:])
        result[i]=argmax(subgradients * x.grid[i,:])
    end
    return(result)
end
###############################################################################################
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
###############################################################################################
function simulatepath(initpoint::Vector{Float64},pathlength::Int64,pathnumber::Int64,disturb::Array{Float64,3},weight::Vector{Float64})
    distribution = Categorical(weight)
    #dnum=size(weight)[1]
    path = zero(Matrix{Float64}(undef,length(initpoint),pathlength * pathnumber))
    path_labels = zero(Matrix{Int64}(undef,1, pathlength * pathnumber))
    k = 1::Int64
    for l in 1:pathnumber
        point = initpoint
        label = 0
        for i in 1: pathlength
            path[:, k] = point
            path_labels[:, k] .= label
            k = k+1
            label = rand(distribution)
            point = disturb[:,:,label ] * point
        end
    end
    return(path, path_labels)
end

function simulatepath(initpoint::Vector{Float64},pathlength::Int64,pathnumber::Int64,x::rcss)
    return(simulatepath(initpoint,pathlength,pathnumber,x.disturb,x.weight))
end
###############################################################################################
function expectedfast(value::Matrix{Float64},x::rcss)
    u = x.dmat[:,:,x.rnum + 1] * value * x.w
    for l in 1:x.rnum
        u[:,x.r_index[l,2]] += x.dmat[:,:,l] * value[:,x.r_index[l,1]]
    end
    return(u)
end
###############################################################################################
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
############################################################################
function stochasticgrid(gridsize::Int64,initpoint::Vector{Float64},length::Int64,pathnumber::Int64,x::rcss)
    return(stochasticgrid(gridsize,initpoint,length,pathnumber,x.disturb,x.weight))
end
############################################################################
function stochasticgrid(gridsize::Int64, initpoint::Vector{Float64},length::Int64, pathnumber::Int64, disturb::Array{Float64,3}, weight::Vector{Float64})
    path = simulatepath(initpoint,length::Int64,pathnumber::Int64,disturb,weight)[1]
    Y = kmeans(path, gridsize; maxiter = 200, display=:iter)
    return(transpose(Y.centers))
end
###########################################
function get_val(field::Array{Float64},argument::Vector{Float64},x::rcss,index_va::Int64)
    if index_va == 0
        outcome = sum(maximum(field[:,:] * argument))
    elseif -50 <= index_va < 0
        hosts, dists = knn(x.tree,argument,-index_va,true)
        hosts   = transpose(hcat(hosts...))
        outcome = sum(maximum(field[hosts[:,1],:] * argument))
    elseif 0 < index_va <= 50
        kweights    =   zero(Array{Float64}(undef,index_va))
        hosts, dists=   knn(x.tree,argument,index_va,true)
        hosts       =   transpose(hcat(hosts...))
        dists       =   transpose(hcat(dists...))
        kweights[:] =   kern(dists[:],x)
        result      =   zero(Vector{Float64}(undef,x.snum))
        for m in 1:index_va
            result += kweights[m] * field[hosts[m,1],:]
        end
        outcome     =   sum(result.* argument)
    else error("Wrong Neighbors number in value access")
    end
    return(outcome)
end
#############################################
function get_ph(p::Int64,a::Int64,value::Array{Float64},evalue::Array{Float64},x::rcss,z::Array{Float64},z_labels::Array{Int64},t::Int64,index_ph::Int64,index_va::Int64)
    if index_ph == 0
        output=0.0
        for pp in 1:x.pnum
            s = 0
            for k in 1:x.dnum
                s += x.weight[k] * get_val(value[:,:, pp,t+1],x.disturb[:,:,k] * z[:, t],x,index_va)
            end
            output +=  x.positioncontrol[p,pp,a]*(s - get_val(value[:,:, pp,t+1], z[:,t+1], x,index_va))
        end
    elseif 0 < index_ph <= 50
        kweights = zero(Array{Float64}(undef,index_ph))
        hosts,dists = knn(x.tree,z[:, t],index_ph,true)
        hosts = transpose(hcat(hosts...))
        dists = transpose(hcat(dists...))
        kweights[:] = kern(dists[:],x)
        output = 0.0
        for m in 1:index_ph
            result = 0.0
            point = vec(x.grid[hosts[m,1],:])
            for pp in 1:x.pnum
                s   = get_val(evalue[:,:, pp,t],point,x, index_va)
                #s=0 #true martingale incremets, if access method in Bellman the same
                #for k in 1:x.dnum
                #s+=x.weight[k]*get_val(value[:,:, pp, t+1],  x.disturb[:,:, k]*point, x, index_va)
                #end
                result +=  x.positioncontrol[p,pp,a] * (s-get_val(value[:,:,pp,t+1],x.disturb[:,:,z_labels[t+1]] * point,x,index_va))
            end
            output += kweights[m]*result
        end
    else error("Wrong Neighbors number in Martingale correction")
    end
    return(output)
end
#############################################################
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
################################
function boundest(value::Array{Float64},evalue::Array{Float64},initpoint::Vector{Float64},trajectorynumber::Int64,x::rcss,index_ph::Int64,index_va::Int64)
    if (index_va != x.nnum[1]) & (index_ph > 0)
        error("Access Index for Fast methods must match!")
    end
    mid     = zero(Array{Float64}(undef,x.pnum,trajectorynumber))
    high    = zero(Array{Float64}(undef,x.pnum,trajectorynumber))
    low     = zero(Array{Float64}(undef,x.pnum,trajectorynumber))

    #number of time steps
    tnum = size(value)[4]

    #running value
    tvalue = zero(Array{Float64}(undef,x.pnum))
    lvalue = zero(Array{Float64}(undef,x.pnum))
    hvalue = zero(Array{Float64}(undef,x.pnum))

   ######
   corrections  = zero(Array{Float64}(undef,x.pnum))
   evalues      = zero(Array{Float64}(undef,x.pnum))
   #evalue      = zero(Array{Float64}(undef,x.pnum))
   #######
   tvalue_new = zero(Array{Float64}(undef,x.pnum))
   lvalue_new = zero(Array{Float64}(undef,x.pnum))
   hvalue_new = zero(Array{Float64}(undef,x.pnum))

   #container for values
   container_true = zero(Vector{Float64}(undef,x.anum))
   container_low = zero(Vector{Float64}(undef,x.anum))
   container_high = zero(Vector{Float64}(undef,x.anum))

   #The loop for trajectories
   for j in 1:trajectorynumber
       print(j); print(".")
       t = tnum
       pathsimulation = simulatepath(initpoint,tnum,1,x)
       z = pathsimulation[1]
       z_labels = pathsimulation[2]
       for p in 1:x.pnum
           tvalue[p] = x.scrap(t,z[:,t],p, x.mp,"scalar") #sum(x.scrap(t, z[:,t], p, x.mp).*z[:,t])
           lvalue[p] = tvalue[p]
           hvalue[p] = lvalue[p]
       end
       while (t>1)
           t = t-1
           corrections[:]= get_corrections(value,evalue,x,z,z_labels,t,index_ph,index_va)
           for pp in 1:x.pnum
               evalues[pp]= get_val(evalue[:,:,pp,t],z[:,t], x, index_va)
           end
           for p in 1:x.pnum
               container_true   =   zero(container_true)
               container_low    =   zero(container_low)
               container_high   =   zero(container_high)
               for a in 1:x.anum
                   correction = 0.0
                   for pp in 1:x.pnum
                       container_true[a]+= x.positioncontrol[p,pp,a] * evalues[pp]
                       #get_val(evalue[:,:,pp,t],  z[:,t], x, index_va)
                       #container_true[a]+= x.control[p,pp,a] * evalue[pp]
                       container_low[a] += x.positioncontrol[p,pp,a] * lvalue[pp]   #get_val(value[:,:,pp,t+1],  z[:,t+1], x, index_va)
                       container_high[a]+= x.positioncontrol[p,pp,a] * hvalue[pp]
                       correction += x.positioncontrol[p,pp,a] * corrections[pp]
                   end
                   payoff = x.reward(t,z[:,t],p,a,x.mp,"scalar") #sum(x.reward(t,z[:,t],p,a,x.mp) * z[:,t])
                   container_true[a]+= payoff
                   container_low[a] += payoff + correction
                   container_high[a]+= payoff + correction
               end
               policy_action = argmax(container_true)
               maxima_action = argmax(container_high)

               tvalue_new[p] = container_true[policy_action]
               lvalue_new[p] = container_low[policy_action]
               hvalue_new[p] = container_high[maxima_action]
           end
           tvalue[:] = tvalue_new[:]
           lvalue[:] = lvalue_new[:]
           hvalue[:] = hvalue_new[:]
       end
       mid[:,j] = tvalue[:]
       low[:,j] = lvalue[:]
       high[:,j]= hvalue[:]
   end
   result = zero(Array{Float64}(undef,x.pnum,4, 4))
   for p in 1:x.pnum
       #
       result[p, 1, 1] = mean(low[p,:]) #mltrack[p]
       result[p, 2, 1] = std(low[p,:])#sltrack[p]
       #
       result[p, 1, 2] = mean(high[p,:]) #mhtrack[p]
       result[p, 2, 2] = std(high[p,:])
       #
       result[p, 1, 3] = get_val(value[:,:,p,1], initpoint,x,index_va)
       result[p, 2, 3] = 0
       #
       result[p, 1, 4] = mean(mid[p,:]) #mttrack[p]
       result[p, 2, 4] = std(mid[p,:]) #sttrack[p]
   end
   for j in 1:2
       for p in 1:x.pnum
           if  result[p,2 ,j] > 0
               result[p,3:4,j]= quantile.(Normal(result[p,1,j],result[p,2,j]/sqrt(trajectorynumber)),[0.025,0.975])
           else
               result[p,3:4,j]= [result[p,1 ,j], result[p,1 ,j]]
           end
       end
   end
   print("\n")
   println("--Bound Estimation on--",trajectorynumber, "--trajectories")

   for p in 1:x.pnum
       println(" for p=", p)
       #
       println("approximate value function")
       @printf "%.6f" result[p,1 ,3];print("("); @printf "%.6f" result[p,2 ,3]; print(") \n")
       #
       println("Backward induction value")
       @printf "%.6f" result[p,1 ,4];print("("); @printf "%.6f" result[p,2 ,4]; print(")  \n")
       println("-------")
       #
       println("Lower estimate")
       @printf "%.6f" result[p,1 ,1];print("("); @printf "%.6f" result[p,2 ,1]; print(") ")
       print(" [" );
       @printf "%.6f" result[p,3 ,1]; print(","); @printf "%.6f" result[p,4 ,1]; print("] \n")
       #
       println("Upper estimate")
       @printf "%.6f" result[p,1 ,2];print("("); @printf "%.6f" result[p,2 ,2]; print(") ")
       print(" [" );
       @printf "%.6f" result[p,3 ,2];  print(","); @printf "%.6f" result[p,4 ,2]; print("] \n")
   end
   return(result)
end
##########################################################################################

#################################################################################
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
#######################################################################
function kern(distances::Vector{Float64},x)
    ex = x.mp["ex"][1]
    minndx = argmin(distances)
    minval = distances[minndx]
    result = zero(distances)

    if minval < 0
        error("Distance in kernel is negative! = ", minva)
        result[minndx] = 1
    else
        if minval < 0.0000000001
            result[minndx] = 1
        else
            for i in 1:length(distances)
                result[i] = minval/distances[i]
            end
            result = result.^ex
            result = result/sum(result)
        end
    end
    if any(isnan,result)
        error("Distance in kernel is too small! = ")
    end
    return(result)
end
########################################################################
function showplot(value::Matrix{Float64},x::rcss)
    y =(value.* x.grid) * [1,1]
    oo = sortperm(x.grid[:,2])
    plot(x.grid[oo,2], y[oo])
end
#######################################################################
function changeex(ex::Int64,x::rcss)
    if (ex!= x.mp["ex"][1]) & (ex > 0)
        print("\n")
        print("Changing interpolation parameter from", x.mp["ex"][1], "to",ex,"\n" )
        x.mp["ex"][1] = ex
        x.dmat[:,:,:] = make_dmat(x)
    else
        print("\n")
        print("Nothing changed \n" )
    end
end
#######################################################################
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
