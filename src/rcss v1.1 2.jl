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

    Grid::Matrix{Float64}           #(gnum,snum) grid
    W::Matrix{Float64}              #(snum,snum) disturbance skeleton
    R_index::Matrix{Int64}          #(rnum,2) rows and columns of radom elements
    Modif::Matrix{Float64}          #(rnum,dnum)for each column = disturbance random elements
    Weight::Vector{Float64}         #(dnum)weights of discrete distribution
    disturb::Array{Float64, 3}      #(snum,snum,dnum)all disturbances
    Dmat::Array{Float64,3}          #(gnum,gnum,rnum+1)permutation matrices
    tree::NearestNeighbors.BallTree #{Float64,Distances.Euclidean}# Next Neighbors tree
    control::Array{Float64,3}       #(pnum,pnum,anum)stochastic control of positions
    Control::Function               #control function
    Reward::Function                #reward functions
    Scrap:: Function                #scrap function
    #ex::Vector{Int64}              #parameter for kernel function
    mp::Dict{String, Array{Real}}

    ########################################
    # inner constructor
    #######################################
    rcss(gnum,snum,dnum,rnum,pnum,anum,tree,Control,Reward,Scrap,mp) = new(gnum,snum,dnum,rnum,pnum,anum,ones(Int64, 1),
                                     zero(Matrix{Float64}(undef, gnum,snum)),
                                     zero(Matrix{Float64}(undef, snum,snum)),
                                     zero(Matrix{Int64}(undef, rnum,2)),
                                     zero(Matrix{Float64}(undef, rnum,dnum)),
                                     zero(Vector{Float64}(undef, dnum)),
                                     zero(Array{Float64}(undef, snum,snum,dnum)),
                                     zero(Array{Float64}(undef, gnum, gnum,rnum+1)),
                                     tree,
                                     zero(Array{Float64}(undef, pnum, pnum, anum)),
                                     Control,
                                     Reward,
                                     Scrap,
                                     #ones(Vector{Int64}(1)),
                                     mp
                                     )
                                 end

##########################
# outer constructors
#########################
function rcss(Gridsize::Int64,Initpoint::Array{Float64,1},
                                Pathlength::Int64,
                                Pathnumber::Int64,
                                W::Matrix{Float64},
                                R_index::Array{Int64,2},
                                Modif::Matrix{Float64},
                                Weight::Vector{Float64},
                                Control::Function,
                                Reward::Function,
                                Scrap::Function,
                                mp::Dict{String,Array{Real}}
                                )
disturb     =   Make_disturb(W, R_index, Modif)  #generate stochastic grid
Grid        =   StochasticGrid(Gridsize,Initpoint,Pathlength,Pathnumber,disturb,Weight)

return(rcss(Grid, W, R_index, Modif, Weight, Control, Reward, Scrap, mp))

end
##########################################################################################
function rcss(Grid::AbstractArray{Float64, 2},W::Matrix{Float64},R_index::Matrix{Int64},Modif::Matrix{Float64},Weight::Vector{Float64},Control,Reward,Scrap,mp)

    gnum = size(Grid)[1]
    snum = size(Grid)[2]
    dnum = size(Modif)[2]
    rnum = size(Modif)[1]
    pnum = mp["pnum"][1]
    anum = mp["anum"][1]

    ################
    # disturbances
    ###############
    disturb =   Make_disturb(W, R_index,Modif)
    tree    =   BallTree(transpose(Grid))

    result = rcss(gnum, snum, dnum, rnum, pnum, anum,tree,Control,Reward,Scrap,mp)
    # call inner constructer to define fields
    result.Grid[:,:] = Grid               # fill fields
    result.W[:,:] = W
    result.R_index[:,:] = R_index
    result.Modif[:,:] = Modif
    result.Weight[:] = Weight

    ###################
    println("making control")
    ######################
    # control
    #####################
    control =   zero(Array{Float64,3}(undef, pnum,pnum,anum))
    for p in 1:size(control)[1]
        for pp in 1:size(control)[2]
            for a in 1:size(control)[3]
                control[p,pp,a]=Control(p,pp,a, mp)
            end
        end
    end
    ######################
    result.control[:,:,:] = control
    result.disturb[:,:,:] = disturb
    ################
    # Dmat
    ################
    result.Dmat[:,:,:] = Make_Dmat(result)
    #################
    return(result)
end

################################################################################################
# function definitions
###############################################################################################
function Make_disturb(W::Matrix{Float64},R_index::Matrix{Int64},Modif::Matrix{Float64})
    snum = size(W)[1]
    dnum = size(Modif)[2]
    rnum = size(Modif)[1]
    disturb = zero(Array{Float64}(undef, snum, snum, dnum))
    for k in 1:dnum
        disturb[:,:,k] = W
        for i in 1:rnum
            disturb[R_index[i,1],R_index[i,2],k] = Modif[i,k]
        end
    end
    return(disturb)
end
###############################################################################################
function Enlarge(subgradients::Matrix{Float64},X::rcss)
    result = Matrix{Float64}(X.gnum,X.snum)
    for i in 1:X.gnum
        #result[i,:]=subgradients[indmax(subgradients*X.grid[i,:]),:]
        result[i,:]=subgradients[argmax(subgradients*X.grid[i,:]),:]
    end
    return(result)
end
###############################################################################################
function IndEnlarge(subgradients::Matrix{Float64},X::rcss)
    result = Vector{Int64}(undef,X.gnum)
    for i in 1:X.gnum
        #result[i]=indmax(subgradients * X.Grid[i,:])
        result[i]=argmax(subgradients * X.Grid[i,:])
    end
    return(result)
end
###############################################################################################
function ExpectedSlow(Value::Matrix{Float64},X::rcss)
    #result = zero(Matrix{Float64}(X.gnum,X.snum))
    result = zero(Matrix{Float64}(undef,X.gnum,X.snum))
    if  X.nnum[1]== 0
        for k in 1:X.dnum
            subgradients = Value * X.disturb[:,:,k] * X.Weight[k]
            result += subgradients[IndEnlarge(subgradients,X),:]
        end
    else
        if (X.nnum[1] > 0)
            nnum = X.nnum[1]
            kweights = zero(Array{Float64}(undef,X.gnum,nnum))
        else
            nnum =- X.nnum[1]
            container = zero(Vector{Float64}(undef,nnum))
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
                    #result[i,:]+= S[:, indmax(transpose(S) * X.Grid[i,:])]
                    result[i,:]+= S[:, argmax(transpose(S) * X.Grid[i,:])]
                end
            end
        end
    end
    return(result)
end
###############################################################################################
function SimulatePath(Initpoint::Vector{Float64},Pathlength::Int64,Pathnumber::Int64,disturb::Array{Float64,3},Weight::Vector{Float64})
    distribution = Categorical(Weight)
    #dnum=size(weight)[1]
    path = zero(Matrix{Float64}(undef,length(Initpoint),Pathlength * Pathnumber))
    path_labels = zero(Matrix{Int64}(undef,1, Pathlength * Pathnumber))
    k = 1::Int64
    for l in 1:Pathnumber
        point = Initpoint
        label = 0
        for i in 1: Pathlength
            path[:, k] = point
            path_labels[:, k] .= label
            k = k+1
            label = rand(distribution)
            point = disturb[:,:,label ] * point
        end
    end
    return(path, path_labels)
end

function SimulatePath(Initpoint::Vector{Float64},Pathlength::Int64,Pathnumber::Int64,X::rcss)
    return(SimulatePath(Initpoint,Pathlength,Pathnumber,X.disturb,X.Weight))
end
###############################################################################################
function ExpectedFast(Value::Matrix{Float64},X::rcss)
    U = X.Dmat[:,:,X.rnum + 1] * Value * X.W
    for l in 1:X.rnum
        U[:,X.R_index[l,2]] += X.Dmat[:,:,l] * Value[:,X.R_index[l,1]]
    end
    return(U)
end
###############################################################################################
function Bellman(tnum::Int64,X::rcss,Index_be::Int64)
    if  -50 <= Index_be <= 0
        CondExpectation = ExpectedSlow
        println("Bellman with Slow method")
        if X.nnum[1]!= Index_be
            print("\n")
            print("Changing index to ", - Index_be, " neighbors \n" )
            X.nnum[:].=Index_be
        end
    elseif  0 < Index_be <= 50
        CondExpectation = ExpectedFast
        println("Bellman with Fast method")
        if X.nnum[1]!= Index_be
            print("\n")
            print("Recalculating matrix to ", Index_be, " neighbors \n" )
            X.nnum[:] .= Index_be
            X.Dmat[:,:,:] = Make_Dmat(X)
        end
    else error("No option for Bellman recursion")
    end
    #fields for value and expected value function
    Value = zero(Array{Float64}(undef,X.gnum,X.snum,X.pnum,tnum))
    Evalue= zero(Array{Float64}(undef,X.gnum,X.snum,X.pnum,tnum-1))
    #initialize backward induction

    t = tnum
    container = zero(Matrix{Float64}(undef, X.anum,X.snum))

    for p in 1: X.pnum
        for i in 1: X.gnum
            Value[i,:,p, t] = X.Scrap(t,X.grid[i,:],p, X.mp)
        end
    end

    t = tnum
    while (t > 1)
        t = t-1
        print(t)
        for p in 1: X.pnum
            #Evalue[:,:,p,t]= CondExpectation(value[:,:,p, t+1], X)
            Evalue[:,:,p,t]= CondExpectation(Value[:,:,p, t+1], X)
            print(".")
        end
        #
        for i in 1: X.gnum
            for p in 1:X.pnum
                container = zero(container)
                for a in 1:X.anum
                    for pp in 1:X.pnum
                        container[a,:] += X.control[p,pp,a] * Evalue[i,:,pp,t]
                    end
                    container[a,:]+= X.Reward(t,X.Grid[i,:], p, a, X.mp)
                end
                #amax = indmax(container * X.Grid[i,:])
                amax = argmax(container * X.Grid[i,:])
                Value[i,:,p,t] = container[amax,:]
            end
        end
    end
    return(Value, Evalue)
end
############################################################################
function Bellman(tnum::Int64,X::rcss,Index_be::Int64,scalar::String)
    if  -50 <= Index_be <= 0
        CondExpectation = ExpectedSlow
        println("Bellman with slow method")
        if X.nnum[1]!= Index_be
            print("\n")
            print("Changing index to ", -Index_be,"neighbors \n")
            X.nnum[:] .= Index_be
        end
    elseif  0 < Index_be <=50
        CondExpectation = ExpectedFast
        println("Bellman with fast method")
        if X.nnum[1] != Index_be
            print("\n")
            print("Recalculating matrix to ", Index_be, " neighbors \n")
            X.nnum[:] .= Index_be
            X.Dmat[:,:,:] = Make_Dmat(X)
        end
    else error("No such option for Bellman recursion")
    end
    #fields for value and expected value function
    Value   =   zero(Array{Float64}(undef, X.gnum,X.snum,X.pnum,tnum))
    Evalue  =   zero(Array{Float64}(undef, X.gnum,X.snum,X.pnum,tnum-1))
    #initialize backward induction
    t = tnum
    container   = zero(Array{Float64}(undef, X.pnum,X.anum))
    Evalues     = zero(Vector{Float64}(undef, X.pnum))
    #Evalue    = zero(Vector{Float64}(undef, X.pnum))
    #
    for p in 1: X.pnum
        for i in 1: X.gnum
            Value[i,:,p, t]= X.Scrap(t, X.Grid[i,:],p,X.mp)
        end
    end
    #
    t = tnum
    while (t > 1)
        t = t - 1
        print(t)
        for p in 1: X.pnum
            Evalue[:,:,p,t]= CondExpectation(Value[:,:,p, t+1],X)
            print(".")
        end
        #
        for i in 1: X.gnum
            for pp in 1:X.pnum
                Evalues[pp] = sum(Evalue[i,:,pp, t].* X.Grid[i,:])
                #Evalue[pp] = sum(Evalue[i,:,pp, t].* X.Grid[i,:])
            end
            for a in 1:X.anum
                container[:,a] = X.control[:,:,a] * Evalues
            end
            for p in 1:X.pnum
                for a in 1:X.anum
                    container[p,a] += X.Reward(t,X.Grid[i,:],p,a,X.mp,scalar)
                end
                amax = argmax(container[p,:])
                Value[i,:,p,t] = X.Reward(t,X.Grid[i,:], p,amax,X.mp)
                for pp in 1:X.pnum
                    Value[i,:,p,t]+= X.control[p,pp,amax] * Evalue[i,:,pp, t]
                end
            end
        end
    end
    return(Value,Evalue)
end
############################################################################
function StochasticGrid(Gridsize::Int64,Initpoint::Vector{Float64},Length::Int64,Pathnumber::Int64,X::rcss)
    return(StochasticGrid(Gridsize,Initpoint,Length,Pathnumber,X.disturb,X.Weight))
end
############################################################################
function StochasticGrid(Gridsize::Int64, Initpoint::Vector{Float64},Length::Int64, Pathnumber::Int64, disturb::Array{Float64,3}, Weight::Vector{Float64})
    path = SimulatePath(Initpoint,Length::Int64,Pathnumber::Int64,disturb,Weight)[1]
    Y = kmeans(path, Gridsize; maxiter = 200, display=:iter)
    return(transpose(Y.centers))
end
###########################################
function Get_val(Field::Array{Float64},Argument::Vector{Float64},X::rcss,Index_va::Int64)
    if Index_va == 0
        outcome = sum(maximum(Field[:,:] * Argument))
    elseif -50 <= Index_va < 0
        hosts, dists = knn(X.tree,Argument,-Index_va,true)
        hosts   = transpose(hcat(hosts...))
        outcome = sum(maximum(Field[hosts[:,1],:] * Argument))
    elseif 0 < Index_va <= 50
        kweights    =   zero(Array{Float64}(undef,Index_va))
        hosts, dists=   knn(X.tree,Argument,Index_va,true)
        hosts       =   transpose(hcat(hosts...))
        dists       =   transpose(hcat(dists...))
        kweights[:] =   Kern(dists[:],X)
        result      =   zero(Vector{Float64}(undef,X.snum))
        for m in 1:Index_va
            result += kweights[m] * Field[hosts[m,1],:]
        end
        outcome     =   sum(result.* Argument)
    else error("Wrong Neighbors number in value access")
    end
    return(outcome)
end
#############################################
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
        kweights = zero(Array{Float64}(undef,Index_ph))
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
#############################################################
function Get_corrections(Value::Array{Float64},Evalue::Array{Float64},X::rcss,z::Array{Float64},z_labels::Array{Int64},t::Int64,Index_ph::Int64,Index_va::Int64)
    output  =   zero(Vector{Float64}(undef,X.pnum))
    if Index_ph == 0
        for pp in 1:X.pnum
            s = 0
            for k in 1:X.dnum
                s += X.Weight[k] * Get_val(Value[:,:,pp,t+1],X.disturb[:,:,k] * z[:,t],X,Index_va)
            end
            output[pp] = (s - Get_val(Value[:,:,pp,t+1],z[:,t+1],X,Index_va))
        end
    elseif   0 < Index_ph <= 50
        kweights    = zero(Array{Float64}(undef,Index_ph))
        hosts,dists = knn(X.tree,z[:,t],Index_ph,true)
        hosts       = transpose(hcat(hosts...))
        dists       = transpose(hcat(dists...))
        kweights[:] = Kern(dists[:], X)
        for m in 1:Index_ph
            result  = zero(Vector{Float64}(undef,X.pnum))
            point   = vec(X.Grid[hosts[m,1],:])
            for pp in 1:X.pnum
                s = Get_val(Evalue[:,:,pp,t],point,X,Index_va)
                #s=0 #  true martingale incremets, if access method in Bellman the same
                #for k in 1:X.dnum
                #s+=X.weight[k]*get_val(value[:,:, pp, t+1],  X.disturb[:,:, k]*point, X, index_va)
                #end
                result[pp] = (s - Get_val(Value[:,:,pp,t+1], X.disturb[:,:,z_labels[t+1]] * point,X,Index_va))
            end
            output += kweights[m]*result
        end
    else  error("Wrong Neighbors number in martingale correction")
    end
    return(output)
end
################################
function BoundEst(Value::Array{Float64},Evalue::Array{Float64},Initpoint::Vector{Float64},Trajectory_number::Int64,X::rcss,Index_ph::Int64,Index_va::Int64)
    #println(Value, Evalue,Initpoint,Trajectory_number,X,Index_ph,Index_va)
    println(typeof(Value))
    println(typeof(Evalue))
    println(typeof(Initpoint))
    println(typeof(Trajectory_number))
    println(typeof(Index_ph))
    println(typeof(Index_va))
    println(X.nnum[1])
    if (Index_va != X.nnum[1]) & (Index_ph > 0)
        error("Access Index for Fast methods must match!")
    end
    mid     = zero(Array{Float64}(undef,X.pnum,Trajectory_number))
    high    = zero(Array{Float64}(undef,X.pnum,Trajectory_number))
    low     = zero(Array{Float64}(undef,X.pnum,Trajectory_number))

    #number of time steps
    tnum = size(Value)[4]

    #running value
    tvalue = zero(Array{Float64}(undef,X.pnum))
    lvalue = zero(Array{Float64}(undef,X.pnum))
    hvalue = zero(Array{Float64}(undef,X.pnum))

   ######
   corrections  = zero(Array{Float64}(undef,X.pnum))
   Evalues      = zero(Array{Float64}(undef,X.pnum))
   #Evalue      = zero(Array{Float64}(undef,X.pnum))
   #######
   tvalue_new = zero(Array{Float64}(undef,X.pnum))
   lvalue_new = zero(Array{Float64}(undef,X.pnum))
   hvalue_new = zero(Array{Float64}(undef,X.pnum))

   #container for values
   container_true = zero(Vector{Float64}(undef,X.anum))
   container_low = zero(Vector{Float64}(undef,X.anum))
   container_high = zero(Vector{Float64}(undef,X.anum))

   #The loop for trajectories
   for j in 1:Trajectory_number
       print(j); print(".")
       t = tnum
       Pathsimulation = SimulatePath(Initpoint,tnum,1,X)
       z = Pathsimulation[1]
       z_labels = Pathsimulation[2]
       for p in 1:X.pnum
           tvalue[p] = X.Scrap(t,z[:,t],p, X.mp,"scalar") #sum(X.Scrap(t, z[:,t], p, X.mp).*z[:,t])
           lvalue[p] = tvalue[p]
           hvalue[p] = lvalue[p]
       end
       while (t>1)
           t = t-1
           corrections[:]= Get_corrections(Value,Evalue,X,z,z_labels,t,Index_ph,Index_va)
           for pp in 1:X.pnum
               Evalues[pp]= Get_val(Evalue[:,:,pp,t],z[:,t], X, Index_va)
               #Evalue[pp]= Get_val(Evalue[:,:,pp,t],z[:,t], X, Index_va)
           end
           for p in 1:X.pnum
               container_true   =   zero(container_true)
               container_low    =   zero(container_low)
               container_high   =   zero(container_high)
               for a in 1:X.anum
                   correction = 0.0
                   for pp in 1:X.pnum
                       container_true[a]+= X.control[p,pp,a] * Evalues[pp] #get_val(evalue[:,:,pp,t],  z[:,t], X, index_va)
                       #container_true[a]+= X.control[p,pp,a] * Evalue[pp]
                       container_low[a] += X.control[p,pp,a] * lvalue[pp]   #get_val(value[:,:,pp,t+1],  z[:,t+1], X, index_va)
                       container_high[a]+= X.control[p,pp,a] * hvalue[pp]
                       correction += X.control[p,pp,a] * corrections[pp]
                   end
                   payoff = X.Reward(t,z[:,t],p,a,X.mp,"scalar") #sum(X.Reward(t,z[:,t],p,a,X.mp) * z[:,t])
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
   result = zero(Array{Float64}(undef,X.pnum,4, 4))
   for p in 1:X.pnum
       #
       result[p, 1, 1] = mean(low[p,:]) #mltrack[p]
       result[p, 2, 1] = std(low[p,:])#sltrack[p]
       #
       result[p, 1, 2] = mean(high[p,:]) #mhtrack[p]
       result[p, 2, 2] = std(high[p,:])
       #
       result[p, 1, 3] = Get_val(Value[:,:,p,1], Initpoint,X,Index_va)
       result[p, 2, 3] = 0
       #
       result[p, 1, 4] = mean(mid[p,:]) #mttrack[p]
       result[p, 2, 4] = std(mid[p,:]) #sttrack[p]
   end
   for j in 1:2
       for p in 1:X.pnum
           if  result[p,2 ,j] > 0
               result[p,3:4,j]= quantile.(Normal(result[p,1,j],result[p,2,j]/sqrt(Trajectory_number)),[0.025,0.975])
           else
               result[p,3:4,j]= [result[p,1 ,j], result[p,1 ,j]]
           end
       end
   end
   print("\n")
   println("--Bound Estimation on--",Trajectory_number, "--trajectories")

   for p in 1:X.pnum
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
#######################################################################
function Kern(distances::Vector{Float64},X)
    ex = X.mp["ex"][1]
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
function Showplot(Value::Matrix{Float64},X::rcss)
    y =(Value.* X.Grid) * [1,1]
    oo = sortperm(X.Grid[:,2])
    plot(X.Grid[oo,2], y[oo])
end
#######################################################################
function ChangeEx(ex::Int64,X::rcss)
    if (ex!= X.mp["ex"][1]) & (ex > 0)
        print("\n")
        print("Changing interpolation parameter from", X.mp["ex"][1], "to",ex,"\n" )
        X.mp["ex"][1] = ex
        X.Dmat[:,:,:] = Make_Dmat(X)
    else
        print("\n")
        print("Nothing changed \n" )
    end
end
#######################################################################
function PolicyRun(Evalue::Array{Float64},X::rcss,Initpoint::Vector{Float64},Initposition::Int64)
    tnum            = size(Evalue)[4]+1
    Evalues         = zero(Array{Float64}(undef,X.pnum))
    #Evalue         = zero(Array{Float64}(undef,X.pnum))
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
            #Evalues[p]= Get_val(Evalue[:,:,p,t],states[:,t],X,Index_va)
            Evalue[p]= Get_val(Evalue[:,:,p,t],states[:,t],X,Index_va)
        end
        for a in 1:X.anum
            container[:, a]= X.control[:,:,a] * Evalues
            #container[:, a]= X.control[:,:,a] * Evalue
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
