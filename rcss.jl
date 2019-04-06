using Distributions
using PyPlot
using NearestNeighbors
using Clustering
########################
immutable rcss
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
     grid::Matrix{Float64} #  (gnum, snum) grid
     W::Matrix{Float64}  #   (snum, snum)  distrubance skeleton
    r_index::Matrix{Int64}  # (rnum, 2) rows and columns of radom elements
    modif::  Matrix{Float64} #  (rnum, dnum)for each column=disturbance random elements
    weight::Vector{Float64} # (dnum) weights of discrete distribution
    disturb::Array{Float64, 3} # (snum,snum,dnum)all distrubances
    Dmat::Array{Float64,3}   #(gnum, gnum,  rnum+1)  permutation matrices
    tree::NearestNeighbors.BallTree#{Float64,Distances.Euclidean}# next neighbors tree
    control::Array{Float64,3} #  (pnum, pnum, anum) stochastic control of positions
    Control::Function # control function
    Reward::Function # reward functions
    Scrap:: Function # scrap function
  #  ex::Vector{Int64} # parameter for kenrel function
    mp::Dict{String, Array{Real}}
    ########################################
    # inner constructor
    #######################################
    #
    rcss(gnum, snum, dnum, rnum, pnum, anum,  tree, Control,  Reward, Scrap, mp)= new(
        gnum, snum,  dnum, rnum, pnum, anum,
                                     ones(Vector{Int64}(1)),
                                     zero(Matrix{Float64}(gnum, snum)),
                                     zero(Matrix{Float64}(snum, snum)),
                                     zero(Matrix{Int64}(rnum, 2)),
                                     zero(Matrix{Float64}(rnum, dnum)),
                                     zero(Vector{Float64}(dnum)),
                                     zero(Array{Float64}(snum,snum,dnum)),
                                     zero(Array{Float64}(gnum, gnum,  rnum+1)),
                                     tree,
                                     zero(Array{Float64}(pnum, pnum, anum)),
                                     Control,
                                     Reward,
                                     Scrap,
                                    # ones(Vector{Int64}(1)),
                                     mp
                                     )

end

##########################
# outer constructors
#########################

function rcss(gridsize::Int64,
               initpoint::Vector{Float64},
               pathlength::Int64,
               pathnumber::Int64,
               W::Matrix{Float64},  #   (snum, snum)  distrubance skeleton
               r_index::Matrix{Int64},  # (rnum, 2) rows and columns of radom elements
               modif::  Matrix{Float64}, #  (rnum, dnum)for each column=disturbance random elements
               weight::Vector{Float64}, # (dnum) weights of discrete distribution
               Control::Function,  # reward functions
               Reward::Function,  # reward functions
              Scrap::Function,  #  scrap function
              mp::Dict{String, Array{Real}}
              )
              
   
disturb=Make_disturb(W, r_index, modif)  # generate stochastic grid
grid=StochasticGrid(gridsize,
                    initpoint,
                    pathlength,
                    pathnumber,
                    disturb,
                    weight)
    return(rcss(
               grid, #  (gnum, snum) grid
               W,  #   (snum, snum)  distrubance skeleton
               r_index,  # (rnum, 2) rows and columns of radom elements
               modif, #  (rnum, dnum)for each column=disturbance random elements
               weight, # (dnum) weights of discrete distribution
               Control,  # control function
               Reward,  # reward functions
               Scrap,   #  scrap function
               mp)
           )
    
end
#
function rcss(
               grid::Matrix{Float64}, #  (gnum, snum) grid
               W::Matrix{Float64},  #   (snum, snum)  distrubance skeleton
               r_index::Matrix{Int64},  # (rnum, 2) rows and columns of radom elements
               modif::  Matrix{Float64}, #  (rnum, dnum)for each column=disturbance random elements
               weight::Vector{Float64}, # (dnum) weights of discrete distribution
               Control::Function,
               Reward::Function,  # reward functions
               Scrap::Function,    #  scrap function
               mp::Dict{String, Array{Real}}
              )
    

    gnum=size(grid)[1]
    snum=size(grid)[2]
    dnum=size(modif)[2]
    rnum=size(modif)[1]
    pnum=mp["pnum"][1]
    anum=mp["anum"][1]
      
    
    ################
    # distrubances
    ###############
    disturb=Make_disturb(W, r_index, modif)

    
    tree=BallTree(transpose(grid))
    
    result=rcss(gnum, snum, dnum, rnum, pnum, anum,  tree, Control,  Reward, Scrap, mp)
                                        # call inner constructer to define fields
    result.grid[:,:]=grid               # fill fields 
    result.W[:,:]=W
    result.r_index[:,:]=r_index
    result.modif[:,:]=modif
    result.weight[:]=weight
    ###################
    println("making control")
    ######################
    # control
    #####################
   control=zero(Array{Float64,3}(pnum, pnum,  anum))

   for p in 1:size(control)[1]
    for pp in 1:size(control)[2]
       for a in 1:size(control)[3]
           control[p,pp,a]=Control(p,pp,a, mp)
       end
    end
   end
    ######################
    result.control[:,:,:]=control
    result.disturb[:,:,:]=disturb
    ################
    # Dmat
    ################
    result.Dmat[:,:,:]=Make_Dmat(result)
    #################
   return(result)
end
########################
# function definitions
#######################
 function Make_disturb(
               W::Matrix{Float64},  #   (snum, snum)  distrubance skeleton
               r_index::Matrix{Int64},  # (rnum, 2) rows and columns of radom elements
               modif::  Matrix{Float64} #  (rnum, dnum)for each column=disturbance random elements)
                       )
    snum=size(W)[1]
    dnum=size(modif)[2]
    rnum=size(modif)[1]
    disturb=zero(Array{Float64}(snum, snum, dnum))
                 
 for k in 1:dnum
         disturb[:,:,k]=W
         for i in 1:rnum
             disturb[r_index[i, 1], r_index[i, 2],k]=modif[i,k]
         end                                                        
 end
return(disturb)
end
###################
 function Enlarge(subgradients::Matrix{Float64}, X::rcss)
    result=Matrix{Float64}(X.gnum, X.snum)
      for i in 1:X.gnum
          result[i,:]=subgradients[indmax(subgradients*X.grid[i,:]),:]
       end 
return(result)              
end
###############################
 function IndEnlarge(subgradients::Matrix{Float64}, X::rcss)
    result=Vector{Int64}(X.gnum)
      for i in 1:X.gnum
          result[i]=indmax(subgradients*X.grid[i,:])
       end 
return(result)              
end
########################################
 function ExpectedSlow(value::Matrix{Float64}, X::rcss)
    
    result=zero(Matrix{Float64}(X.gnum, X.snum))
    
  if  X.nnum[1]==0
       for k in 1:X.dnum
       subgradients=value*X.disturb[:,:,k]*X.weight[k]
       result+=subgradients[IndEnlarge(subgradients, X),:]
       end

  else

       if (X.nnum[1]>0)
             nnum=X.nnum[1]
             kweights=zero(Array{Float64}(X.gnum, nnum))
       else
           nnum=-X.nnum[1]
           container=zero(Vector{Float64}(nnum))
       end
       
    

     for k in 1:X.dnum

        subgradients=transpose(value*X.disturb[:,:,k]*X.weight[k])
        hosts, dists= knn(X.tree, X.disturb[:,:,k]*transpose(X.grid), nnum, true)
        hosts=transpose(hcat(hosts...))
        dists=transpose(hcat(dists...))

       if (X.nnum[1]>0) 
         for i in 1:X.gnum 
              kweights[i,:]=kern(dists[i,:], X)
         end
         
         for i in 1:X.gnum
           for m in nnum    
            result[i,:]+=kweights[i,m]*subgradients[:, hosts[i,m]]
           end
         end
       else
          for i in 1:X.gnum
          S=subgradients[:, hosts[i,:]]
          result[i,:]+= S[:, indmax(transpose(S)*X.grid[i,:])]
         end    
       end     
     end
  end     
return(result)              
end
########################################
 function SimulatePath(initpoint::Vector{Float64},
                      pathlength::Int64,
                      pathnumber::Int64,
                      disturb::Array{Float64,3},
                      weight::Vector{Float64}
                      )
    distribution=Categorical(weight)
    #dnum=size(weight)[1]
    path=zero(Matrix{Float64}(length(initpoint), pathlength*pathnumber))
    path_labels=zero(Matrix{Int64}(1, pathlength*pathnumber)) 
    k=1::Int64
   
    for l in 1:pathnumber
        point=initpoint
        label=0
      for i in 1: pathlength
          path[:, k]=point
          path_labels[:, k]=label
          k=k+1
          label= rand(distribution)
          point=disturb[:,:, label ]*point
      end
    end
   return(path, path_labels)
 
end
#########################
 function SimulatePath(  initpoint::Vector{Float64},
                                            pathlength::Int64,
                                            pathnumber::Int64,
                                             X::rcss)
   return( SimulatePath(initpoint,pathlength, pathnumber, X.disturb, X.weight))
end
#############################
 function ExpectedFast(value::Matrix{Float64}, X::rcss)
  U=X.Dmat[:,:,X.rnum+1]*value*X.W
   for l in 1:X.rnum
     U[:,X.r_index[l,2]]+=X.Dmat[:,:,l]*value[:,X.r_index[l,1]]
   end
  return(U)
end
##############################
  function Bellman(  tnum:: Int64,
                  X::rcss,
                  index_be:: Int64
                   )
    #
    if  -50<=index_be<=0
        CondExpectation=ExpectedSlow
        println("Bellman with slow method")
        if X.nnum[1]!=index_be
             print("\n")
            print("changing index to ", -index_be, "neighbors \n" )
            X.nnum[:]=index_be
        end    
    elseif  0<index_be<=50
        CondExpectation=ExpectedFast
        println("Bellman with fast method")
        if X.nnum[1]!=index_be
            print("\n")
            print("recalculating matrix to ", index_be, " neighbors \n" )
            X.nnum[:]=index_be
            X.Dmat[:,:,:]=Make_Dmat(X)
        end
      else error("no such option for Bellman recursion")      
     end

    
# fields for value and expected value function    
value=zero(Array{Float64}(X.gnum, X.snum, X.pnum, tnum))
evalue=zero(Array{Float64}( X.gnum,  X.snum,  X.pnum,  tnum-1)) 
# initialize backward induction
t=tnum
container=zero(Matrix{Float64}(X.anum, X.snum))
    #

    #
  for p in 1: X.pnum    
  for i in 1: X.gnum
      value[i,:,p, t]= X.Scrap(t, X.grid[i,:], p, X.mp)
  end  
  end

     
#
t=tnum
    while (t>1)
     t=t-1
     print(t)
      for p in 1: X.pnum
     evalue[:,:,p, t]= CondExpectation(value[:,:,p, t+1], X)
     print(".")
 end
    #
  for i in 1: X.gnum
    for p in 1:X.pnum
     container=zero(container)
      for a in 1:X.anum
        for pp in 1:X.pnum
                  container[a,:]+= X.control[p,pp,a]*evalue[i,:,pp, t]
        end
       container[a,:]+=X.Reward(t, X.grid[i,:], p, a, X.mp)
      end
     amax=indmax(container*X.grid[i,:])
     value[i,:,p,t]=container[amax,:]
   end
  end
 end

    return (value, evalue)
end

##############################
      function Bellman(  tnum:: Int64,
                          X::rcss,
                           index_be:: Int64,
                           scalar::String )

    
    #
    if  -50<=index_be<=0
        CondExpectation=ExpectedSlow
        println("Bellman with slow method")
        if X.nnum[1]!=index_be
             print("\n")
            print("changing index to ", -index_be, "neighbors \n" )
            X.nnum[:]=index_be
        end    
    elseif  0<index_be<=50
        CondExpectation=ExpectedFast
        println("Bellman with fast method")
        if X.nnum[1]!=index_be
            print("\n")
            print("recalculating matrix to ", index_be, " neighbors \n" )
            X.nnum[:]=index_be
            X.Dmat[:,:,:]=Make_Dmat(X)
        end
      else error("no such option for Bellman recursion")      
     end

    
# fields for value and expected value function    
value=zero(Array{Float64}(X.gnum, X.snum, X.pnum, tnum))
evalue=zero(Array{Float64}( X.gnum,  X.snum,  X.pnum,  tnum-1)) 
# initialize backward induction
t=tnum
container=zero(Array{Float64}(X.pnum, X.anum))
evalues=zero(Vector{Float64}(X.pnum))     
    #

    #
  for p in 1: X.pnum    
   for i in 1: X.gnum
      value[i,:,p, t]= X.Scrap(t, X.grid[i,:], p, X.mp)
    end  
  end

     
#
t=tnum
    while (t>1)
     t=t-1
     print(t)
      for p in 1: X.pnum
     evalue[:,:,p, t]= CondExpectation(value[:,:,p, t+1], X)
     print(".")
     end
    #
     for i in 1: X.gnum
       for pp in 1:X.pnum           
       evalues[pp]=sum(evalue[i,:,pp, t].*X.grid[i,:])
       end


       for a in 1:X.anum
        container[:,a]=X.control[:,:,a]*evalues
       end 
         
      for p in 1:X.pnum
         for a in 1:X.anum
           container[p,a]+=X.Reward(t, X.grid[i,:], p, a, X.mp, scalar)
         end
       amax=indmax(container[p,:])
       value[i,:,p,t]=X.Reward(t, X.grid[i,:], p, amax, X.mp)
            for pp in 1:X.pnum
                   value[i,:,p,t]+= X.control[p,pp,amax]*evalue[i,:,pp, t]
               end   
      end
  end
 end
                                                      
    return (value, evalue)
end
##############################

 function StochasticGrid(gridsize::Int64,
                        initpoint::Vector{Float64},
                        length::Int64,
                        pathnumber::Int64,
                        disturb::Array{Float64,3},
                        weight::Vector{Float64})

path=SimulatePath(initpoint, length::Int64,pathnumber::Int64, disturb, weight)[1]
Y=kmeans(path, gridsize; maxiter=200, display=:iter)
return(transpose(Y.centers))
end
#########################
 function StochasticGrid(gridsize::Int64,
                        initpoint::Vector{Float64},
                        length::Int64,
                        pathnumber::Int64,
                        X::rcss
                        )


return(StochasticGrid(gridsize,
                      initpoint,
                      length,
                      pathnumber,
                      X.disturb,
                      X.weight)
       )
end
###########################################
 function get_val(
                   field::Array{Float64},
                   argument::Vector{Float64},
                   X::rcss,
                   index_va::Int64
                   )
   
  if index_va==0
      outcome=sum(maximum(field[:,:]*argument))
  elseif -50<=index_va<0
   
      hosts, dists= knn(X.tree, argument, -index_va, true)
      hosts=transpose(hcat(hosts...))
      outcome=sum(maximum(field[hosts[:,1],:]*argument))      
  elseif 0<index_va<=50
    
    kweights=zero(Array{Float64}(index_va ))
    hosts, dists= knn(X.tree, argument, index_va, true)
    
     hosts=transpose(hcat(hosts...))
     dists=transpose(hcat(dists...))
     kweights[:]=kern(dists[:], X)
        
     result=zero(Vector{Float64}( X.snum)) 
    
     for m in 1:index_va     
            result+=kweights[m]*field[hosts[m,1],:]
     end

      outcome=sum(result.*argument)
    
               
    else  error("wrong neighbors number in value access")

  end
    
     return(outcome)     
end
#############################################

function get_ph(
                 p::Int64,
                 a::Int64,
                 value::Array{Float64},
                 evalue::Array{Float64},
                 X::rcss,
                 z::Array{Float64},
                 z_labels::Array{Int64},
                 t::Int64,
                 index_ph::Int64,
                 index_va::Int64,
                 ) 

 if index_ph==0
        
       output=0.0
       for pp in 1:X.pnum
                  s=0
              for k in 1:X.dnum
                  s+=X.weight[k]*get_val(value[:,:, pp,t+1],  X.disturb[:,:,k]*z[:, t], X, index_va)
                  end
                  output +=  X.control[p,pp,a]*(
                        s   - get_val(value[:,:, pp, t+1], z[:, t+1], X, index_va)
                                               )
       end
     
 elseif   0<index_ph<=50
     
    kweights=zero(Array{Float64}(index_ph))
    hosts, dists= knn(X.tree, z[:, t], index_ph, true)
    
     hosts=transpose(hcat(hosts...))
     dists=transpose(hcat(dists...))
     kweights[:]=kern(dists[:], X)
    
    output=0.0
    
   for m in 1:index_ph
       result=0.0
       point=vec(X.grid[hosts[m,1],:])
       for pp in 1:X.pnum
             s= get_val(evalue[:,:, pp, t],  point, X, index_va)
                 # s=0 #  true martingale incremets, if access method in Bellman the same
                 # for k in 1:X.dnum
                 # s+=X.weight[k]*get_val(value[:,:, pp, t+1],  X.disturb[:,:, k]*point, X, index_va)
                #  end
                  
               result +=  X.control[p,pp,a]*(
                   s
                 - get_val(value[:,:, pp, t+1],  X.disturb[:,:, z_labels[t+1]]*point, X, index_va)
                                                  )
     end
       output+=kweights[m]*result 
   end
  else  error("wrong neighbors number in martingale correction")
 end     
    
return(output)   
end
#############################################################


function get_corrections(
                 value::Array{Float64},
                 evalue::Array{Float64},
                 X::rcss,
                 z::Array{Float64},
                 z_labels::Array{Int64},
                 t::Int64,
                 index_ph::Int64,
                 index_va::Int64,
                          )
                         
output=zero(Vector{Float64}(X.pnum))
                         
 if index_ph==0
        
       for pp in 1:X.pnum
                  s=0
              for k in 1:X.dnum
                  s+=X.weight[k]*get_val(value[:,:, pp,t+1],  X.disturb[:,:,k]*z[:, t], X, index_va)
                  end
                  output[pp] =  (
                        s   - get_val(value[:,:, pp, t+1], z[:, t+1], X, index_va)
                                               )
       end
     
 elseif   0<index_ph<=50
     
    kweights=zero(Array{Float64}(index_ph))
    hosts, dists= knn(X.tree, z[:, t], index_ph, true)
    
     hosts=transpose(hcat(hosts...))
     dists=transpose(hcat(dists...))
     kweights[:]=kern(dists[:], X)
    
    
   for m in 1:index_ph
       result=zero(Vector{Float64}(X.pnum))
       point=vec(X.grid[hosts[m,1],:])
       for pp in 1:X.pnum
             s= get_val(evalue[:,:, pp, t],  point, X, index_va)
                 # s=0 #  true martingale incremets, if access method in Bellman the same
                 # for k in 1:X.dnum
                 # s+=X.weight[k]*get_val(value[:,:, pp, t+1],  X.disturb[:,:, k]*point, X, index_va)
                #  end
                  
               result[pp] =   (
                   s
                 - get_val(value[:,:, pp, t+1],  X.disturb[:,:, z_labels[t+1]]*point, X, index_va)
                                                  )
     end
       output+=kweights[m]*result 
   end
  else  error("wrong neighbors number in martingale correction")
 end     
    
return(output)   
end
################################
 function Bound_est(value::Array{Float64},
                   evalue::Array{Float64},  
                    initpoint::Vector{Float64},
                   trajectory_number::Int64,
                   X::rcss,
                   index_ph::Int64,
                   index_va::Int64
                 )
#    
    if (index_va!=X.nnum[1])&(index_ph>0)
        println()
        error("access index for fast methods must match!")
    end

 mid=zero(Array{Float64}(X.pnum, trajectory_number ))
 hig=zero(Array{Float64}(X.pnum,  trajectory_number))
 low=zero(Array{Float64}(X.pnum,  trajectory_number))
#    
# number of time steps
tnum=size(value)[4]
# running value
tvalue=zero(Array{Float64}(X.pnum))
lvalue=zero(Array{Float64}(X.pnum))
hvalue=zero(Array{Float64}(X.pnum))
######
     corrections=zero(Array{Float64}(X.pnum))
     evalues=zero(Array{Float64}(X.pnum))
#######
tvalue_new=zero(Array{Float64}(X.pnum))
lvalue_new=zero(Array{Float64}(X.pnum))
hvalue_new=zero(Array{Float64}(X.pnum))     
#
#     
# container for values
container_true=zero(Vector{Float64}(X.anum))
container_low=zero(Vector{Float64}(X.anum))
container_high=zero(Vector{Float64}(X.anum))
# here the loop for trajectories
for j in 1:trajectory_number
 print(j); print(".")           
 t=tnum 
 pathsimulation=SimulatePath(initpoint, tnum, 1,X)
   z= pathsimulation[1]
   z_labels=pathsimulation[2]

            
   for p in 1:X.pnum
    tvalue[p]=X.Scrap(t, z[:,t], p, X.mp, "scalar")#sum(X.Scrap(t, z[:,t], p, X.mp).*z[:,t]) 
    lvalue[p]=tvalue[p] 
    hvalue[p]=lvalue[p]
   end

    while (t>1)       
        t=t-1

     
       corrections[:]= get_corrections(
                                value,
                                evalue,
                                X,
                                z,
                                z_labels,
                                t,
                                index_ph,
                                index_va
                                       )
       for pp in 1:X.pnum
        evalues[pp]= get_val(evalue[:,:,pp,t],  z[:,t], X, index_va)
       end     
        
       for p in 1:X.pnum
        container_true=zero(container_true)
        container_low=zero(container_low)
        container_high=zero(container_high)
        
        
      for a in 1:X.anum
            correction=0.0
       for pp in 1:X.pnum
           container_true[a]+= X.control[p,pp,a]*evalues[pp] #get_val(evalue[:,:,pp,t],  z[:,t], X, index_va)
           container_low[a]+= X.control[p,pp,a]*lvalue[pp]   #get_val(value[:,:,pp,t+1],  z[:,t+1], X, index_va)
           container_high[a]+= X.control[p,pp,a]*hvalue[pp]
           correction+= X.control[p,pp,a]*corrections[pp]
       end
      
          payoff=X.Reward(t, z[:,t], p, a, X.mp, "scalar") #sum(X.Reward(t, z[:,t], p, a, X.mp).*z[:,t])
          #correction=get_ph(p,a, value, evalue, X, z, z_labels,  t, index_ph, index_va)
  
          container_true[a]+=payoff 
          container_low[a]+= payoff+correction
          container_high[a]+=payoff+correction
      end
        policy_action=indmax(container_true)
        maxima_action=indmax(container_high)

        
        tvalue_new[p]=container_true[policy_action] 
        lvalue_new[p]=container_low[policy_action]
        hvalue_new[p]=container_high[maxima_action]

    end
       tvalue[:] = tvalue_new[:]
       lvalue[:] = lvalue_new[:]
       hvalue[:] = hvalue_new[:]

 end
 
    mid[:,j]=tvalue[:]
    low[:,j]=lvalue[:]
    hig[:,j]=hvalue[:]

 
end
result=zero(Array{Float64}(X.pnum, 4, 4))
            for p in 1:X.pnum
            #
            result[p, 1, 1]=mean(low[p,:]) #mltrack[p] 
            result[p, 2, 1]=std(low[p,:])#sltrack[p]
            #
            result[p, 1, 2]=mean(hig[p,:]) #mhtrack[p]
            result[p, 2, 2]=std(hig[p,:]) 
            #
            result[p, 1, 3]=get_val(value[:,:,p,1], initpoint,X, index_va)
            result[p, 2, 3]=0
             #
            result[p, 1, 4]=mean(mid[p,:]) #mttrack[p]
            result[p, 2, 4]=std(mid[p,:])#sttrack[p]
            end
  for j in 1:2
    for p in 1:X.pnum
       if  result[p,2 ,j]>0
           result[p,3:4,j]=
           quantile(Normal(result[p,1 ,j], result[p,2 ,j]/sqrt(trajectory_number) ), [0.025, 0.975]) 
        else
        result[p,3:4,j]= [result[p,1 ,j], result[p,1 ,j]]   
       end
    end
  end
     print("\n")
println("-----bound estimation-on--", trajectory_number, "--trajectories")
for p in 1:X.pnum
    println(" for p=", p)
    #
   println("approx value function")
    @printf "%.6f" result[p,1 ,3];print("("); @printf "%.6f" result[p,2 ,3]; print(") \n")
    #
    println("backward induction value")
    @printf "%.6f" result[p,1 ,4];print("("); @printf "%.6f" result[p,2 ,4]; print(")  \n")  
    println("-------")
    #
    println("lower estimate")
    @printf "%.6f" result[p,1 ,1];print("("); @printf "%.6f" result[p,2 ,1]; print(") ")
    print(" [" );
    @printf "%.6f" result[p,3 ,1]; print(","); @printf "%.6f" result[p,4 ,1]; print("] \n")
    #
    println("upper estimate")
    @printf "%.6f" result[p,1 ,2];print("("); @printf "%.6f" result[p,2 ,2]; print(") ")
    print(" [" ); 
    @printf "%.6f" result[p,3 ,2];  print(","); @printf "%.6f" result[p,4 ,2]; print("] \n")
end

    return(result)
 end
      
##########################################################################################
 function Make_Dmat(X::rcss)
      print(" Making Dmat \n")

Dmats=zero(Array{Float64}(X.gnum, X.gnum,  X.rnum+1))
kweights=zero(Array{Float64}(X.gnum, X.nnum[1]))

    for k in 1:X.dnum 
        hosts, dists= knn(X.tree, X.disturb[:,:,k]*transpose(X.grid), X.nnum[1], true)
        hosts=transpose(hcat(hosts...))
        dists=transpose(hcat(dists...))
          for i in 1:X.gnum 
              kweights[i,:]=kern(dists[i,:], X)
          end
        for i in 1:X.gnum
            for m in 1:X.nnum[1]
                Dmats[i, hosts[i,m], X.rnum+1]+=  kweights[i,m]*X.weight[k]
            end
            for l in 1:X.rnum
              for m in 1:X.nnum[1]
                  Dmats[i, hosts[i,m],l] +=  kweights[i,m]*X.modif[l,k]*X.weight[k]
              end    
            end
         end
        
    end
    println(" finished Dmat \n")
return(Dmats)
end
#######################################################################
 function kern(distances::Vector{Float64}, X)
    
   ex=X.mp["ex"][1]
    
    minndx=indmin(distances)
    minval=distances[minndx]
    result=zero(distances)

 if minval<0
     error("distance in kernel negative ! = ", minva)
     result[minndx]=1
  else
   if minval < 0.0000000001
     result[minndx]=1
   else
     for i in 1:length(distances)
                result[i]=minval/distances[i]
     end
     result=result.^ex  
     result=result/sum(result)
    end
  end              
    if any(isnan, result)
                  error("distance in kernel too small ! = ")
     end
    
return(result)    
end
########################################################################
 function Showplot(value::Matrix{Float64},  X::rcss)
y=(value.*X.grid)*[1,1]
oo=sortperm(X.grid[:,2])
plot(X.grid[oo,2], y[oo])
end
#######################################################################
 function ChangeEx(ex::Int64,  X::rcss)
    if (ex!=X.mp["ex"][1])&(ex>0)
       print("\n")
       print("changing interpolation parameter from", X.mp["ex"][1], "  to ", ex, " \n" )
        X.mp["ex"][1]=ex
        X.Dmat[:,:,:]=Make_Dmat(X)
    else     print("\n")
        print("nothing changed \n" )
    end         
end
#######################################################################

function policy_run(evalue::Array{Float64},
                    X:: rcss,
                    initpoint::Vector{Float64},
                    initposition::Int64
                    )

tnum=size(evalue)[4]+1
evalues=zero(Array{Float64}(X.pnum))
container=zero(Array{Float64}(X.pnum, X.anum))
policy=zero(Array{Int64}(X.pnum, tnum-1))
pathsimulation=SimulatePath(initpoint, tnum, 1,X)
states= pathsimulation[1]
index_va=X.nnum[1]
 #
t=1
# determine control policy
while t<tnum 

 for p in 1:X.pnum
        evalues[p]= get_val(evalue[:,:,p,t],  states[:,t], X, index_va)
 end
    
   
 for a in 1:X.anum
           container[:, a]= X.control[:,:,a]*evalues 
 end

 for p in 1:X.pnum
        for a in 1:X.anum
           container[p, a]+=X.Reward(t, states[:,t], p, a, X.mp, "scalar")
        end
 end

 for p in 1:X.pnum
     policy[p, t]=indmax(container[p,:])
 end

 t=t+1   
end
# now calculate positions and actiions trajectory
distributions=Array{Categorical}(X.pnum, X.anum)
for p in 1:X.pnum
 for a in 1:X.anum
     distributions[p, a]=Categorical(X.control[p, :, a])
 end
end


positions=zero(Vector{Int64}(tnum))
actions=zero(Vector{Int64}(tnum-1) )
positions[1]=initposition
t=1
while (t<tnum)
    actions[t]= policy[positions[t], t]
    positions[t+1]=rand(distributions[positions[t], actions[t] ] )
    t=t+1
end

result=Dict{String, Array{Real}}()

    result["policy"]=policy
    result["states"]=states
    result["positions"]=positions
    result["actions"]=actions
 return(result)
end
####################################################################################################
