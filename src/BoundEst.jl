function BoundEst(Value::Array{Float64},Evalue::Array{Float64},Initpoint::Vector{Float64},Trajectory_number::Int64,X::rcss,Index_ph::Int64,Index_va::Int64)
    if (Index_va!= X.nnum[1]) & (Index_ph > 0)
        println()
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
           end
           for p in 1:X.pnum
               container_true   =   zero(container_true)
               container_low    =   zero(container_low)
               container_high   =   zero(container_high)
               for a in 1:X.anum
                   correction = 0.0
                   for pp in 1:X.pnum
                       container_true[a]+= X.control[p,pp,a] * Evalues[pp] #get_val(evalue[:,:,pp,t],  z[:,t], X, index_va)
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
