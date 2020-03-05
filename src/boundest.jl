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
