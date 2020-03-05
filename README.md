# Julia programming-rcss

## Authors
* **Piyachat Leelasilapasart** 
* **Juri Hinz**

## About
This Julia package is implemented in Julia V. 1.1.0 which provides a method for approximating the value function in Markov decision processes under linear state dynamic, convex reward function, 
and convex scrap function using convex piecewise linear functions. Please submit any issues through my GitHub or email 
(piyachat.peung@gmail.com) 

## Problem setting 
We impose the following restrictions on our Markov decision process
1)	A finite number of time points
2)	A Markov process consisting of 
<br />  a) A controlled Markov chain with a finite number of possible positions 
<br />  b) continuous process that evolves linearly i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=X_{t&plus;1}&space;=&space;W_{t&plus;1}X_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{t&plus;1}&space;=&space;W_{t&plus;1}X_t" title="X_{t+1} = W_{t+1}X_t" /></a> where <a href="https://www.codecogs.com/eqnedit.php?latex=W_{t&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W_{t&plus;1}" title="W_{t+1}" /></a> is a matrix with random entries.
3)	Reward and scarp functions that are convex and Lipchitz continuous in the continuous process.
4)	A finite number of actions.
<br /> This Julia package approximate all the value functions in the Bellman recursion and also computes their lower and upper bounds. The following code demonstrates:

## Example: Battery storage-Value Function Approximation
Let us assume that within a given time horizon <a href="https://www.codecogs.com/eqnedit.php?latex=t=0,...,T-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t=0,...,T-1" title="t=0,...,T-1" /></a>
(unit:hours), an energy retailer has the obligation to satisfy the unknown energy demand of his customers while retailer's renewable 
energy sources produce a random electricity amount. We first set our parameters;

```
###################################################
# Load Packages
###################################################
using Distributions
#using PyPlot
using NearestNeighbors
using Clustering
using Plots
using Printf
using Random

##################################################
#Load rcss Module
##################################################
using rcss_module

##################################################
#PARAMETERS
##################################################
mp=Dict{String, Array{Real}}()  #model parameters
##################################################
mp["battery_capacity"]  =[100.0]  ##battery capacity
mp["nbattery_levels"]   =[21]     ##battery level
mp["step"]              =[mp["battery_capacity"][1]/(mp["nbattery_levels"][1]-1)]
mp["std"]               =[15.0]
mp["upper_safety"]      =[30.0]
mp["lower_safety"]      =[0.0]
mp["UPenalty"]          =[50]
mp["LPenalty"]          =[0]
mp["upper_charge"]      =[10]
mp["lower_charge"]      =[-10]
```
<br /> Set the transition probabilities for controlled Markov chain.
```
control =   zero(Array{Float64,3}(undef,pnum,pnum,anum))
    for p in 1:size(control)[1]
        for pp in 1:size(control)[2]
            for a in 1:size(control)[3]
                control[p,pp,a]=Control(p,pp,a, mp)
            end
        end
    end
```
Introduce the derivative of the reward and scrap functions:

```
####################################
# SCRAP FUNCTION
####################################
function Scrap(timestep::Int64,state::Array{Float64,1},p::Int64,mp::Dict{String,Array{Real}})
    result=zero(Array{Float64,1}(undef,mp["snum"][1]))
    result[1]=mp["u"][timestep] #u
    result[2]=mp["v"][timestep] #v
    result= mp["battery_levels"][p]*result
    return(result)
    #return(zero(result))
end
#################################
function Scrap(timestep::Int64,state::Array{Float64,1},p::Int64,mp::Dict{String,Array{Real}},scalar::String)
    return(mp["battery_levels"][p]*(mp["u"][timestep]+mp["v"][timestep]*state[2]))
    #return(0.0)
end
#################################
# Reward function
#################################
function Reward(timestep::Int64,state::Array{Float64,1},p::Int64,a:: Int64,mp::Dict{String,Array{Real}})
    result      =zero(Array{Float64,1}(undef,mp["snum"][1]))
    result[1]   =mp["u"][timestep]
    result[2]   =mp["v"][timestep]
    result      =-mp["safety_margins"][mp["action_map"][a,1]] * result
    result[1]   =result[1]-mp["shortage"][p,a] * mp["UPenalty"][1]+mp["excess"][p,a]*mp["LPenalty"][1]-mp["deep_discharge"][p,a]
    return(result)
end
###################################
function Reward(timestep::Int64,state::Array{Float64,1},p::Int64,a:: Int64,mp::Dict{String,Array{Real}},scalar::String)
    return(-mp["safety_margins"][ mp["action_map"][a,1]]*(mp["u"][timestep] +mp["v"][timestep]*state[2])-
              mp["shortage"][p,a]*mp["UPenalty"][1]+mp["excess"][p,a]*mp["LPenalty"][1]-mp["deep_discharge"][p,a])
end
```

We define the sampling disturbance <a href="https://www.codecogs.com/eqnedit.php?latex=(W_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(W_t)" title="(W_t)" /></a> which we assume it would be identically distributed across time. 

```
mp["dnum"]  =[1000] # distribution size
W   =   zero(Array{Float64}(undef,mp["snum"][1],mp["snum"][1]))
W[1,1] = 1                 # skeleton
W[2,2] = mp["phi"][1]
##############################
# Realization of random entries
###############################
Modif   =zero(Matrix{Float64}(undef,mp["rnum"][1],mp["dnum"][1])) # random entries
Modif[1,:] = mp["mu"][1] .+ mp["sigma"][1]*quantile.(Normal(0,1), range(1/(mp["dnum"][1]+2), 1-1/(mp["dnum"][1]+2);length=mp["dnum"][1]))
Weight  = zero(Vector{Float64}(undef, mp["dnum"][1]))
Weight[:] .= 1 / mp["dnum"][1]     
```
We summarize all the model information into an object

```
#create the rcss type 
rcss(Grid, W, R_index, Modif, Weight, Control, Reward, Scrap, mp)
```
Perform the Bellman recursion using fast method.

```
R_index=fill!(Matrix{Int64}(undef,mp["rnum"][1],2),0)
R_index[1,:]=[2,1]        # random entry index

#Function to construct the search structure

#Bellman
```
Bellman contains our approximates of the value functions, continuation value functions and prescribed policy at each grid point. The value
function of the option can be plotted using following
```
using Plots
price_plot  = plot(prices, leg=false)  #Change from false to true
state_plot  = plot(state, leg=false)
battery_plot= plot(battery_levels, leg=false, w=2)

function Getvalues(Value::Matrix{Float64},X::rcss)
    y = (Value .* X.Grid) * [1, 1]
    oo = sortperm(X.Grid[:, 2])
    return(X.Grid[oo, 2], y[oo])
end

value_plot = plot(Getvalues(Value[:,:,1, 1],X),leg=false)
for i in 2:size(Value)[3]
    value_plot=plot!(Getvalues(Value[:,:,i, 1],X),leg=false )
end
```
## Example Battery storage - Bounds
We compute the function approximation above. Now, we can calculate the bounds using a pathwise dynamic programming approach. We
generate a set of sample paths for
```
Trajectory_number=50
result=BoundEst(Value,Evalue,Initpoint,Trajectory_number,X,1,2);

for p in 1:X.pnum
    print("\n")
    @printf "%.0f"  X.mp["battery_levels"][p]; print(" & ");
    print(" [" );
    @printf "%.4f" result[p,3 ,1]; print(",  "); @printf "%.4f" result[p,4 ,1]; print("] &")
    print(" [" );
    @printf "%.f" result[p,3 ,2];  print(",  "); @printf "%.4f" result[p,4 ,2]; print("] " )
    print("\\"); print("\\")
end
```
The above gives us:

## Conclusion
Tighter bounds can be acheived by either improving the value function approximations by using
* a more dense grid
* a larger disturbance sampling
<br /> or by obtaining more suitable martingale increments through
* a larger number of sample paths
* a larger number of nested simulations
