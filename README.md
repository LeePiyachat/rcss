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
using NearestNeighbors
using Clustering
using Plots
using Random
using Printf
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
function alfa(p::Int64, a::Int64) 
    temp1 = convert(Array{Float64},mp["battery_levels"][:])
    temp2 = mp["battery_levels"][p]
    temp3 = mp["charge_levels"][mp["action_map"][a,2]]
    pp = argmin(abs.(temp1 .- (temp2 + temp3)))
    return(pp) #new level reached from p while action a is taken
end
```
Introduce the derivative of the reward and scrap functions and control function:

```
# Control function
function control(p::Int64,pp::Int64,a::Int64,mp::Dict{String,Array{Real}})
    result=0.0
    if pp==argmin(abs.(mp["battery_levels"][:] .- (mp["battery_levels"][p] + mp["charge_levels"][mp["action_map"][a,2]])))
        result=1.0
    end
    return(result)
end
#Scrap function
function scrap(timestep::Int64,state::Array{Float64,1},p::Int64, mp::Dict{String,Array{Real}})
    result=zero(Array{Float64,1}(undef,mp["snum"][1]))
    result[1]=mp["u"][timestep] 
    result[2]=mp["v"][timestep] 
    result= mp["battery_levels"][p]*result
    return(result)
end
# Reward function
function reward(timestep::Int64,state::Array{Float64,1},p::Int64,a:: Int64, mp::Dict{String,Array{Real}})
    result=zero(Array{Float64,1}(undef,mp["snum"][1]))
    result[1]=mp["u"][timestep]
    result[2]=mp["v"][timestep]
    result=-mp["safety_margins"][mp["action_map"][a,1]] * result
    result[1]=result[1]-mp["shortage"][p,a] * mp["UPenalty"][1]+mp["excess"][p,a] *mp["LPenalty"][1]-mp["deep_discharge"][p,a]
    return(result)
end
```
We define the sampling disturbance which we assume it would be identically distributed across time. 

```
#subgradient representation of reward functions
mp["u"]=zero(Vector{Float64}(undef,169))
mp["u"][:]=-1 .+ cos.( (2*pi/24)*(0:168))
mp["v"]=zero(Vector{Float64}(undef,169))
mp["v"][:]=1 .+ sin.((2*pi/24)*(0:168)) .^ 2

#sampling disturbance
w=zero(Array{Float64}(undef,mp["snum"][1],mp["snum"][1]))
w[1,1]=1  #skeleton
w[2,2]=mp["phi"][1]

#Random entries indices
r_index=fill!(Matrix{Int64}(undef,mp["rnum"][1],2),0)
r_index[1,:]=[2,1]    #random entry index

# Realization of random entries
modif= zero(Matrix{Float64}(undef, mp["rnum"][1], mp["dnum"][1])) 
modif[1,:] = mp["mu"][1].+mp["sigma"][1]*quantile.(Normal(0,1), range(1/(mp["dnum"][1]+2),1-1/(mp["dnum"][1]+2);length=mp["dnum"][1]))

# Weights of realizations
weight= zero(Vector{Float64}(undef, mp["dnum"][1]))
weight[:].= 1 / mp["dnum"][1]  # weights

```
We summarize all the model information into an object
```
#create the rcss type 
global  x = rcss(gridsize,initpoint,pathlength,pathnumber,w,r_index,modif,weight,
			control,reward,scrap,mp);
```
Perform the Bellman recursion. Note that in this version of rcss, the user can set the number of nearest neighbors.The index_be can be 0, negative value, and positive value. 
    1) If index_be = 0, the traditional method is used. Traditional method finds the maximization of all tangents
    2) If index_be = positive value (1-50), the fast method is used.
    3) If index_be = negative value, the slow method method is used.
```
value, evalue = Bellman(tnum, x,index_be, scalar)
```
Bellman contains our approximates of the value functions, continuation value functions and prescribed policy at each grid point. 

## Example Battery storage - Confidence bounds
We compute the value function approximation above. Now, we can calculate the bounds using a duality martigale based approach. 
```
#Run diagnostic
trajectory_number=50  #setting number of trajectory
#Produce result
for i in 1:samp
    initposition = 1  
    result = policyrun(evalue,x,initpoint,initposition) #prescribed policy
    actions = result["actions"]
    positions=result["positions"]
    states = result["states"]
    push!(safety_margins,x.mp["safety_margins"][x.mp["action_map"][actions[:],1]]);
    push!(charge_levels,x.mp["charge_levels"][x.mp["action_map"][actions[:],2]]);
    push!(battery_levels,x.mp["battery_levels"][positions]);
    push!(prices, x.mp["u"][1:48]+x.mp["v"][1:48] .* states[2,:]);
    push!(state, states[2,:])
end
#Bound estimation (index_ph = 1 and index_va = 2)
result=boundest(value,evalue,initpoint,trajectory_number,x,1,2); 
for p in 1:x.pnum
    print("\n")
    @printf ".0f"  x.mp["battery_levels"][p]; print(" & "); print(" [" );
    @printf ".4f"result[p,3,1];print(",");@printf".4f"result[p,4 ,1]; print("] &")print("[");
    @printf ".f"result[p,3,2];print(",");@printf".4f"result[p,4 ,2];print("]")
    print("\\"); print("\\")
end
```
The above gives us: 
For the fast method with 2 nearest neighbors (computational time approximately 1.35 mins)
-484.7975 (0.2963) for lower bound
-484.7973 (0.2963) for upper bound
For the slow method with 2 nearest neighbors (computational time approximately 33.35 mins)
-484.7939 (0.2961) for lower bound
-484.7937 (0.2960) for upper bound
For the traditional method (computational time approximately 37 mins)
-484.7937 (0.2964) for lower bound
-484.7934 (0.2963) for upper bound

## Conclusion
Tighter bounds can be acheived by either improving the value function approximations by using
* a more dense grid
* a larger disturbance sampling
<br /> or by obtaining more suitable martingale increments through
* a larger number of sample paths
* a larger number of nested simulations
