##################################################
# load modules
###################################################
using Distributions
#using PyPlot
using NearestNeighbors
using Clustering
using Plots
#using StaticArrays
###################################################
#include("/home/juri/workspace/julia_rcss/rcss.jl")
####################################################
# run as module
#####################################################
push!(LOAD_PATH, ("/home/juri/workspace/julia_rcss/Modules/"))
using rcss_module
##################################################
#
# PARAMETERS
#

#let

#################################################
mp=Dict{String, Array{Real}}() # model parameters
################################################
mp["battery_capacity"]=[400.0]
mp["nbattery_levels"]=[21]
mp["step"]=[mp["battery_capacity"][1]/(mp["nbattery_levels"][1]-1)]
mp["std"]=[15.0]
mp["upper_safety"]=[30.0]
mp["lower_safety"]=[0.0]
mp["UPenalty"]=[50]
mp["LPenalty"]=[0]
mp["upper_charge"]=[10]
mp["lower_charge"]=[-10]
#############################
mp["nsafety_levels"]=[13]
mp["nbattery_actions"]=[9]
mp["action_map"]=zero(Matrix{Int64}(mp["nsafety_levels"][1]*mp["nbattery_actions"][1], 2))
mp["action_map"][:,1]=repeat( 1:mp["nsafety_levels"][1] , inner=mp["nbattery_actions"][1])
mp["action_map"][:,2]= repeat(1:mp["nbattery_actions"][1], outer=mp["nsafety_levels"][1] )
##############################
mp["mu"]=[1]
mp["sigma"]=[1]  # dynamics parameters
mp["phi"]= [0.9]
#############################
mp["snum"]=[2]   # problem dimension
mp["dnum"]=[1000] # distribution size
mp["rnum"]=[1]   # number of random elements
mp["pnum"]=[mp["nbattery_levels"][1]] # position number
mp["anum"]=[size(mp["action_map"])[1]]  # action number
###########################################
mp["ex"]=[2]     # kenrel parameter
###############################
mp["battery_levels"]=linspace(0,mp["battery_capacity"][1], mp["nbattery_levels"][1])
mp["safety_margins"]=linspace(mp["lower_safety"][1], mp["upper_safety"][1] ,mp["nsafety_levels"][1])
mp["charge_levels"]=linspace(mp["lower_charge"][1], mp["upper_charge"][1] , mp["nbattery_actions"][1])
###############################
function alfa(p::Int64, a::Int64) # yields next level index started with  index p and action a
pp=indmin(abs(
    mp["battery_levels"][:] -  mp["battery_levels"][p] - mp["charge_levels"][mp["action_map"][a,2]]
          ) )

    return(pp) # new battery level reached from p while action a is taken
end
#############################
mp["kapa"]=[1] # efficiency
mp["beta"]=zero(Matrix{Float64}(mp["pnum"][1], mp["anum"][1]))
               #
for p in 1:mp["pnum"][1] # energy amount supplied to (if positive)
                         # or taken from (if negative)  the battery
 for a in 1:mp["anum"][1]
     mp["beta"][p,a]=  mp["battery_levels"][alfa(p,a)]-mp["battery_levels"][p]
     if  mp["beta"][p,a]<0
        mp["beta"][p,a]=mp["kapa"][1]*mp["beta"][p,a]
     end
 end
end
####################################
#  Here continuation
###################################
mp["excess"]=zero(Matrix{Float64}(mp["pnum"][1], mp["anum"][1])  )
mp["shortage"]=zero(Matrix{Float64}(mp["pnum"][1], mp["anum"][1])  )
dist=Normal(0, mp["std"][1])

for a in 1:mp["anum"][1]
    for p in 1:mp["pnum"][1]
         d=mp["battery_levels"][1]-mp["step"][1]/2
         b=mp["battery_levels"][1]
         c=-mp["beta"][p, a]+ mp["safety_margins"][mp["action_map"][a,1]]
         mp["shortage"][p,a]=(mp["std"][1]^2)*pdf(dist, c-d)-(c-b)*(1-cdf(dist, c-d))

         d=mp["battery_levels"][mp["pnum"][1]]+mp["step"][1]/2
         b=mp["battery_levels"][ mp["pnum"][1] ]
         c=-mp["beta"][p,a]+ mp["safety_margins"][mp["action_map"][a,1]]
         mp["excess"][p,a]=(mp["std"][1]^2)*pdf(dist, d-c)-(b-c)*(1-cdf(dist, d-c))

    end
end
####################################
mp["eta1"]=[0.0]#[100.0]
mp["eta2"]=[0.0]#[15.0]
mp["deep_discharge"]=zero(Matrix{Float64}(mp["pnum"][1], mp["anum"][1]))
    for p in 1:mp["pnum"][1]
        for a in 1:mp["anum"][1]
            mp["deep_discharge"][p,a]=mp["eta1"][1]*(1 +
             mp["eta2"][1]*mp["battery_levels"][p]/mp["battery_capacity"][1])^(-1)
        end
    end
####################################
mp["u"]=zero(Vector{Float64}(169))
mp["u"][:]=-1 + cos( (2*pi/24)*(0:168))
mp["v"]=zero(Vector{Float64}(169))
mp["v"][:]= 1 + sin((2*pi/24)*(0:168)).^2

###############################
#
# DISTURBANCES
#
##############################
W=zero(Array{Float64}(mp["snum"][1], mp["snum"][1]))
W[1,1]=1                 # skeleton
W[2,2]=mp["phi"][1]
##############################
# Random entries indices
##############################
r_index=fill!(Matrix{Int64}(mp["rnum"][1],2), 0)
r_index[1,:]=[2,1]        # random entry index
##############################
# Realization of random entries
###############################
modif=zero(Matrix{Float64}(mp["rnum"][1], mp["dnum"][1])) # random entries
modif[1,:]=mp["mu"][1]+mp["sigma"][1]*quantile(Normal(0,1), linspace(1/(mp["dnum"][1]+2), 1-1/(mp["dnum"][1]+2), mp["dnum"][1]) )
###############################
# Weights of realizations
##############################
weight=zero(Vector{Float64}(mp["dnum"][1]))
weight[:]=1/mp["dnum"][1]            # weights
###############################
#
# CONTROL
#
################################
function Control(p::Int64, pp::Int64, a::Int64, mp::Dict{String, Array{Real}})

result=0.0
if pp==indmin(abs(mp["battery_levels"][:] -  mp["battery_levels"][p] - mp["charge_levels"][mp["action_map"][a,2]]))
     result=1.0
 end
 return(result)
end
#
####################################
# Scrap function
####################################
function Scrap(timestep:: Int64, state::Array{Float64,1}, p::Int64,  mp::Dict{String, Array{Real}})
    result=zero(Array{Float64,1}(mp["snum"][1]))
     result[1]=mp["u"][timestep]#u
    result[2]=mp["v"][timestep]#v
    result= mp["battery_levels"][p]*result
    return(result)
    # return(zero(result))
end
#################################
function Scrap(timestep:: Int64, state::Array{Float64,1}, p::Int64,  mp::Dict{String, Array{Real}}, scalar::String)

    return(mp["battery_levels"][p]*(mp["u"][timestep]+mp["v"][timestep]*state[2]))
   # return(0.0)
end
#################################
# Reward function
#################################
function Reward(timestep:: Int64,
                state::Array{Float64,1},
                p::Int64,
                a:: Int64,
                mp::Dict{String, Array{Real}}
                )


    result=zero(Array{Float64,1}(mp["snum"][1]))
    result[1]=mp["u"][timestep]
    result[2]=mp["v"][timestep]

    result=-mp["safety_margins"][ mp["action_map"][a,1]  ]*result

    result[1]=result[1]-mp["shortage"][p,a]*mp["UPenalty"][1]+
                        mp["excess"][p,a]*mp["LPenalty"][1]-
                         mp["deep_discharge"][p,a]

 return(result)
end
###################################

function Reward(timestep:: Int64,
                 state::Array{Float64,1},
                p::Int64,
                a:: Int64,
                mp::Dict{String, Array{Real}},
                scalar::String
                )

    return(  -mp["safety_margins"][ mp["action_map"][a,1]  ]*
             (mp["u"][timestep] +mp["v"][timestep]*state[2])-
              mp["shortage"][p,a]*mp["UPenalty"][1] + mp["excess"][p,a]*mp["LPenalty"][1]-
              mp["deep_discharge"][p,a]
             )
end

#######################################
gridsize=500
global initpoint=vec([1.0,10.0])
pathlength=168
pathnumber=100

global  X=rcss(gridsize,
               initpoint,
               pathlength,
               pathnumber,
               W,  #   (snum, snum)  distrubance skeleton
               r_index,  # (pran, 2) rows and columns of radom elements
               modif, #  (nran, dnum)for each column=disturbance random elements
               weight, # (dnum) weights of discrete distribution
               Control,  # control function
               Reward,  # reward functions
               Scrap,    #  scrap function
               mp
         );

#end
#############################################
# Solve control problem
############################################
tnum=48#  #pathlength
scalar="scalar"
index_be=2

#value, evalue = Bellman(tnum, X,2)

tic()
value, evalue = Bellman(tnum, X,index_be, scalar);
toc()

##################################################
# Show value functions
##################################################
Showplot(value[:,:,1, 1], X)
Showplot(value[:,:,10, 1], X)
Showplot(value[:,:,20, 1], X)
#################################################
# Run diagnostics
################################################
srand(1234)
#
trajectory_number=50
#
#################################################
#  produce results
################################################
#
target="/home/juri/data/windows/Documents/Lectures/artikel/BatteryStorage/BatteryPaper/"
samp=10
safety_margins=[]
charge_levels=[]
battery_levels=[]
prices=[]
state=[]

for i in 1:samp

initposition=1

result=policy_run(evalue,
                    X,
                    initpoint,
                    initposition,
                    )

  actions=result["actions"]
  postions=result["positions"]
  states=result["states"]

    safety_margins=[safety_margins...,
                    X.mp["safety_margins"][X.mp["action_map"][actions[:],1]]];
    charge_levels=[charge_levels...,
                    X.mp["charge_levels"][X.mp["action_map"][actions[:],2]]];
    battery_levels=[battery_levels...,
                     X.mp["battery_levels"][postions]];
    prices= [prices...,
             X.mp["u"][1:48]+ X.mp["v"][1:48].*states[2,:]];
    state=[state...,  states[2,:]]

end

############################################################
# plot results
###########################################################
using Plots
pyplot()
###########################################################
# plot price trajectories
##########################################################
price_plot= plot(prices, leg=false )
price_plot= title!(price_plot, "Price trajectories" )
display(price_plot)
savefig(string(target, "prices.eps"))
########################################################
# plot  state trajectories
##########################################################
state_plot= plot(state, leg=false )
state_plot= title!(state_plot, "State trajectories" )
display(state_plot)
savefig(string(target, "states.eps"))
########################################################
# plot running battery level with prices
########################################################
battery_plot= plot(battery_levels, leg=false, w=2 )
battery_plot= plot!(prices, leg=false)#,  palette=:grays)
battery_plot= title!(battery_plot, "Prices and battery levels" )
display(battery_plot)
savefig(
    string(target, "battery", Int(round(X.mp["eta1"]...)) ,"-", Int(round( X.mp["eta2"]...)), ".eps")
        )
##################################################
# plot value functions
########################################################
function getvalues(value::Matrix{Float64},  X::rcss)
y=(value.*X.grid)*[1,1]
oo=sortperm(X.grid[:,2])
return(X.grid[oo,2], y[oo])
end

value_plot=plot(getvalues(value[:,:,1, 1], X), leg=false )

for i in 2:size(value)[3]
     value_plot=plot!(getvalues(value[:,:,i, 1], X), leg=false )
end
value_plot= title!(value_plot, "Value functions" )
display(value_plot)
savefig(string(target, "value.eps"))
##################################################
# bound estimation
#################################################
result=Bound_est(value, evalue, initpoint, trajectory_number,X, 1, 2);

for p in 1:X.pnum
    print("\n")
    #
    @printf "%.0f"  X.mp["battery_levels"][p]; print(" & ");
    #
       print(" [" );
    @printf "%.4f" result[p,3 ,1]; print(",  "); @printf "%.4f" result[p,4 ,1]; print("] &")
    #
    print(" [" );
    @printf "%.f" result[p,3 ,2];  print(",  "); @printf "%.4f" result[p,4 ,2]; print("] " )
    print("\\"); print("\\")
end
