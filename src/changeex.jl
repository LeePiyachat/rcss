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
