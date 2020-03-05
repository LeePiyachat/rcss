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
