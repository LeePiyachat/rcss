function ExpectedFast(Value::Matrix{Float64},X::rcss)
    U = X.Dmat[:,:,X.rnum + 1] * Value * X.W
    for l in 1:X.rnum
        U[:,X.R_index[l,2]] += X.Dmat[:,:,l] * Value[:,X.R_index[l,1]]
    end
    return(U)
end
