function IndEnlarge(subgradients::Matrix{Float64},X::rcss)
    result = Vector{Int64}(X.gnum)
    for i in 1:X.gnum
        result[i]=indmax(subgradients * X.Grid[i,:])
    end
    return(result)
end
