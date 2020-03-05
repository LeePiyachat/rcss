function Enlarge(subgradients::Matrix{Float64},X::rcss)
    result = Matrix{Float64}(X.gnum,X.snum)
    for i in 1:X.gnum
        result[i,:]=subgradients[indmax(subgradients*X.grid[i,:]),:]
    end
    return(result)
end
