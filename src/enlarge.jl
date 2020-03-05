function enlarge(subgradients::Matrix{Float64},x::rcss)
    result = Matrix{Float64}(x.gnum,x.snum)
    for i in 1:x.gnum
        result[i,:]=subgradients[argmax(subgradients*x.grid[i,:]),:]
    end
    return(result)
end
