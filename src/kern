function kern(distances::Vector{Float64},x)
    ex = x.mp["ex"][1]
    minndx = argmin(distances)
    minval = distances[minndx]
    result = zero(distances)

    if minval < 0
        error("Distance in kernel is negative! = ", minva)
        result[minndx] = 1
    else
        if minval < 0.0000000001
            result[minndx] = 1
        else
            for i in 1:length(distances)
                result[i] = minval/distances[i]
            end
            result = result.^ex
            result = result/sum(result)
        end
    end
    if any(isnan,result)
        error("Distance in kernel is too small! = ")
    end
    return(result)
end
