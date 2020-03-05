function showplot(value::Matrix{Float64},x::rcss)
    y =(value.* x.grid) * [1,1]
    oo = sortperm(x.grid[:,2])
    plot(x.grid[oo,2], y[oo])
end
