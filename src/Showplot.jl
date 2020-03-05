function Showplot(Value::Matrix{Float64},X::rcss)
    y =(Value.* X.Grid) * [1,1]
    oo = sortperm(X.Grid[:,2])
    plot(X.Grid[oo,2], y[oo])
end
