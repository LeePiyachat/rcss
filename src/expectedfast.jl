function expectedfast(value::Matrix{Float64},x::rcss)
    u = x.dmat[:,:,x.rnum + 1] * value * x.w
    for l in 1:x.rnum
        u[:,x.r_index[l,2]] += x.dmat[:,:,l] * value[:,x.r_index[l,1]]
    end
    return(u)
end
