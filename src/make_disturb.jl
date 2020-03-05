function make_disturb(w::Matrix{Float64},r_index::Matrix{Int64},modif::Matrix{Float64})
    snum = size(w)[1]
    dnum = size(modif)[2]
    rnum = size(modif)[1]
    disturb = zero(Array{Float64}(undef, snum, snum, dnum))
    for k in 1:dnum
        disturb[:,:,k] = w
        for i in 1:rnum
            disturb[r_index[i,1],r_index[i,2],k] = modif[i,k]
        end
    end
    return(disturb)
end
