function Make_disturb(W::Matrix{Float64},R_index::Matrix{Int64},Modif::Matrix{Float64})
    snum = size(W)[1]
    dnum = size(Modif)[2]
    rnum = size(Modif)[1]
    disturb = zero(Array{Float64}(undef, snum, snum, dnum))
    for k in 1:dnum
        disturb[:,:,k] = W
        for i in 1:rnum
            disturb[R_index[i,1],R_index[i,2],k] = Modif[i,k]
        end
    end
    return(disturb)
end
