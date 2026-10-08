mutable struct State{T<:Real}
    u::Matrix{T}
    w::Matrix{T}
    p::Matrix{T}
    rho::Matrix{T}
    eta::Vector{T}
end

function from_shape(nz::Int, nx::Int; T::Type{<:Real}=Float64)
    return State(
        zeros(T, nz, nx + 1),
        zeros(T, nz + 1, nx),
        zeros(T, nz, nx),
        zeros(T, nz, nx),
        zeros(T, nx),
    )
end

function copy_state(st::State{T}) where {T<:Real}
    return State(copy(st.u), copy(st.w), copy(st.p), copy(st.rho), copy(st.eta))
end
