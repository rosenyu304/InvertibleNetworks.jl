# Normalizing layer using sigmoid function


export Normalizing, reset!


struct NormalizingLayer <: NeuralNetLayer
    ε::Parameter
    max::Parameter
    min::Parameter
    logdet::Bool
end

@Flux.functor NormalizingLayer

# Constructor: Initialize with nothing
function NormalizingLayer(nx::Int64, ny::Int64, nc::Int64; logdet=false)


end

# Foward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, NL::NormalizingLayer) where {T, N}

    
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, NL::NormalizingLayer; eps::T=T(0)) where {T, N}
    
end

# Backward pass: Input (ΔY, Y), Output (ΔY, Y)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, NL::NormalizingLayer; set_grad::Bool=true) where {T, N}
    
end


## Jacobian-related utils

function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter}, X::AbstractArray{T, N}, NL::NormalizingLayer) where {T, N}

end

adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, NL::NormalizingLayer) where {T, N} = backward(ΔY, Y, NL; set_grad=false)

