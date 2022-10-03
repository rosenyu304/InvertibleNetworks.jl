# Normalizing layer using sigmoid function



export Normalizing


struct NormalizingLayer <: NeuralNetLayer
    ε::Float32
    _max::Float32
    _min::Float32
    logdet::Bool
end

@Flux.functor NormalizingLayer

# Constructor: Initialize with nothing
NormalizingLayer(; ε=1f-6, _max=1f0 , _min=0f0, logdet=false) = NormalizingLayer(ε, _max, _min, logdet)


# Foward pass: Input X, Output Y
function forward(X::AbstractArray{T, N}, NL::NormalizingLayer) where {T, N}

#     X = min.(max.(X, NL._min + NL.ε), NL._max - NL.ε)
    Y = SigmoidInv.(X)
    
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, NL::NormalizingLayer; eps::T=T(0)) where {T, N}
    
    X = Sigmoid.(Y)

end

# Backward pass: Input (ΔY, Y), Output (ΔY, Y)
function backward(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, NL::NormalizingLayer; set_grad::Bool=true) where {T, N}
    
end


## Jacobian-related utils

function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter}, X::AbstractArray{T, N}, NL::NormalizingLayer) where {T, N}

end

adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, NL::NormalizingLayer) where {T, N} = backward(ΔY, Y, NL; set_grad=false)

