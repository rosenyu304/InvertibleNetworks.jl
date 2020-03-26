using InvertibleNetworks, LinearAlgebra, Test

# Input
nx = 16
ny = 16
n_in = 2
n_hidden = 4
batchsize = 2
maxiter = 2

# Observed data
nrec = 14
nt = 8
d = randn(Float32, nt, nrec, 1, batchsize)

# Modeling/imaging operator
J = randn(Float32, nt*nrec, nx*ny)

# Link function
Ψ(η) = identity(η)

# Unrolled loop
L = NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ)

# Initializations
η = randn(Float32, nx, ny, 1, batchsize)
s = randn(Float32, nx, ny, n_in-1, batchsize)

###################################################################################################

# Test invertibility
η_, s_ = L.forward(η, s, d, J)
ηInv, sInv = L.inverse(η_, s_, d, J)
@test isapprox(norm(ηInv - η)/norm(η), 0f0, atol=1e-5)
@test isapprox(norm(sInv - s)/norm(sInv), 0f0, atol=1e-5)

η_, s_ = L.inverse(η, s, d, J)
ηInv, sInv = L.forward(η_, s_, d, J)
@test isapprox(norm(ηInv - η)/norm(η), 0f0, atol=1e-5)
@test isapprox(norm(sInv - s)/norm(sInv), 0f0, atol=1e-5)

###################################################################################################

# Initializations
η = randn(Float32, nx, ny, 1, batchsize)
s = randn(Float32, nx, ny, n_in-1, batchsize)
η0 = randn(Float32, nx, ny, 1, batchsize)
s0 = randn(Float32, nx, ny, n_in-1, batchsize)
Δη = η - η0
Δs = s - s0

# Observed data
η_ = L.forward(η, s, d, J)[1]   # only need η

function loss(L, η0, s0, d, η)
    η_, s_ = L.forward(η0, s0, d, J)  # reshape
    Δη = η_ - η
    Δs = 0f0    # no "observed" s, so Δs=0
    f = .5f0*norm(Δη)^2
    Δη_, Δs_ = L.backward(Δη, Δs, η_, s_, d, J)[1:2]
    return f, Δη_, Δs_, L.L[1].C.v1.grad, L.L[1].RB.W1.grad
end

# Gradient test for input
f0, gη, gs = loss(L, η0, s0, d, η_)[1:3]
h = 0.1f0
maxiter = 6
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

print("\nGradient test loop unrolling\n")
for j=1:maxiter
    f = loss(L, η0 + h*Δη, s0 + h*Δs, d, η_)[1]
    err1[j] = abs(f - f0)
    err2[j] = abs(f - f0 - h*dot(Δη, gη) - h*dot(Δs, gs))
    print(err1[j], "; ", err2[j], "\n")
    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)


# Gradient test for weights
L = NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ); L.forward(η0, s0, d, J)
L0 = NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ)#; L0.forward(η0, s0, d, J)
L_ini = deepcopy(L0)
dv = L.L[1].C.v1.data - L0.L[1].C.v1.data   # just test for 2 parameters
dW = L.L[1].RB.W1.data - L0.L[1].RB.W1.data
f0, gη, gs, gv, gW = loss(L0, η, s, d, η_)
h = 0.01f0
maxiter = 5
err3 = zeros(Float32, maxiter)
err4 = zeros(Float32, maxiter)

print("\nGradient test loop unrolling\n")
for j=1:maxiter
    L0.L[1].C.v1.data = L_ini.L[1].C.v1.data + h*dv
    L0.L[1].RB.W1.data = L_ini.L[1].RB.W1.data + h*dW
    f = loss(L0, η, s, d, η_)[1]
    err3[j] = abs(f - f0)
    err4[j] = abs(f - f0 - h*dot(dv, gv))
    print(err3[j], "; ", err4[j], "\n")
    global h = h/2f0
end

@test isapprox(err3[end] / (err3[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err4[end] / (err4[1]/4^(maxiter-1)), 1f0; atol=1f1)

