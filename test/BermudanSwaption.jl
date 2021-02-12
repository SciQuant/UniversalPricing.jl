using UniversalDynamics
using UniversalPricing
using OrderedCollections
using UnPack
using Test


Δ = 0.25
τ = [Δ for i in 1:4]
Tenors = vcat(zero(eltype(τ)), cumsum(τ))

# esto es de prueba viendo ej. pg 32 paper copado, aunque ahi es non-diagonal noise
function σt(i, t)
    return 0.6 * exp(-0.1 * (Tenors[i] - t))
end

function σ!(u, t)

    # podria loopear solo en los indices 2:4-1 en Terminal measure
    for i in 1:4
        u[i] = zero(eltype(u))
        if t ≤ Tenors[i]
            u[i] = σt(i, t)
        end
    end

    return nothing
end

ρ = [1.0 0.2 0.2 0.2
    0.2 1.0 0.2 0.2
    0.2 0.2 1.0 0.2
    0.2 0.2 0.2 1.0]
L0 = [0.0112, 0.0118, 0.0123, 0.0127]
L = LiborMarketModelDynamics(L0, τ, σ!, ρ, measure=Terminal(), imethod=Schlogl(true))

function f!(du, u, p, t)
    @unpack L_dynamics, L_security = p

    L = UniversalDynamics.remake(L_security, u, du)

    IR = FixedIncomeSecurities(L_dynamics, L)

    UniversalDynamics.drift!(L.dx, L(t), UniversalDynamics.parameters(L_dynamics), t)

    return nothing
end

function g!(du, u, p, t)
    @unpack L_dynamics, L_security = p

    L = UniversalDynamics.remake(L_security, u, du)

    IR = FixedIncomeSecurities(L_dynamics, L)

    UniversalDynamics.diffusion!(L.dx, L(t), UniversalDynamics.parameters(L_dynamics), t)

    return nothing
end

dynamics = OrderedDict(:L => L)
ds_iip = DynamicalSystem(f!, g!, dynamics, nothing)
sol_iip = solve(ds_iip, 1., seed=1)






# exercise value at time Tn = Tenors[n] as a solved expectation
function BermudanSwaptionExercise(u, p, t, Tenors=nothing, n=nothing)
    @unpack L_dynamics, L_security = p
    @unpack τ, C = p

    L = UniversalDynamics.remake(L_security, u)
    IR = FixedIncomeSecurities(L_dynamics, L)

    # eventualmente solo vamos a llamar aca a FixedIncomeInstruments y tener un swap y
    # evaluar su present value. Es decir, definiremos un swap con una estructura de tenors
    # cada vez mas corta y evaluaremos su present value.

    Tn = Tenors[n] #! reemplazar por t
    # this is one way
    # sum(IR.P(Tn, Tenors[i+1]) * τ[i] * (C[i] - IR.L(i, Tn)) for i in n:length(Tenors)-1)

    # this is another
    res = zero(eltype(Tenors))
    for i in n:length(Tenors)-1
        res += IR.P(Tn, Tenors[i+1]) * τ[i] * (C[i] - IR.L(i, Tn))
    end
    return res
end




using StaticArrays
using OrderedCollections
using UnPack
using UniversalDynamics

include("DaiSingletonParameters_A3_1.jl")

(υ₀, θ₀, r₀, μ, ν, κ_rυ, κ, ῡ, θ̄, η, σ_θυ, σ_θr, σ_rυ, σ_rθ, ζ, α_r, β_θ) = DaiSingletonParameters()

x0 = @SVector [υ₀, θ₀, r₀]

ξ₀(t) = zero(t) # ξ₀ = zero
ξ₁(t) = @SVector [0, 0, 1]

ϰ(t) = @SMatrix([
    μ     0 0
    0     ν 0
    κ_rυ -κ κ
])
θ(t) = @SVector [ῡ, θ̄, θ̄ ]
Σ(t) = @SMatrix [
    η           0    0
    η * σ_θυ    1 σ_θr
    η * σ_rυ σ_rθ    1
]

α(t) = @SVector [0, ζ^2, α_r]
β(t) = @SMatrix [
    1   0 0
    β_θ 0 0
    1   0 0
]

x = MultiFactorAffineModelDynamics(x0, ϰ, θ, Σ, α, β, ξ₀, ξ₁; noise=NonDiagonalNoise(3))
B = SystemDynamics(one(eltype(x)))

function f(u, p, t)
    @unpack x_dynamics, x_security, B_security = p

    x = UniversalDynamics.remake(x_security, u)
    B = UniversalDynamics.remake(B_security, u)

    IR = FixedIncomeSecurities(x_dynamics, x, B)

    dx = UniversalDynamics.drift(x(t), UniversalDynamics.parameters(x_dynamics), t)
    dB = IR.r(t) * B(t)

    return vcat(dx, dB)
end

function g(u, p, t)
    @unpack x_dynamics, x_security, B_security = p

    x = UniversalDynamics.remake(x_security, u)
    B = UniversalDynamics.remake(B_security, u)

    dx = UniversalDynamics.diffusion(x(t), x_dynamics, t)
    dB = zero(eltype(u)) # @SMatrix zeros(eltype(u), 1, 1)

    return @SMatrix [dx[1,1] dx[1,2] dx[1,3]  0
                     dx[2,1] dx[2,2] dx[2,3]  0
                     dx[3,1] dx[3,2] dx[3,3]  0
                           0       0       0 dB]
end

dynamics = OrderedDict(:x => x, :B => B)
ds_oop = DynamicalSystem(f, g, dynamics, (T=1., K=0.1))
sol_oop = solve(ds_oop, 1., alg=UniversalDynamics.EM(), dt=0.01, seed=1)
mc = montecarlo(ds_oop, 1., 10; alg=UniversalDynamics.EM(), seed=1, dt=0.01)

function payoff(u, p)

    @unpack x_dynamics, x_security, B_security, T, K = p

    x = UniversalPricing.remake(x_security, u)
    B = UniversalPricing.remake(B_security, u)

    IR = FixedIncomeSecurities(x_dynamics, x, B)

    return IR.D(0, T) * max(IR.r(T) - K, zero(K))
end

𝔼(payoff, mc, ds_oop.params)
