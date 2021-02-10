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
function BermudanSwaptionExercise(u, p, Tenors, n)
    @unpack L_dynamics, L_security = p
    @unpack τ, C = p

    L = UniversalDynamics.remake(L_security, u)
    IR = FixedIncomeSecurities(L_dynamics, L)

    # eventualmente solo vamos a llamar aca a FixedIncomeInstruments y tener un swap y
    # evaluar su present value. Es decir, definiremos un swap con una estructura de tenors
    # cada vez mas corta y evaluaremos su present value.

    Tn = Tenors[n]
    # this is one way
    # sum(IR.P(Tn, Tenors[i+1]) * τ[i] * (C[i] - IR.L(i, Tn)) for i in n:length(Tenors)-1)

    # this is another
    res = zero(eltype(Tenors))
    for i in n:length(Tenors)-1
        res += IR.P(Tn, Tenors[i+1]) * τ[i] * (C[i] - IR.L(i, Tn))
    end
    return res
end