using UniversalDynamics
using UniversalPricing
using OrderedCollections
using UnPack
using Test

x0 = @SVector ones(1)
x = SystemDynamics(x0)

function f(u, p, t)
    @unpack x_security = p
    @unpack μ = p
    x = remake(x_security, u)
    return SVector(μ * x(t))
end

function g(u, p, t)
    @unpack x_security = p
    @unpack σ = p
    x = remake(x_security, u)
    return SVector(σ * x(t))
end

dynamics = OrderedDict(:x => x)
ds_oop = DynamicalSystem(f, g, dynamics, (μ=0.01, σ=0.15, r=0.01, K=0.95))
sol_oop = solve(ds_oop, 1., alg=UniversalDynamics.EM(), dt=0.01, seed=1)
mc = montecarlo(ds_oop, .3, 8; alg=UniversalDynamics.EM(), seed=1, dt=0.1)
plot(mc)

function AmericanPutExercise(u, p, Tenors, n)
    @unpack x_security = p
    @unpack K = p
    X = remake(x_security, u)
    t = Tenors[n]

    return max(K - X(t), zero(K))
end

function Regressors(u, p, Tenors, n)
    @unpack x_security = p
    X = remake(x_security, u)
    t = Tenors[n]
    return X(t)
end

function Discount(p, t, T)
    @unpack r = p
    return exp(-r * (T - t))
end

τ = fill(0.1, 3)

#! este algoritmo no esta diseñado para tener ejercicio en la ultima fecha, por lo que es de
#! esperar que no me vaya a dar bien si hago esto! Es para CLEs.
ζ, U, H, Q, μ, σ = LongstaffSchwartzExpectation(AmericanPutExercise, Discount, Regressors, τ, mc, ds_oop.params)