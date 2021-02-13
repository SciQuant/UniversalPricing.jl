using UniversalDynamics
using UniversalPricing
using OrderedCollections
using UnPack
using Test

x0 = @SVector [90.]
x = SystemDynamics(x0)

function f(u, p, t)
    @unpack x_security = p
    @unpack r, σ = p
    x = remake(x_security, u)
    return SVector(r * x(t))
end

function g(u, p, t)
    @unpack x_security = p
    @unpack σ = p
    x = remake(x_security, u)
    return SVector(σ * x(t))
end

dynamics = OrderedDict(:x => x)
ds_oop = DynamicalSystem(f, g, dynamics, (σ=0.15, r=0.03, K=100.))
sol_oop = solve(ds_oop, 1., alg=UniversalDynamics.EM(), dt=0.01, seed=1)
mc = montecarlo(ds_oop, 1., 1000; alg=UniversalDynamics.EM(), seed=1, dt=0.01)
plot(mc)

function AmericanPutExercise(u, p, t, Tenors=nothing, n=nothing)
    @unpack x_security = p
    @unpack K = p
    X = remake(x_security, u)
    return max(K - X(t), zero(K))
end

function Regressors(u, p, t, Tenors=nothing, n=nothing)
    @unpack x_security = p
    X = remake(x_security, u)
    return X(t)
end

function Discount(p, t, T, Tenors=nothing, n=nothing, n′=nothing)
    @unpack r = p
    return exp(-r * (T - t))
end

τ = fill(0.01, 100)

res = callable_product_valuation(AmericanPutExercise, Discount, Regressors, τ, mc, ds_oop.params)




function AmericanPutExercise(u, p, t, Tenors=nothing, n=nothing)
    # @unpack x_security = p
    @unpack K = p
    # X = remake(x_security, u)
    t = Tenors[n]

    return max(K - u[n], zero(K))
end

function Regressors(u, p, t, Tenors=nothing, n=nothing)
    # @unpack x_security = p
    # X = remake(x_security, u)
    # t = Tenors[n]
    return u[n]
end

function Discount(p, t, T, Tenors=nothing, n=nothing, n′=nothing)
    @unpack r = p
    return exp(-r * (T - t))
end

mc = [
    1.0  1.09  1.08  1.34
    1.0  1.16  1.26  1.54
    1.0  1.22  1.07  1.03
    1.0  0.93  0.97  0.92
    1.0  1.11  1.56  1.52
    1.0  0.76  0.77  0.9
    1.0  0.92  0.84  1.01
    1.0  0.88  1.22  1.34
]

τ = fill(1., 3)
params = (K=1.10, r = 0.06)

res = callable_product_valuation(mc, params, AmericanPutExercise, Discount, Regressors, τ=τ)



# mas ejemplos de LS
x0 = @SVector [36.]
x = SystemDynamics(x0)

function f(u, p, t)
    @unpack x_security = p
    @unpack r, σ = p
    x = remake(x_security, u)
    return SVector(r * x(t))
end

function g(u, p, t)
    @unpack x_security = p
    @unpack σ = p
    x = remake(x_security, u)
    return SVector(σ * x(t))
end

dynamics = OrderedDict(:x => x)
ds_oop = DynamicalSystem(f, g, dynamics, (σ=0.40, r=0.06, K=40.))
mc = montecarlo(ds_oop, 1., 1000; alg=UniversalDynamics.EM(), seed=1, dt=0.02)

function AmericanPutExercise(u, p, t, Tenors=nothing, n=nothing)
    @unpack x_security = p
    @unpack K = p
    X = remake(x_security, u)
    return max(K - X(t), zero(K))
end

function Regressors(u, p, t, Tenors=nothing, n=nothing)
    @unpack x_security = p
    X = remake(x_security, u)
    return X(t)
end

function Discount(p, t, T, Tenors=nothing, n=nothing, n′=nothing)
    @unpack r = p
    return exp(-r * (T - t))
end

τ = fill(0.02, 50)
res = callable_product_valuation(mc, ds_oop.params, AmericanPutExercise, Discount, Regressors, τ=τ)

