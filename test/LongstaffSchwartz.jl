using OrderedCollections
using UnPack

S0s = collect(36.:2.:44.)
σs = collect(0.2:0.2:0.4)
Ts = [1., 2.]

results = [
    4.472, 4.821, 7.091, 8.488, 3.244, 3.735, 6.139, 7.669, 2.313, 2.879,
    5.308, 6.921, 1.617, 2.206, 4.588, 6.243, 1.118, 1.675, 3.957, 5.622
]

@testset "American Put Option" begin

    function f(u, p, t)
        @unpack x_security = p
        @unpack r = p
        x = remake(x_security, u)
        return SVector(r * x(t))
    end

    function g(u, p, t)
        @unpack x_security = p
        @unpack σ = p
        x = remake(x_security, u)
        return SVector(σ * x(t))
    end

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

    function test(S0s, σs, Ts, results)
        r = 0.06
        K = 40.0
        Δt = 1 / 50

        i = 1
        for S0 in S0s, σ in σs, T in Ts
            x0 = @SVector [S0]
            x = SystemDynamics(x0)

            # en LongstaffSchwartz usaron 100_000 paths, pero creo que yo tengo
            # inestabilidades al testear asi...
            dynamics = OrderedDict(:x => x)
            ds = DynamicalSystem(f, g, dynamics, (σ=σ, r=r, K=K))
            mc = montecarlo(ds, T, 10_000; alg=UniversalDynamics.EM(), seed=1, dt=Δt)

            τ = fill(Δt, Int(T/Δt))

            res = callable_product_valuation(
                AmericanPutExercise, Discount, Regressors,mc, ds.params, τ=τ
            )

            @test res.μ ≈ results[i] atol = 3*res.σ
            i += 1
        end
    end

    test(S0s, σs, Ts, results)
end