using OrderedCollections
using Interpolations
using UnPack

@testset "American Put Option" begin

    function american_put_test()

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

        function Discount(u, p, t, T, Tenors=nothing, n=nothing, n′=nothing)
            @unpack r = p
            return exp(-r * (T - t))
        end

        r = 0.06
        K = 40.0
        Δt = 1 / 50

        S0s = collect(36.:2.:44.)
        σs = collect(0.2:0.2:0.4)
        Ts = [1., 2.]

        results = [
            4.472, 4.821, 7.091, 8.488, 3.244, 3.735, 6.139, 7.669, 2.313, 2.879,
            5.308, 6.921, 1.617, 2.206, 4.588, 6.243, 1.118, 1.675, 3.957, 5.622
        ]

        i = 1
        for S0 in S0s, σ in σs, T in Ts
            x0 = SVector(S0)
            x = SystemDynamics(x0)

            # en LongstaffSchwartz usaron 100_000 paths, pero creo que yo tengo
            # inestabilidades al testear asi...
            dynamics = OrderedDict(:x => x)
            ds = DynamicalSystem(f, g, dynamics, (σ=σ, r=r, K=K))
            mc = montecarlo(ds, T, 10_000; alg=UniversalDynamics.EM(), seed=1, dt=Δt)

            τ = fill(Δt, Int(T/Δt))

            res = callable_product_valuation(
                mc, ds.params, AmericanPutExercise, Discount, Regressors, τ=τ
            )

            @test res.μ ≈ results[i] atol = 3*res.σ
            i += 1
        end
    end

    american_put_test()
end

@testset "Cancelable Index Amortizing Swaps" begin

    function cancelable_index_amortizing_swap_test()
        I0 = SVector(1.)
        I = SystemDynamics(I0)

        xdata = @SVector [0.04, 0.05, 0.06, 0.07]
        ydata = @SVector [4.00, 0.50, 0.10, 0.00]
        f = LinearInterpolation((xdata,), ydata, extrapolation_bc=Interpolations.Flat())

        x0 = SVector(0.002, 0.050)

        # r(t) = x(t) + y(t)
        ξ₀(t) = zero(t)
        ξ₁(t) = @SVector ones(2)

        β′ = 0.1
        η  = 1.0
        ϰ(t) = @SMatrix([
            β′  0
            0   η
        ])

        α′ = 0.0010
        γ  = 0.0525
        θ(t) = @SVector [α′/β′, γ/η]

        σ = 0.006951
        s = 0.008670
        Σ(t) = @SVector [σ, s] # since we have diagonal noise

        α(t) = @SVector ones(2)
        β(t) = @SMatrix zeros(2, 2)

        x = MultiFactorAffineModelDynamics(x0, ϰ, θ, Σ, α, β, ξ₀, ξ₁)
        B = SystemDynamics(one(eltype(x)))

        #! no estoy seguro del searchsortedfirst, pero de otra forma, si arranco en 1 siempre,
        #! voy a tener casos como P(t, 0.5) con t > 0.5! Por se motivo estoy haciendo esto.
        #! Hay que charlarlo! Sospecho que el numerador tambien deberia tener en cuenta algo
        #! acerca de que el swap se va acortando. Aunque creo que esta bien.
        function CMS(P, t, T)
            return 2 * (1 - P(t, 10.)) / sum(P(t, i/2) for i in searchsortedfirst(T, t):20)
        end

        function drift(u, p, t)
            @unpack x_dynamics, I_security, x_security, B_security = p
            @unpack f, T = p

            I = remake(I_security, u)
            x = remake(x_security, u)
            B = remake(B_security, u)

            IR = FixedIncomeSecurities(x_dynamics, x, B)

            dI = -f(CMS(IR.P, t, T))
            dx = UniversalDynamics.drift(x(t), UniversalDynamics.parameters(x_dynamics), t)
            dB = IR.r(t) * B(t)

            return vcat(dI, dx, dB)
        end

        function diffusion(u, p, t)
            @unpack x_dynamics, x_security = p

            x = remake(x_security, u)

            dI = zero(eltype(u))
            dx = UniversalDynamics.diffusion(x(t), UniversalDynamics.parameters(x_dynamics), t)
            dB = zero(eltype(u))

            return vcat(dI, dx, dB)
        end

        dynamics = OrderedDict(:I => I, :x => x, :B => B)
        ds = DynamicalSystem(drift, diffusion, dynamics, (f=f, T=[i/2 for i in 1:20]))
        sol = solve(ds, 5., alg=UniversalDynamics.EM(), dt=0.01, seed=1)
        sol = solve(ds, 5., alg=UniversalDynamics.SRIW1(), seed=1)
        mc = montecarlo(ds, 5., 2; alg=UniversalDynamics.SRIW1(), seed=1)

    end

    cancelable_index_amortizing_swap_test()
end
