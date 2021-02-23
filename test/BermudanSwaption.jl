using UniversalDynamics
using UniversalPricing
using OrderedCollections
using UnPack
using Test

# from CONTROL VARIATES FOR CALLABLE LIBOR EXOTICS A PRELIMINARY STUDY, Jacob Buitelaar and
# Roger Lord

# function test_bermudan_swaption()
    Δ = 1/2
    τ = @SVector [Δ for i in 1:12]
    Tenors = UniversalDynamics.tenor_structure(τ)

    Φ = @SVector [NaN, 0.153, 0.143, 0.140, 0.140, 0.139, 0.138, 0.137, 0.136, 0.135, 0.134, 0.132]
    a = 0.976
    b = 2.0
    c = 1.5
    d = 0.5
    ρ∞ = 0.663

    C = 2.22 / 100

    ρ = @SMatrix [exp(abs(i - j) / 9 * log(ρ∞)) for i in 1:12, j in 1:12]
    L0 = @SVector [0.023, 0.025, 0.027, 0.027, 0.031, 0.031, 0.033, 0.034, 0.036, 0.036, 0.036, 0.038]

    # "abcd" time dependent volatility structure:
    function σt(i, t)
        Δt = Tenors[i] - t
        return Φ[i] * ((a * Δt + d) * exp(-b * Δt) + c)
    end

    function σ(t)

        # this one computes unnecesary values
        # return @SVector [σt(i, t) for i in 1:4]

        # we could use an MVector, modify and convert to SVector

        # or this method:
        # notar que podria loopear solo en los indices 2:12-1 en Terminal measure
        return SVector(ntuple(Val{12}()) do i
            if t ≤ Tenors[i]
                return σt(i, t)
            else
                return zero(eltype(L0))
            end
        end)
    end

    function f(u, p, t)
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)

        IR = FixedIncomeSecurities(L_dynamics, L)

        dL = UniversalDynamics.drift(L(t), UniversalDynamics.parameters(L_dynamics), t)

        return dL
    end

    function g(u, p, t)
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)

        IR = FixedIncomeSecurities(L_dynamics, L)

        dL = UniversalDynamics.diffusion(L(t), UniversalDynamics.parameters(L_dynamics), t)

        return dL
    end

    # exercise value at time t = Tenors[n] as a solved expectation
    function BermudanSwaptionExercise(u, p, t, Tenors=nothing, n=nothing)
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)
        IR = FixedIncomeSecurities(L_dynamics, L)

        # eventualmente solo vamos a llamar aca a FixedIncomeInstruments y tener un swap y
        # evaluar su present value. Es decir, definiremos un swap con una estructura de tenors
        # cada vez mas corta y evaluaremos su present value.

        # this is one way
        # sum(IR.P(t, Tenors[i+1]) * τ[i] * (C[i] - IR.L(i, t)) for i in n:length(Tenors)-1)

        if n == 12
            return 0.
        end

        # this is another
        res = zero(eltype(L_dynamics))
        for i in n:length(Tenors)-1
            res += IR.P(t, Tenors[i+1]) * τ[i] * (C - IR.L(i, t))
        end
        return res
    end

    Regressors = BermudanSwaptionExercise

    function Discount(u, p, t, T, Tenors=nothing, n=nothing, n′=nothing)
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)
        IR = FixedIncomeSecurities(L_dynamics, L)

        return IR.D(t, T)
    end

    τ = @SVector [Δ for i in 1:12]
    L = LiborMarketModelDynamics(L0, τ, σ, ρ, measure=Spot()) # imethod=Schlogl(true)
    dynamics = OrderedDict(:L => L)
    ds = DynamicalSystem(f, g, dynamics, nothing)

    # tengo un problema si meto varios trials y uso SRI1W()
    mc = montecarlo(ds, 6., 5_000; alg=UniversalDynamics.EM(), seed=1, dt=0.01)

    # los siguientes dos algoritmos no me estan dando lo mismo y ninguno da como en la ref.
    res = callable_libor_exotic_valuation(
        mc, ds.params, BermudanSwaptionExercise, Discount, Regressors, τ=τ
    )

    res = callable_product_valuation(
        mc, ds.params, BermudanSwaptionExercise, Discount, Regressors, τ=τ
    )


    #! tengo que simular en Spot para esta, aunque estoy medio confundido porque la realidad
    #! es que hay una expectation interna que ha sido resuelta en la Tn+1...
    function Swap(u, p)
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)
        IR = FixedIncomeSecurities(L_dynamics, L)

        res = zero(eltype(L_dynamics))
        for i in 1:length(Tenors)-1

            # fixing and payment
            Tx = Tenors[i]
            Tp = Tenors[i+1]

            res += IR.D(0, Tp) * τ[i] * (IR.L(i, Tx) - C) # IR.L(Tx, Tx, Tp) is the same
        end
        return res

        # return sum(
        #     IR.D(0, Tenors[i+1]) * τ[i] * (IR.L(i, Tenors[i]) - C) for i in 1:length(Tenors)-1
        # )

    end

    v1 = 𝔼(Swap, mc, ds.params)

    #! esta solo necesita un trial ya que en realidad la expectation esta resuelta.
    function Swap2(u, p)
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)
        IR = FixedIncomeSecurities(L_dynamics, L)

        res = zero(eltype(L_dynamics))
        for i in 1:length(Tenors)-1

            # fixing and payment
            Tx = Tenors[i]
            Tp = Tenors[i+1]

            res += IR.P(0, Tp) * τ[i] * (IR.L(i, 0) - C) # IR.L(0, Tx, Tp) is the same
        end
        return res
    end

    v2 = 𝔼(Swap2, mc, ds.params)

    # v1 y v2 deben dar lo mismo, ya lo dan!

    # esta simulamos en Spot()
    function Swaption(u, p)
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)
        IR = FixedIncomeSecurities(L_dynamics, L)

        #! tengo que escribir esto pensando que en el swaption la unica fecha de
        #! opcionalidad es solo la primera de ejercicio de un Bermudan
        T₀ = Tenors[2]

        # fixing and payment dates of the underlying IRS
        Tx = @view Tenors[2:end-1]
        Tp = @view Tenors[3:end]

        return IR.D(0, T₀) * max(
            sum(IR.P(T₀, Tp[i]) * 0.5 * (IR.L(i+1, T₀) - C) for i in 1:length(Tx)),
            zero(eltype(L_dynamics))
        )
    end

    v3 = 𝔼(Swaption, mc, ds.params)

    # en esta funcion se ve claramente que Tenors indica en realidad las fechas de opcionalidad
    # pero aca necesitamos las fechas de un underlying swap al que entramos.
    TenorStructure = collect(0.5:0.5:6.)
    function SwaptionExercise(u, p, t, Tenors=nothing, n=nothing)
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)
        IR = FixedIncomeSecurities(L_dynamics, L)

        Tx = TenorStructure[1:end-1]
        Tp = TenorStructure[2:end]

        res = zero(eltype(L_dynamics))
        for i in 1:length(Tp)
            res += IR.P(t, Tp[i]) * 0.5 * (IR.L(i+1, t) - C)
        end
        return res
    end

    # como hay solo una fecha de optionality, no necesitamos regressors
    τ = [0.5, 5.5]
    res = callable_libor_exotic_valuation(
        mc, ds.params, SwaptionExercise, Discount, nothing, τ=τ
    )

    τ = [0.5]
    res = callable_product_valuation(
        mc, ds.params, SwaptionExercise, Discount, nothing, τ=τ
    )

    #! hasta aca todos los Swaption dieron iguales

    # esta simulamos en Terminal().... mmmm no! no se puede claramente
    function Swaption2()
        @unpack L_dynamics, L_security = p

        L = remake(L_security, u)
        IR = FixedIncomeSecurities(L_dynamics, L)

        # TODO: EN ESTA ME SUENA A QUE HAY UN ERROR... COMO PUEDE SER IGUAL A LA ANTERIOR EN SPOT()?
        #! tengo que escribir esto pensando que en el swaption la unica fecha de
        #! opcionalidad es solo la primera de ejercicio de un Bermudan
        IR.D(0, T₀) * max(sum(IR.P(T₀, T[i+1]) * τ[i] * (IR.L(i+1, T₀) - c) for i in 1:N), 0.0)
    end

# end





δ = 1/4
N = 20
τ = @SVector [δ for i in 1:N]
Tenors = UniversalDynamics.tenor_structure(τ)


P₀(n) = exp(-0.05 * Tenors[n])
L₀(n) = 1/δ * (P₀(n) / P₀(n+1) - 1)

L0 = @SVector [L₀(n) for n in 1:20]

L = LiborMarketModelDynamics(L0, τ, σ, ρ, measure=Terminal())
dynamics = OrderedDict(:L => L)






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
