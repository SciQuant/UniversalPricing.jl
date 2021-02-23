
using UniversalDynamics: tenor_structure

struct LongstaffSchwartzEstimate{T} <: ExpectedValueEstimate
    μ::T
    σ::T
end

const LongstaffSchwartzExpectation = LongstaffSchwartzEstimate

# ExerciseValue: function ExerciseValue(u, p, Tenors, n) # Tenors, n representa t pero
# evitamos hacer una busqueda con ello

# DiscountFactor: function DiscountFactor(u, p, Tenors, n)

# Regressors: retorna un vector con los regresores para cada exercise date y trial:
# Regressors(u, p, Tenors, n), aunque por ahora handleamos un numero

# τ: representa los delta t comenzando en t = 0 de la Tenor Structure
# u: monte carlo simulation from UniversalDynamics
# p: parameters

function least_squares_loss(layer, θ, xdata, ydata)
    ypred = [layer(x, θ)[1] for x in xdata]
    loss = sum((ypred - ydata) .^ 2)
    return loss
end

# from Longstaff and Schwartz
function callable_product_valuation(
    mc, p, ExerciseValue, DiscountFactor, Regressors; τ=nothing, Tenors=tenor_structure(τ)
)

    if isnothing(Tenors)
        throw(ArgumentError("Provide either a `Tenors` or a `τ`s structure."))
    end

    # number of exercise dates + 1 and Monte Carlo paths
    N = length(Tenors)
    K = length(mc)
    # K, N = size(mc)

    S = eltype(mc)

    # Exercise values
    U = Vector{S}(undef, K)

    # Payoff values
    V = Vector{S}(undef, K)

    # for regressions
    x = Vector{S}(undef, K)
    y = Vector{S}(undef, K)

    # explanatory variables
    ζ = Vector{S}(undef, K)
    # ζ = Matrix{S}(undef, K, Q)

    # for now use this, later it is going to be a `TensorLayer` from DiffEqFlux
    @. f(x, p) = p[1] + p[2] * x + p[3] * x^2
    param = [0.1, 0.1, 0.1]

    # layer = TensorLayer([PolynomialBasis(3)], 1)

    # loop over exercise dates
    for n in N:-1:2

        # exercise date
        t = Tenors[n]

        for k in 1:K
            U[k] = ExerciseValue(mc[k], p, t, Tenors, n)
        end

        if n == N
            for k in 1:K
                V[k] = max(U[k], zero(S)) # in case the exercise value is coded as, e.g., a IRS
            end
        else

            # previous exercise date inspected
            T = Tenors[n+1]

            # Perform regression for each exercise date considering only in the money cases
            i = 0
            for k in 1:K
                if U[k] > zero(S)
                    i += 1
                    uₖ = mc[k]
                    x[i] = ζ[k] = Regressors(uₖ, p, t, Tenors, n)
                    y[i] = DiscountFactor(uₖ, p, t, T, Tenors, n, n+1) * V[k]
                end
            end
            x′ = @view x[1:i]
            y′ = @view y[1:i]

            # optfunc = GalacticOptim.OptimizationFunction(
            #     (x, p) -> least_squares_loss(layer, x, x′, y′),
            #     GalacticOptim.AutoZygote()
            # )

            # optprob = GalacticOptim.OptimizationProblem(optfunc, layer.p)
            # HoldValue = GalacticOptim.solve(optprob, NelderMead(), maxiters=10000)

            HoldValue = curve_fit(f, x′, y′, param; autodiff=:forwarddiff)

            for k in 1:K
                Uₖ = U[k]
                Hₖ = f(ζ[k], HoldValue.param)[1]
                if Uₖ > zero(S) && Uₖ > Hₖ
                # if Uₖ > zero(S) && Uₖ > layer(ζ[k], HoldValue.minimizer)[1]
                    V[k] = Uₖ
                else
                    V[k] *= DiscountFactor(mc[k], p, t, T, Tenors, n, n+1)
                end
            end
        end
    end

    for k in 1:K
        V[k] *= DiscountFactor(mc[k], p, Tenors[1], Tenors[2], Tenors, 1, 2)
    end

    μ = mean(V)
    σ = stdm(V, μ; corrected=true) / sqrt(K)

    return LongstaffSchwartzEstimate{S}(μ, σ)
end

# from Andersen and Piterbarg:

# Entiendo que las unicas diferencias con el algoritmo de arriba son:
# 1. La fecha N no se analiza ya que los callable libor exotics tienen ejercicio nulo en ese
#    momento (no restan accruals).
# 2. En N-1 comparamos el ejercicio con zero ya que en N el producto vale cero.
# 3. No hay filtrado para hacer las regresiones.
# Por lo tanto, un Bermudan Swaption deberia dar lo mismo tanto con este algoritmo como con
# el anterior considerando que remuevo las filtraciones y que la funcion ExerciseValue retorne
# zero en la fecha N.

function callable_libor_exotic_valuation(
    mc, p, ExerciseValue, DiscountFactor, Regressors; τ=nothing, Tenors=tenor_structure(τ)
)

    if isnothing(Tenors)
        throw(ArgumentError("Provide either a `Tenors` or a `τ`s structure."))
    end

    # number of exercise dates + 1 and Monte Carlo paths
    N = length(Tenors)
    K = length(mc)

    S = eltype(mc)

    # Exercise values
    U = Vector{S}(undef, K)

    # Payoff values
    V = Vector{S}(undef, K)

    # for regressions
    y = Vector{S}(undef, K)

    # explanatory variables
    ζ = Vector{S}(undef, K)
    # ζ = Matrix{S}(undef, K, Q)

    # for now use this, later it is going to be a `TensorLayer` from DiffEqFlux
    @. f(x, p) = p[1] + p[2] * x + p[3] * x^2
    param = [0.1, 0.1, 0.1]

    # loop over exercise dates
    for n in N-1:-1:2

        # exercise date
        t = Tenors[n]

        for k in 1:K
            U[k] = ExerciseValue(mc[k], p, t, Tenors, n)
        end

        if n == N-1
            for k in 1:K
                V[k] = max(U[k], zero(S))
            end
        else

            # previous exercise date inspected
            T = Tenors[n+1]

            # Perform a regression for each exercise date.
            # Note that Andersen doesn't apply any kind of filtering.
            for k in 1:K
                uₖ = mc[k]
                ζ[k] = Regressors(uₖ, p, t, Tenors, n)
                y[k] = DiscountFactor(uₖ, p, t, T, Tenors, n, n+1) * V[k]
            end

            HoldValue = curve_fit(f, ζ, y, param; autodiff=:forwarddiff)

            for k in 1:K
                Uₖ = U[k]
                Hₖ = f(ζ[k], HoldValue.param)[1]
                V[k] = max(Uₖ, Hₖ)
            end
        end
    end

    for k in 1:K
        V[k] *= DiscountFactor(mc[k], p, Tenors[1], Tenors[2], Tenors, 1, 2)
    end

    μ = mean(V)
    σ = stdm(V, μ; corrected=true) / sqrt(K)

    return LongstaffSchwartzEstimate{S}(μ, σ)
end