import Base: (==), (+), (-), (*), (/)

struct MCExpectation{T<:Real} <: MonteCarloExpectedValueEstimate
    μ::T
    σ::T
end

# alias
const 𝔼 = MCExpectation

MCExpectation(x::Real, y::Real) = MCExpectation(promote(x, y)...)
MCExpectation(x::Real) = MCExpectation(x, zero(x))

"""
    MCExpectation(f, sol, p) -> MCExpectation

Computes a Monte Carlo estimation for the expectation of the random variable `f(u, p)` using
realizations `u` and parameters `p`.
"""
function MCExpectation(f, sol::EnsembleSolution, p)
    trajectories = length(sol)
    evs = zeros(trajectories)

    for n in 1:trajectories
        evs[n] = f(sol[n], p)
    end

    # fair value and standard deviation
    μ = mean(evs)
    σ = stdm(evs, μ; corrected=true) / sqrt(trajectories)

    return MCExpectation(μ, σ)
end

# TODO: should we do a Statistical hypothesis testing instead?
(==)(x::MCExpectation, y::MCExpectation) = isapprox(x.μ, y.μ, atol = x.σ + y.σ)
(==)(x::MCExpectation, y::Real) = isapprox(x.μ, y, atol = x.σ)
(==)(x::Real, y::MCExpectation) = (==)(y, x)

(+)(x::MCExpectation) = MCExpectation(+x.μ, x.σ)
(-)(x::MCExpectation) = MCExpectation(-x.μ, x.σ)

(+)(x::MCExpectation, y::MCExpectation) = MCExpectation(x.μ + y.μ, sqrt(x.σ^2 + y.σ^2))
(-)(x::MCExpectation, y::MCExpectation) = MCExpectation(x.μ - y.μ, sqrt(x.σ^2 + y.σ^2))
(*)(x::MCExpectation, y::MCExpectation) = MCExpectation(x.μ * y.μ, sqrt((x.μ^2 + x.σ^2) * (y.μ^2 + y.σ^2) - x.μ^2 * y.μ^2))
(/)(x::MCExpectation, y::MCExpectation) = error("compute `1/y` `MCExpectation` and apply multiplication instead.")

(+)(x::MCExpectation, y::Real) = MCExpectation(x.μ + y, x.σ)
(-)(x::MCExpectation, y::Real) = MCExpectation(x.μ - y, x.σ)
(*)(x::MCExpectation, y::Real) = MCExpectation(x.μ * y, x.σ)
(/)(x::MCExpectation, y::Real) = MCExpectation(x.μ / y, x.σ)

(+)(x::Real, y::MCExpectation) = (+)(y, x)
(-)(x::Real, y::MCExpectation) = (-)(y, x)
(*)(x::Real, y::MCExpectation) = (*)(y, x)
(/)(x::Real, y::MCExpectation) = error("compute `1/y` `MCExpectation` and apply multiplication instead.")
