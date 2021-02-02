module UniversalPricing

using UniversalDynamics
using Statistics


struct ExpectedValueEstimate{T<:Real}
    μ::T
    σ::T
end
ExpectedValueEstimate(x::Real, y::Real) = ExpectedValueEstimate(promote(x, y)...)
ExpectedValueEstimate(x::Real) = ExpectedValueEstimate(x, zero(x))

import Base: (==), (+), (-), (*), (/)

# TODO: hacer un test de hipotesis?
(==)(x::ExpectedValueEstimate, y::ExpectedValueEstimate) = isapprox(x.μ, y.μ, atol = x.σ + y.σ)
(==)(x::ExpectedValueEstimate, y::Real) = isapprox(x.μ, y, atol = x.σ)
(==)(x::Real, y::ExpectedValueEstimate) = (==)(y, x)

(+)(x::ExpectedValueEstimate) = ExpectedValueEstimate(+x.μ, x.σ)
(-)(x::ExpectedValueEstimate) = ExpectedValueEstimate(-x.μ, x.σ)

(+)(x::ExpectedValueEstimate, y::ExpectedValueEstimate) = ExpectedValueEstimate(x.μ + y.μ, sqrt(x.σ^2 + y.σ^2))
(-)(x::ExpectedValueEstimate, y::ExpectedValueEstimate) = ExpectedValueEstimate(x.μ - y.μ, sqrt(x.σ^2 + y.σ^2))
(*)(x::ExpectedValueEstimate, y::ExpectedValueEstimate) = ExpectedValueEstimate(x.μ * y.μ, sqrt((x.μ^2 + x.σ^2) * (y.μ^2 + y.σ^2) - x.μ^2 * y.μ^2))
/(x::ExpectedValueEstimate, y::ExpectedValueEstimate) = error("compute `1/y` `ExpectedValueEstimate` and apply multiplication instead.")

(+)(x::ExpectedValueEstimate, y::Real) = ExpectedValueEstimate(x.μ + y, x.σ)
(-)(x::ExpectedValueEstimate, y::Real) = ExpectedValueEstimate(x.μ - y, x.σ)
(*)(x::ExpectedValueEstimate, y::Real) = ExpectedValueEstimate(x.μ * y, x.σ)
(/)(x::ExpectedValueEstimate, y::Real) = ExpectedValueEstimate(x.μ / y, x.σ)

(+)(x::Real, y::ExpectedValueEstimate) = (+)(y, x)
(-)(x::Real, y::ExpectedValueEstimate) = (-)(y, x)
(*)(x::Real, y::ExpectedValueEstimate) = (*)(y, x)
(/)(x::Real, y::ExpectedValueEstimate) = error("compute `1/y` `ExpectedValueEstimate` and apply multiplication instead.")

"""
    expectation(f, u, p) -> ExpectedValueEstimate

Estimates an expected value for the payoff `f` using simulations `u` and parameters `p`.
"""
function expectation(f, u, p)
    trajectories = length(u)
    evs = zeros(trajectories)

    for n in 1:trajectories
        evs[n] = f(u[n], p)
    end

    # fair value and standard deviation
    μ = mean(evs)
    σ = stdm(evs, μ; corrected=true) / sqrt(trajectories)

    return ExpectedValueEstimate(μ, σ)
end

# alias
const 𝔼 = expectation

include("securities.jl")

end
