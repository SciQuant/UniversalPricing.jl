module UniversalPricing

using UniversalDynamics
using StochasticDiffEq
using Statistics

include("securities.jl")
export remake

abstract type ExpectedValueEstimate end

include("mc_estimate.jl")
export MCExpectation, 𝔼

end
