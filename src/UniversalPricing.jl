module UniversalPricing

using Reexport

@reexport using UniversalDynamics

using StochasticDiffEq
using Statistics
using LsqFit
using DiffEqFlux
using GalacticOptim

include("securities.jl")
export remake

abstract type ExpectedValueEstimate end
abstract type MonteCarloExpectedValueEstimate <: ExpectedValueEstimate end

include("mc_estimate.jl")
export MCExpectation, 𝔼

include("longstaffschwartz_estimate.jl")
export callable_product_valuation, callable_libor_exotic_valuation

include("show.jl")

end
