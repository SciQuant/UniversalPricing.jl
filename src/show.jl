
function Base.summary(::MonteCarloExpectedValueEstimate)
    return "Expected value estimate by Monte Carlo method:"
end

function Base.show(io::IO, x::MonteCarloExpectedValueEstimate)
    ps = 7
    println(io, summary(x))
    println(io, rpad(" mean: ", ps), x.μ)
    print(io,   rpad(" stdv: ", ps), x.σ)
end
