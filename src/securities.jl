import UniversalDynamics: Security, remake

function remake(::Security{Di,Df,Mi,Mf}, u::UniversalDynamics.StochasticDiffEq.RODESolution) where {Di,Df,Mi,Mf}
    return Security{Di,Df,Mi,Mf}(u, t -> u(t; idxs=Di:Df), nothing, nothing)
end

(s::Security{D,D,M,M,U})(t::Real) where {D,M,U<:UniversalDynamics.StochasticDiffEq.RODESolution} = s.x(t)[]
(s::Security{Di,Df,Mi,Mf,U})(t::Real) where {Di,Df,Mi,Mf,U<:UniversalDynamics.StochasticDiffEq.RODESolution} = s.x(t)
