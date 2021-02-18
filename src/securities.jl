import UniversalDynamics: Security, remake

# TODO: es fundamental que remake no aloque y al parecer lo esta haciendo en estos casos!
function remake(::Security{Di,Df,Mi,Mf}, u::RODESolution) where {Di,Df,Mi,Mf}
    return Security{Di,Df,Mi,Mf}(u, nothing, nothing, nothing)
end

(s::Security{D,D,M,M,U})(t::Real) where {D,M,U<:RODESolution} = s.u(t; idxs=D)[]
(s::Security{Di,Df,Mi,Mf,U})(t::Real) where {Di,Df,Mi,Mf,U<:RODESolution} = s.u(t; idxs=Di:Df)

(s::Security{Di,Df,Mi,Mf,U})(i::Integer, t::Real) where {Di,Df,Mi,Mf,U<:RODESolution} = s.u(t; idxs=i)
