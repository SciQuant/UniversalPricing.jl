
struct LongstaffSchwartzEstimate <: ExpectedValueEstimate
    μ::T
    σ::T
end

const LongstaffSchwartzExpectation = LongstaffSchwartzEstimate

# τ: representa los delta t comenzando en t = 0
function LongstaffSchwartzExpectation(τ, Ex, u, p)

    # estructura de tenors
    Tenors = UniversalDynamics.tenor_structure(τ)

    # exercise dates
    Tₑ = @view Tenors[2:N-1]

    N = length(Tenors)
    K = length(u) # number of paths

    # hold values from Tenors[1] to Tenors[end-1]
    H = Matrix{T}(undef, N, K)
    H[N-1] = zero(eltype(Tenors)) # H[N] nunca se usa, ver si hago algo al respecto

    # exercise values from Tenors[begin+1] to Tenors[end]
    U = Matrix{T}(undef, N, K)
    U[N] = zero(eltype(Tenors)) # U[0] nunca se usa, ver si hago algo al respecto

    Q = Matrix{T}(undef, N, K)

    V = Vector{T}(undef, K)

    # if the exercise value has a closed form solution for its expectation
    for k in 1:K
        uk = getindex(u, k)
        for n in 2:N-1
            U[n,k] = Exercise(uk, p, Tenors, n)
        end
    end

    # explanatory variables
    ζ = Matrix{T}(undef, N, K)
    for k in 1:K
        uk = getindex(u, k)
        for n in 2:N-1
            # en el caso general, el usuario entrega una funcion que retorna un vector de
            # dimension `q` con las variables explanatorias dadas las simulaciones del trial
            # k. Ahora igual estoy considerando una unica explanatory variable
            # ζ[n,k] = Variables(uk, p, Tenors, n) # caso general
            ζ[n,k] = Exercise(uk, p, Tenors, n) # para este caso, un bermudan swaption
        end
    end


    # leer el paper de piterbarg con el de LS e ir haciendo las cosas en papel primero.


    for k in 1:K
        uk = u[k]
        for n in N-1:2

            if n == N - 1
                U[n,k] = Exercise(uk, p, Tenors, n)
                Q[n,k] = max(U[n,k], zero(T))
            else
                U[n,k] = Exercise(uk, p, Tenors, n)

                # tenemos que construir un interpolador

                H[n,k] =


                Q[n,k] = max(U[n,k], H[n,k])
            end


        end

end


# exercise value at time Tn = Tenors[n] as a solved expectation
function Exercise(u, p, Tenors, n)
    @unpack L_dynamics, L_security = p
    @unpack τ, C = p

    L = UniversalDynamics.remake(L_security, u)
    IR = FixedIncomeSecurities(L_dynamics, L)

    # eventualmente solo vamos a llamar aca a FixedIncomeInstruments y tener un swap y
    # evaluar su present value. Es decir, definiremos un swap con una estructura de tenors
    # cada vez mas corta y evaluaremos su present value.

    Tn = Tenors[n]
    # this is one way
    # sum(IR.P(Tn, Tenors[i+1]) * τ[i] * (C[i] - IR.L(i, Tn)) for i in n:length(Tenors)-1)

    # this is another
    res = zero(eltype(Tenors))
    for i in n:length(Tenors)-1
        res += IR.P(Tn, Tenors[i+1]) * τ[i] * (C[i] - IR.L(i, Tn))
    end
    return res
end