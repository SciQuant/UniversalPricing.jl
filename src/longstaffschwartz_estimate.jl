
struct LongstaffSchwartzEstimate{T} <: ExpectedValueEstimate
    μ::T
    σ::T

    ζ::Matrix{T}
    U::Matrix{T}
    H::Matrix{T}
    Q::Matrix{T}
end

const LongstaffSchwartzExpectation = LongstaffSchwartzEstimate

# ExerciseValue: function ExerciseValue(u, p, Tenors, n) # Tenors, n representa t pero evitamos hacer una busqueda con ello
# DiscountFactor: function DiscountFactor(u, p, Tenors, n), pero por ahora uso DiscountFactor(p, t, T). Luego Discount.(u, Ref(p), Ref(Tenors), n) me retorna un vector de dimension K y ese multiplico .* a Q[n+1,:]
# Regressors: retorna un vector con los regresores para cada exercise date y trial: Regressors(u, p, Tenors, n), aunque por ahora handleamos un numero
# τ: representa los delta t comenzando en t = 0 de la Tenor Structure
# u: monte carlo simulation from UniversalDynamics
# p: parameters

# from Longstaff and Schwartz
function callable_product_valuation(ExerciseValue, DiscountFactor, Regressors, τ, u, p)

    T = eltype(u)

    # tenor structure
    Tenors = UniversalDynamics.tenor_structure(τ)

    N = length(Tenors)
    K = length(u) # number of paths
    # K, N = size(u)

    # exercise dates from last to first, note that the last date is included
    Te = @view Tenors[N:-1:2]

    # Hold or Continuation values goes from Tenors[1] to Tenors[N-1]
    # H[N] is never used but we keep it in order to use same indexes
    H = Matrix{T}(undef, K, N)

    # Exercise values goes from Tenors[2] to Tenors[end]
    # U[1] is never used but we keep it in order to use same indexes
    U = Matrix{T}(undef, K, N)

    # cashflow matrix
    C = Matrix{T}(undef, K, N)

    # for regressions
    x = zeros(T, K)
    y = zeros(T, K)

    # the index `n` where the option is executed for each trial, i.e. the stopping rule
    E = Vector{Int32}(undef, K)

    # by default, we take the last cashflow
    fill!(E, N)

    # assuming that the exercise value has a closed form solution for its expectation
    for k in 1:K
        uₖ = u[k] #! si es UniversalDynamics solution
        # uₖ = u[k,:]
        for n in 2:N
            U[k,n] = ExerciseValue(uₖ, p, Tenors, n)
        end
    end

    # explanatory variables
    # en el caso general, `ζ` apunta a un vector y esa dimension q la calculo antes
    ζ = Matrix{T}(undef, K, N)
    for k in 1:K
        uₖ = u[k] #! si es UniversalDynamics solution
        # uₖ = u[k,:]
        # there is no need for a regression in Tenors[N]
        for n in 2:N-1
            ζ[k,n] = Regressors(uₖ, p, Tenors, n)
        end
    end

    # for now use this, later it is going to be a `TensorLayer` from DiffEqFlux
    @. f(x, p) = p[1] + p[2] * x + p[3] * x^2
    param = [0.1, 0.1, 0.1]

    # loop over exercise dates
    for (e, n) in enumerate(N:-1:2)

        if isone(e) # n == N

            # the hold value is zero at Tenors[N]
            H[:,n] .= zero(T)

            for k in 1:K
                if U[k,n] > H[k,n]
                    E[k] = n
                    C[k,n] = U[k,n]
                else
                    C[k,n] = H[k,n]
                end
            end
        else

            # Perform a regression for each exercise date considering only in the money cases
            i = 0
            fill!(x, zero(T))
            fill!(y, zero(T))
            for k in 1:K
                # podriamos pasar una funcion para ver si estan on the money
                if U[k,n] > zero(T)
                    i += 1
                    n′ = E[k]
                    x[i] = ζ[k,n]
                    y[i] = DiscountFactor(p, Tenors[n], Tenors[n′]) * C[k,n′]
                end
            end
            x′ = @view x[1:i]
            y′ = @view y[1:i]
            HoldValue = curve_fit(f, x′, y′, param; autodiff=:forwarddiff)

            # @show x′
            # @show y′
            # @show HoldValue.param
            # @show ""

            for k in 1:K

                # solo miramos aquellos donde quizas conviene ejercer, es decir, los que no
                # filtramos en la regresion
                if U[k,n] > zero(T)

                    # hold or continuation value comes from regression
                    H[k,n] = f(ζ[k,n], HoldValue.param)[1]

                    if U[k,n] > H[k,n]
                        # si el ejercicio es mayor al holdeo, ejercito y guardo el cashflow
                        E[k] = n
                        C[k,n] = U[k,n] # podria hacer cero para todo tenor > Tenors[n]
                    else
                        # si no, holdeo y el cashflow es zero
                        #! cual de los dos va? con zero hago bien la regresion...
                        #! creo que ahora puede ir cualquiera de los dos...
                        #! mmm... quizas va C[k,n+1] * Discount(...)
                        C[k,n] = zero(T)# H[k,n]
                    end
                else
                    # conviene no ejercitar, entonces no recibo cashflow
                    C[k,n] = zero(T)
                end
            end
        end
    end

    V = Vector{T}(undef, K)
    for k in 1:K
        if E[k] == -1
            V[k] = zero(T)
        else
            n = E[k]
            V[k] = DiscountFactor(p, Tenors[1], Tenors[n]) * C[k,n]
        end
    end

    μ = mean(V)
    σ = stdm(V, μ; corrected=true) / sqrt(K)

    return LongstaffSchwartzEstimate{T}(μ, σ, ζ, U, H, C)
end

# from Andersen and Piterbarg
function callable_libor_exotic_valuation(ExerciseValue, DiscountFactor, Regressors, τ, u, p)

    T = eltype(u)

    # tenor structure
    Tenors = UniversalDynamics.tenor_structure(τ)

    N = length(Tenors)
    K = length(u) # number of paths

    # exercise dates from last to first
    Te = @view Tenors[N-1:-1:2]

    # Hold or Continuation values goes from Tenors[1] to Tenors[N-1]
    # H[N] is never used but we keep it in order to use same indexes
    H = Matrix{T}(undef, K, N)

    # Exercise values goes from Tenors[2] to Tenors[N]
    # U[1] is never used but we keep it in order to use same indexes
    U = Matrix{T}(undef, K, N)

    # Handler for max(H, U)
    Q = Matrix{T}(undef, K, N)

    # for regressions
    y = Vector{T}(undef, K)

    # assuming that the exercise value has a closed form solution for its expectation
    for k in 1:K
        uₖ = u[k]
        for n in 2:N-1
            U[k,n] = ExerciseValue(uₖ, p, Tenors, n)
        end
        # the exercise value is zero at Tenors[N]
        U[k,N] = zero(T)
    end

    # explanatory variables
    # en el caso general, `ζ` apunta a un vector y esa dimension q la calculo antes
    ζ = Matrix{T}(undef, K, N)
    for k in 1:K
        uₖ = u[k]
        for n in 2:N-1
            # en el caso general, el usuario entrega una funcion que retorna un vector de
            # dimension `q` con las variables explanatorias dadas las simulaciones del trial
            # k. Ahora igual estoy considerando una unica explanatory variable
            ζ[k,n] = Regressors(uₖ, p, Tenors, n)
        end
    end

    # for now use this, later it is going to be a `TensorLayer` from DiffEqFlux
    @. f(x, p) = p[1] + p[2] * x + p[3] * x^2
    param = [0.1, 0.1, 0.1]

    # loop over exercise dates
    for (e, n) in enumerate(N-1:-1:2)

        # T = Tenors[n]
        # Te = T[e]

        if isone(e) # n == N-1

            # the hold value is zero at Tenors[N-1]
            H[:,n] .= zero(T)
            Q[:,n] .= max.(@view(U[:,n]), @view(H[:,n]))
            # @views Q[:,n] .= max.(U[:,n], H[:,n])

        else

            # Perform a regression for each exercise date.
            # Note that Andersen do not apply any kind of filtering.
            x  = @view ζ[:,n]
            y .= DiscountFactor(p, Tenors[n], Tenors[n+1]) .* @view(Q[:,n+1]) #! in the general case, the discount is a vector
            HoldValue = curve_fit(f, x, y, param; autodiff=:forwarddiff)

            for k in 1:K
                H[k,n] = f(ζ[k,n], HoldValue.param)[1]
                Q[k,n] = max(U[k,n], H[k,n])
            end
        end
    end

    # Q(0)
    Q[:,1] .= DiscountFactor(p, Tenors[1], Tenors[2]) .* @view(Q[:,2])
    Q0 = @view Q[:,1]

    μ = mean(Q0)
    σ = stdm(Q0, μ; corrected=true) / sqrt(K)

    return LongstaffSchwartzEstimate{T}(μ, σ, ζ, U, H, Q)
end