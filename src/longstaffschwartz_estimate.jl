
struct LongstaffSchwartzEstimate <: ExpectedValueEstimate
    μ::T
    σ::T
end

const LongstaffSchwartzExpectation = LongstaffSchwartzEstimate

# τ: representa los delta t comenzando en t = 0 de la Tenor Structure
# ExerciseValue: function ExerciseValue(u, p, Tenors, n) # Tenors, n representa t pero evitamos hacer una busqueda con ello
# Discount: function Discount(u, p, Tenors, n), pero por ahora uso Discount(t, T). Luego Discount.(u, Ref(p), Ref(Tenors), n) me retorna un vector de dimension K y ese multiplico .* a Q[n+1,:]
function LongstaffSchwartzExpectation(ExerciseValue, Discount, Regressors, τ, u, p)

    # todavia no se de donde lo voy a sacar
    T = eltype(τ)

    # estructura de tenors
    Tenors = UniversalDynamics.tenor_structure(τ)

    @show Tenors

    # N = length(Tenors)
    # K = length(u) # number of paths
    N, K = size(u)

    #! Dos ideas a implementar:
    # 1. si tengo que hacer una matriz con 1 indicadores para cada path, lo mejor es hacer
    #    un vector que me diga en que posicion de la Tenor Structure va el 1.


    # 2. Si tengo que hacer una regresion sin considerar los valores out of the money, lo
    #    que hago es allocar un vector de dimension K una unica vez y luego rellenarlo solo
    #    con los valores que vamos a usar para la regresion.
    x = zeros(T, K)
    y = zeros(T, K)


    # CLEs:
    # 1. Al parecer, el algoritmo para CLEs no elimina los casos con ejercicio out of the
    #    money porque nunca pasa eso dado que el underlying instrument es un Swap y eso
    #    siempre tiene valor positivo. Ah no, si puede ser negativo, tire cualquiera.
    # 2. La ultima fecha no tiene ejercicio, pero en un caso normal si...


    # exercise dates
    Tₑ = @view Tenors[2:N-1]


    # hold values from Tenors[1] to Tenors[end-1]
    H = Matrix{T}(undef, N, K)
    H[N-1,:] .= zeros(T, K) # H[N] nunca se usa, ver si hago algo al respecto

    # exercise values from Tenors[begin+1] to Tenors[end]
    U = Matrix{T}(undef, N, K)
    U[N,:] .= zeros(T, K) # U[0] nunca se usa, ver si hago algo al respecto

    Q = Matrix{T}(undef, N, K)

    V = Vector{T}(undef, K)

    # if the exercise value has a closed form solution for its expectation
    for k in 1:K
        # uk = getindex(u, k)
        uk = getindex(u, :, k)
        for n in 2:N # for n in 2:N-1
            U[n,k] = ExerciseValue(uk, p, Tenors, n)
        end
    end

    # explanatory variables
    ζ = Matrix{T}(undef, N, K) # en el caso general, esto apunta a un vector
    for k in 1:K
        # uk = getindex(u, k)
        uk = getindex(u, :, k)
        for n in 2:N-1
            # en el caso general, el usuario entrega una funcion que retorna un vector de
            # dimension `q` con las variables explanatorias dadas las simulaciones del trial
            # k. Ahora igual estoy considerando una unica explanatory variable
            ζ[n,k] = Regressors(uk, p, Tenors, n) # caso general
            # ζ[n,k] = ExerciseValue(uk, p, Tenors, n) # para este caso, un bermudan swaption, el regresor es el exercise value. Para un put es el stock price por ej
        end
    end

    @. f(x, p) = p[1] + p[2] * x + p[3] * x^2
    param = [0.1, 0.1, 0.1]

    for n in N:-1:2 # for n in N-1:-1:2

        if isequal(n, N) # isequal(n, N-1)

            H[n,:] .= zero(T) # it already has been set to zero, but do it anyway (remove above)
            Q[n,:] .= max.(@view(U[n,:]), @view(H[n,:]))

        else

            # tenemos que construir un interpolador para el hold value por fecha de ejercicio
            i = 0
            fill!(x, zero(T))
            fill!(y, zero(T))
            for k in 1:K
                # podriamos pasar una funcion para ver si estan on the money
                if U[n,k] > zero(T)
                    i += 1
                    x[i] = ζ[n, k]
                    y[i] = Discount(p, Tenors[n], Tenors[n+1]) * Q[n+1,k]
                end
            end
            x′ = @view x[1:i]
            y′ = @view y[1:i]
            @show x′
            @show y′
            HoldValue = curve_fit(f, x′, y′, param; autodiff=:forwarddiff)
            @show HoldValue.param

            # if n == N-2
            #     plots = plot(x′, x -> f(x, HoldValue.param))
            # else
            #     plot!(x′, x -> f(x, HoldValue.param))
            # end


            # sin filtrar
            # x = @view ζ[n, :]
            # y = Discount(p, Tenors[n], Tenors[n+1]) * Q[n+1,:]
            # HoldValue = curve_fit(f, x, y, param; autodiff=:forwarddiff)

            for k in 1:K
                H[n,k] = getindex(f(ζ[n,k], HoldValue.param), 1)
                Q[n,k] = max(U[n,k], H[n,k])
            end

            @show U[n,:]
            @show H[n,:]
            @show Q[n,:]

        end
    end

    # Q(0)
    Q[1,:] .= Discount(p, Tenors[1], Tenors[2]) * Q[2,:]

    μ = mean(Q[1,:])
    σ = stdm(Q[1,:], μ; corrected=true) / sqrt(K)

    return ζ, U, H, Q, μ, σ
end

