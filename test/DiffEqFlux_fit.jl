using DiffEqFlux
using GalacticOptim

function run_test(x, y, layer, atol)

    # data_train_vals = [rand(length(layer.model)) for k in 1:500]
    # data_train_fn = f.(data_train_vals)

    data_train_vals = x
    data_train_fn = y

    function loss_function(θ)
        data_pred = [layer(x, θ) for x in data_train_vals]
        loss = sum(norm.(data_pred.-data_train_fn))/length(data_train_fn)
        return loss
    end

    function cb(p,l)
        @show l
        return false
    end

    optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_function(x), GalacticOptim.AutoZygote())
    optprob = GalacticOptim.OptimizationProblem(optfunc, layer.p)
    res = GalacticOptim.solve(optprob, ADAM(0.1), cb=cb, maxiters = 100)
    # optprob = GalacticOptim.OptimizationProblem(optfunc, res.minimizer)
    # res = GalacticOptim.solve(optprob, ADAM(0.01), cb=cb, maxiters = 100)
    # optprob = GalacticOptim.OptimizationProblem(optfunc, res.minimizer)
    # res = GalacticOptim.solve(optprob, BFGS(), cb=cb, maxiters = 200)
    # opt = res.minimizer

    # data_validate_vals = [rand(length(layer.model)) for k in 1:100]
    # data_validate_fn = f.(data_validate_vals)

    # data_validate_pred = [layer(x,opt) for x in data_validate_vals]

    # return sum(norm.(data_validate_pred.-data_validate_fn))/length(data_validate_fn) < atol
end

x = [1.08, 1.07, 0.97, 0.77, 0.84]
y = [[0.00, 0.70, 0.18, 0.20, 0.09] .* 0.94176]

layer = TensorLayer([PolynomialBasis(3)], 1)
@test run_test(x, y , layer, 0.05)






function run_test(x, y, layer, atol)

    # data_train_vals = rand(500)
    # data_train_fn = f.(data_train_vals)

    data_train_vals = x
    data_train_fn = y

    function loss_function(θ)
        data_pred = [layer(x, θ)[1] for x in data_train_vals]
        # data_pred = [layer2(x, θ) for x in data_train_vals]
        loss = sum((data_pred - data_train_fn) .^ 2)
        @show θ
        return loss
    end

    @show data_train_vals
    @show data_train_fn

    function cb(p,l)
        @show l
        return false
    end

    optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_function(x), GalacticOptim.AutoZygote())
    optprob = GalacticOptim.OptimizationProblem(optfunc, layer.p)
    optprob = GalacticOptim.OptimizationProblem(optfunc, zeros(3)) # si es layer2
    res = GalacticOptim.solve(optprob, NelderMead(), cb=cb, maxiters = 10000)

    # optprob = GalacticOptim.OptimizationProblem(optfunc, res.minimizer)
    # res = GalacticOptim.solve(optprob, ADAM(0.1), cb=cb, maxiters = 100)
    # opt = res.minimizer

    # data_validate_vals = rand(100)
    # data_validate_fn = f.(data_validate_vals)

    # data_validate_pred = [layer(x,opt) for x in data_validate_vals]

    # output = sum(abs.(data_validate_pred.-data_validate_fn))/length(data_validate_fn)
    # @show output
    # return output < atol
end

##test 01: affine function, Linear Interpolation
a, b = rand(2)
f = x -> a*x + b
layer = SplineLayer((0.0,1.0),0.01,QuadraticInterpolation)
@test run_test(f, layer, 0.1)

x = [1.08, 1.07, 0.97, 0.77, 0.84]
y = [0.00, 0.07, 0.18, 0.20, 0.09] .* 0.94176

idxs = sortperm(x)

x = x[idxs]
y = y[idxs]

# podria tambien escribir una funcion f(x, p) directamente!
layer = TensorLayer([PolynomialBasis(3)], 1)
layer2(x, p) =  p[1] + p[2] * x + p[3] * x ^ 2
# por otro lado, creo que tambien si tengo regresores dados por funciones por ej. un IRS value,
# aca puedo ponerlos en el vector de la tensor layer y wow...

res = run_test(x, y, layer2, 1e-7)


bookf(x) = -1.070 + 2.983 * x -1.813 * x^2

scatter(x, y)
plot!(x, x -> layer(x, res.minimizer)[1])
plot!(x, bookf)

layer.(x, Ref(res.minimizer))
round.(bookf.(x), digits=4)


using LsqFit

@. g(x, p) = p[1] + p[2] * x + p[3] * x ^ 2
xdata = x
ydata = y
p0 = [0.1, 0.1, 0.1]
fit_g = curve_fit(g, xdata, ydata, p0; autodiff=:forwarddiff)

# o si lo hago multiparametrico
@. h(x, p) = p[1] + p[2] * x[:,1] + p[3] * x[:,2]
xdata = hcat(x, x.^2)
ydata = y
p0 = [0.1, 0.1, 0.1]
fit_h = curve_fit(h, xdata, ydata, p0; autodiff=:forwarddiff)

fit_g.param == fit_h.param