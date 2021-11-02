using Flux
using Plots
using DiffEqFlux
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using SciMLBase
using GalacticOptim
import ModelingToolkit: Interval, infimum, supremum
using DomainSets

# MWE
@parameters x
@variables u(..)
domains = [x ∈ Interval(0.0,1.00)]
chain = FastChain(FastDense(1,15,Flux.σ),FastDense(15,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
strategy = NeuralPDE.GridTraining(0.1)
indvars = [x]
depvars = [u(x)]
parameterless_type_θ = Float64
phi = NeuralPDE.get_phi(chain,parameterless_type_θ)

u_ = (cord, θ, phi)-> phi(cord, θ)
cord = [1.]
lb, ub = [0.], [1.]
θ = initθ

derivative = NeuralPDE.get_numeric_derivative()
integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative)

integral_f = (cord, var"##θ#332", phi, derivative, integral, u_, p)->begin
          begin
              let (x,) = (cord[[1], :],)
                  begin
                      cord1 = vcat(x)
                  end
                  u_(cord1, var"##θ#332", phi)
              end
          end
      end

# build loss
u = NeuralPDE.get_u()
_loss_function = integral_f
loss_function = (cord, θ) -> begin
    _loss_function(cord, θ, phi, derivative, integral, u, default_p)
end

# get loss
train_set = vcat(0:0.05:1...)
loss = (θ) -> mean(abs2,loss_function(train_set, θ))

# turn into optimization and solve
f = OptimizationFunction(loss_function, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f, initθ)

cb_ = function (p,l)
    println("loss: ", l , "losses: ", map(l -> l(p), loss_functions))
    return false
end

# optimizer
opt = BFGS()
res = GalacticOptim.solve(prob, opt; cb = cb_, maxiters=10)


# build integral
# Optimize weights
# compare analytical solution to trained approximation







#=
integrand_func = RuntimeGeneratedFunctions.RuntimeGeneratedFunction{(:cord, Symbol("##θ#257"), :phi, :derivative, :integral, :u, :p), NeuralPDE.var"#_RGF_ModTag", NeuralPDE.var"#_RGF_ModTag", (0x0fb6a092, 0x8bd0d0c7, 0x203e4ec1, 0x0c4e4c77, 0xa39e47b7)}(quote
    #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:586 =#
    #= /Users/gabrielbirnbaum/.julia/dev/NeuralPDE/src/pinns_pde_solve.jl:586 =#
    begin
        let (t,) = (cord[[1], :],)
            begin
                cord1 = vcat(t)
            end
            u(cord1, var"##θ#257", phi)
        end
    end
end)


## LOW LEVEL API REFRESHER

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

@parameters t, x
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

#2D PDE
eq  = Dt(u(t,x)) + u(t,x)*Dx(u(t,x)) - (0.01/pi)*Dxx(u(t,x)) ~ 0

# Initial and boundary conditions
bcs = [u(0,x) ~ -sin(pi*x),
       u(t,-1) ~ 0.,
       u(t,1) ~ 0.,
       u(t,-1) ~ u(t,1)]

# Space and time domains
domains = [t ∈ Interval(0.0,1.0),
           x ∈ Interval(-1.0,1.0)]
# Discretization
dx = 0.05
# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))
initθ = Float64.(DiffEqFlux.initial_params(chain))
eltypeθ = eltype(initθ)
parameterless_type_θ = DiffEqBase.parameterless_type(initθ)
strategy = NeuralPDE.GridTraining(dx)

phi = NeuralPDE.get_phi(chain,parameterless_type_θ)
derivative = NeuralPDE.get_numeric_derivative()

indvars = [t,x]
depvars = [u(t,x)]

integral = NeuralPDE.get_numeric_integral(strategy, indvars, depvars, chain, derivative)


_pde_loss_function = NeuralPDE.build_loss_function(eq,indvars,depvars,
                                                   phi,derivative, nothing, chain,initθ,strategy)

                                                  

bc_indvars = NeuralPDE.get_variables(bcs,indvars,depvars)
_bc_loss_functions = [NeuralPDE.build_loss_function(bc,indvars,depvars,
                                                    phi,derivative, nothing, chain,initθ,strategy,
                                                    bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs,bc_indvars)]

train_sets = NeuralPDE.generate_training_sets(domains,dx,[eq],bcs,eltypeθ,indvars,depvars)
train_domain_set, train_bound_set = train_sets

pde_loss_function = NeuralPDE.get_loss_function(_pde_loss_function,
                                                train_domain_set[1],
                                                eltypeθ,parameterless_type_θ,
                                                strategy)

bc_loss_functions = [NeuralPDE.get_loss_function(loss,set,
                                                 eltypeθ, parameterless_type_θ,
                                                 strategy) for (loss, set) in zip(_bc_loss_functions,train_bound_set)]


loss_functions = [pde_loss_function; bc_loss_functions]
loss_function__ = θ -> sum(map(l->l(θ) ,loss_functions))

function loss_function_(θ,p)
    return loss_function__(θ)
end

f = OptimizationFunction(loss_function_, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(f, initθ)

cb_ = function (p,l)
    println("loss: ", l , "losses: ", map(l -> l(p), loss_functions))
    return false
end

# optimizer
opt = BFGS()
res = GalacticOptim.solve(prob, opt; cb = cb_, maxiters=10)
=#

