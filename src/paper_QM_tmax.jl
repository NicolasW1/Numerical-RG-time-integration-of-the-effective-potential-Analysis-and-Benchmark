#############################################################################
# File contains analysis of max t which can be reached at fixed step size
# Aim: stability of algorithms
#############################################################################

using DifferentialEquations, Plots, Symbolics, ProgressMeter
using LinearAlgebra, SparseArrays, BenchmarkTools, Alert, DelimitedFiles

include("auxiliaries.jl")
include("kernels_QM.jl")
include("run_util.jl")

res_path = joinpath(parentDirectory(@__DIR__), "results", "QM_tmax")

r_kernel! = kernel_standard_NOALLOC!;
r_params = params_paper_QM;

function verify_solution(u, t, integrator)
    ksq = (r_params.Λ * exp(-t))^2
    ksq + u[1] <= 0 || !issorted(u)
end
affect!(integrator) = terminate!(integrator)

u0 = init(r_kernel!, r_params)

jac_sparsity = get_sparsity_pattern(u0, r_kernel!, r_params);
f = ODEFunction(r_kernel!; jac_prototype=Float64.(jac_sparsity));
prob = ODEProblem(f, u0, (0.0, 50.0), r_params)
cb = DiscreteCallback(verify_solution, affect!)

function get_tmax(alg, Δt)
    println("alg : $(typeof(alg).name), Δt : $Δt")
    odeargs = Dict(:dt => Δt, :saveat => 0.1, :adaptive => false)
    sol = solve(prob, alg, callback=cb; odeargs...)

    sol.t[end]
end
scan_tmax(alg, Δts) = broadcast(x -> get_tmax(alg, x), Δts)

Δts_pre = 10 .^ collect(range(-0, -2, length=11))
Δts_old =  10 .^ collect(range(-2, -6, length=21))
Δts = 10 .^ collect(range(-0, -6, length=31))
Δtshort = 10 .^ collect(range(-2, -5, length=13))

function save_to_file(name, tmaxs, Δts)
    data = [Δts tmaxs]

    open(joinpath(res_path, name * ".dat"), "w") do io
        writedlm(io, data)
    end

nothing
end

function run_save(name, alg, Δts)
  tmaxs = scan_tmax(alg, Δts)

  save_to_file(name, tmaxs, Δts)
end

# RadauII
run_save("RadauIIA3", RadauIIA3(), Δtshort)
run_save("RadauIIA5", RadauIIA5(), Δtshort)

# DIRK
run_save("ImplicitEuler", ImplicitEuler(), Δtshort)
run_save("Trapezoid", Trapezoid(), Δtshort)
run_save("TRBDF2", TRBDF2(), Δtshort)
run_save("SDIRK2", SDIRK2(), Δtshort)
run_save("Kvaerno3", Kvaerno3(), Δtshort)
run_save("KenCarp3", KenCarp3(), Δtshort)
run_save("Cash4", Cash4(), Δtshort)
run_save("Hairer4", Hairer4(), Δtshort)
run_save("Hairer42", Hairer42(), Δtshort)
run_save("Kvaerno4", Kvaerno4(), Δtshort)
run_save("KenCarp4", KenCarp4(), Δtshort)
run_save("KenCarp47", KenCarp47(), Δtshort)
run_save("Kvaerno5", Kvaerno5(), Δtshort)
run_save("KenCarp5", KenCarp5(), Δtshort)
run_save("KenCarp58", KenCarp58(), Δtshort)

# Rosenbrock
run_save("ROS3P", ROS3P(), Δtshort)
run_save("Rodas3", Rodas3(), Δtshort)
run_save("RosShamp4", RosShamp4(), Δtshort)
run_save("Veldd4", Veldd4(), Δtshort)
run_save("Velds4", Velds4(), Δtshort)
run_save("GRK4T", GRK4T(), Δtshort)
run_save("GRK4A", GRK4A(), Δtshort)
run_save("Ros4LStab", Ros4LStab(), Δtshort)
run_save("Rodas4", Rodas4(), Δtshort)
run_save("Rodas42", Rodas42(), Δtshort)
run_save("Rodas4P", Rodas4P(), Δtshort)
run_save("Rodas4P2", Rodas4P2(), Δtshort)
run_save("Rodas5", Rodas5(), Δtshort)

# Rosenbrock-Wanner
run_save("Rosenbrock23", Rosenbrock23(), Δtshort)
run_save("Rosenbrock32", Rosenbrock32(), Δtshort)
run_save("ROS34PW1a", ROS34PW1a(), Δtshort)
run_save("ROS34PW1b", ROS34PW1b(), Δtshort)
run_save("ROS34PW2", ROS34PW2(), Δtshort)
run_save("ROS34PW3", ROS34PW3(), Δtshort)

# Implicit Multistep
run_save("QNDF1", QNDF1(), Δtshort)
run_save("QBDF1", QBDF1(), Δtshort)
run_save("ABDF2", ABDF2(), Δtshort)
run_save("QNDF2", QNDF2(), Δtshort)
run_save("QBDF2", QBDF2(), Δtshort)
run_save("QNDF", QNDF(), Δtshort)
run_save("QBDF", QBDF(), Δtshort)
run_save("FBDF", FBDF(), Δtshort)

# Implicit Extrapolation
run_save("ImplicitEulerExtrapolation", ImplicitEulerExtrapolation(), Δtshort)
run_save("ImplicitDeuflhardExtrapolation", ImplicitDeuflhardExtrapolation(), Δtshort)
run_save("ImplicitHairerWannerExtrapolation", ImplicitHairerWannerExtrapolation(), Δtshort)

# ExpRB
run_save("Exprb32", Exprb32(), Δtshort)
run_save("Exprb43", Exprb43(), Δtshort)

# Selection
run_save("sel_ImplicitEuler", ImplicitEuler(), Δts)
run_save("sel_QNDF", QNDF(), Δts)
run_save("sel_RadauIIA5", RadauIIA5(), Δts)
run_save("sel_TRBDF2", TRBDF2(), Δts)
run_save("sel_Rodas4", Rodas4(), Δts)
run_save("sel_ROS34PW2", ROS34PW2(), Δts)
