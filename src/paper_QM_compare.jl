using DifferentialEquations, Symbolics, ProgressMeter
using LinearAlgebra, SparseArrays, BenchmarkTools, DelimitedFiles

include("auxiliaries.jl")
include("kernels_QM.jl")
include("run_util.jl")

res_path = joinpath(parentDirectory(@__DIR__), "results", "compare")

r_params = params_paper_QM;

odeargs = Dict(:dt => 1.0e-4, :saveat => 0.005, :abstol => 1e-15, :reltol => 1e-14)

s_u0 = init(kernel_standard_NOALLOC!, r_params)

s_jac_sparsity = get_sparsity_pattern(s_u0, kernel_standard_NOALLOC!, r_params);
s_f = ODEFunction(kernel_standard_NOALLOC!; jac_prototype=Float64.(s_jac_sparsity));
s_prob = ODEProblem(s_f, s_u0, (0., r_params.maxt), r_params)
s_sol = solve(s_prob, RadauIIA5(); odeargs...);

# open(joinpath(res_path, "QM_std_sol" * ".dat"), "w") do io
#   writedlm(io, s_sol)
# end
