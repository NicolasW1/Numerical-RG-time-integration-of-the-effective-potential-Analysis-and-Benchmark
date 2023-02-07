using DifferentialEquations, Symbolics, ProgressMeter
using LinearAlgebra, SparseArrays, BenchmarkTools, DelimitedFiles

include("auxiliaries.jl")
include("kernels_scalar.jl")
include("run_util.jl")

res_path = joinpath(parentDirectory(@__DIR__), "results", "compare")

r_params = params_paper_1;

odeargs = Dict(:dt => 1.0e-4, :saveat => 0.005, :abstol => 1e-15, :reltol => 1e-14)

s_u0 = init(kernel_standard_NOALLOC!, r_params)
m_u0 = init(kernel_mass_NOALLOC!, r_params)
l_u0 = init(kernel_log_NOALLOC!, r_params)

s_jac_sparsity = get_sparsity_pattern(s_u0, kernel_standard_NOALLOC!, r_params);
m_jac_sparsity = get_sparsity_pattern(m_u0, kernel_mass_NOALLOC!, r_params);
l_jac_sparsity = get_sparsity_pattern(l_u0, kernel_log_NOALLOC!, r_params);

s_f = ODEFunction(kernel_standard_NOALLOC!; jac_prototype=Float64.(s_jac_sparsity));
m_f = ODEFunction(kernel_mass_NOALLOC!; jac_prototype=Float64.(m_jac_sparsity));
l_f = ODEFunction(kernel_log_NOALLOC!; jac_prototype=Float64.(l_jac_sparsity));

s_prob = ODEProblem(s_f, s_u0, (0., r_params.maxt), r_params)
m_prob = ODEProblem(m_f, m_u0, (0., r_params.maxt), r_params)
l_prob = ODEProblem(l_f, l_u0, (0., r_params.maxt), r_params)

s_sol = solve(s_prob, RadauIIA5(); odeargs...);
m_sol = solve(m_prob, RadauIIA5(); odeargs...);
l_sol = solve(l_prob, RadauIIA5(); odeargs...);

s_sol.destats
m_sol.destats
l_sol.destats

function s_to_m(sol, index)
  k = exp(-sol.t[index]) * r_params.Λ

  grid = r_params.grid
  u = sol[index]
  msq = similar(u)

  for i in eachindex(grid)
    dudx = downwindDifference(u, i, grid)
    msq[i] = u[i] + 2 * grid[i] * dudx
  end

  msq
end
function s_to_l(sol, index)
  ksq = (exp(-sol.t[index]) * r_params.Λ)^2

  grid = r_params.grid
  u = sol[index]
  log_msq = similar(u)

  for i in eachindex(grid)
    dudx = downwindDifference(u, i, grid)
    log_msq[i] = log(u[i] + 2 * grid[i] * dudx + ksq)
  end

  log_msq
end

function l_to_m(sol, index)
  ksq = (exp(-sol.t[index]) * r_params.Λ)^2
  γ = sol[index]
  exp.(γ) .- ksq
end
function m_to_l(sol, index)
  ksq = (exp(-sol.t[index]) * r_params.Λ)^2
  msq = sol[index]
  log.(msq .+ ksq)
end

# begin
#   open(joinpath(res_path, "rho_grid" * ".dat"), "w") do io
#     writedlm(io, r_params.grid)
#   end
#   open(joinpath(res_path, "t_grid" * ".dat"), "w") do io
#     writedlm(io, s_sol.t)
#   end
#   open(joinpath(res_path, "standard_sol" * ".dat"), "w") do io
#     writedlm(io, s_sol)
#   end
#   open(joinpath(res_path, "mass_sol" * ".dat"), "w") do io
#     writedlm(io, m_sol)
#   end
#   open(joinpath(res_path, "log_sol" * ".dat"), "w") do io
#     writedlm(io, l_sol)
#   end
# end

