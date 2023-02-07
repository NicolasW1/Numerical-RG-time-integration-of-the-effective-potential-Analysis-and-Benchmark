#############################################################################
# File contains analysis of the Jacobian
# Aim: Structure of the Jacobian
#############################################################################

using DifferentialEquations, Symbolics, ProgressMeter, DelimitedFiles
using LinearAlgebra, SparseArrays, BenchmarkTools

include("auxiliaries.jl")
include("kernels_scalar.jl")
include("run_util.jl")

# SELECT KERNEL HERE
r_kernel! = kernel_std_NOALLOC!;
r_params = params_paper_1;

odeargs = Dict(:abstol => 1e-15, :reltol => 1e-13, :dt => 1e-4, :saveat => 0.01)
u0 = init(r_kernel!, r_params)

jac_sparsity = get_sparsity_pattern(u0, r_kernel!, r_params);
f = ODEFunction(r_kernel!; jac_prototype=Float64.(jac_sparsity), jac=jacobian(r_kernel!));
prob = ODEProblem(f, u0, (0., r_params.maxt), r_params)

sol_pre = @timed solve(prob, RadauIIA5(); odeargs...);
sol = sol_pre.value

jacs = zeros(length(u0), length(u0), length(sol));
for i in eachindex(sol)
  jac = @view jacs[:,:,i]
  jacobian(r_kernel!)(jac, sol[i], r_params, sol.t[i])
end


#use complex for log version
# ev_vec = zeros(ComplexF64, length(u0), length(sol));
ev_vec = zeros(length(u0), length(sol));
for i in eachindex(sol)
  ev_vec[:,i] = eigvals(jacs[:,:,i])
end

scatter(ev_vec[1,:])

####### Visualize ###########
res_path = joinpath(parentDirectory(@__DIR__), "results", "jacobian_analysis")
function save_data(file, data)
  open(joinpath(res_path, file), "w") do io
    writedlm(io, data)
  end
end

# save_data("discrete_time.dat", sol.t)
# save_data("discrete_x.dat", params_paper_1.grid)

## Single Jacobian (not paper relevant)
jac_plot=scatter(
          [diag(jacs[:,:,550], -1),
          diag(jacs[:,:,550]),
          diag(jacs[:,:,550], 1)],
          xlabel="Index",
          ylabel="Jacobian",
          label=["lower diagonal" "diagonal" "upper diagonal"],
          frame=:box,
          grid=:no, minorticks=true
          )

## Eigenvalues vs time
ev_logdata = log.(10, abs.(ev_vec));
ev_logdata = log.(10, abs.(ev_vec));
heatmap(ev_logdata[1:255,:])
heatmap(real.(ev_vec[1:256,:]))

# SELECT FILE TO SAFE HERE
# save_data("std_jacobian_time.dat", ev_logdata)
# save_data("mass_jacobian_time.dat", ev_logdata)
# save_data("log_jacobian_time.dat", ev_logdata)
save_data("log_jacobian_time__raw.dat", ev_vec)

open(joinpath(res_path, "jacobian_time.dat"), "w") do io
  writedlm(io, ev_logdata[1:255,:])
end

scatter(sol.t, ev_logdata[1,:])

scatter(imag.(ev_vec[:, 400]))

ev_vec