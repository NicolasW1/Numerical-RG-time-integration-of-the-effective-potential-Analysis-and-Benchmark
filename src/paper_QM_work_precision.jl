#############################################################################
# File contains the calculations for work-precision diagrams
# Aim: compare performance of algorithms
#############################################################################

using DifferentialEquations, Symbolics, ProgressMeter, DelimitedFiles, JSON
using LinearAlgebra, SparseArrays, BenchmarkTools
using Sundials, LSODA, IRKGaussLegendre

include("auxiliaries.jl")
include("kernels_QM.jl")
include("run_util.jl")

######################################
r_params = params_paper_QM
r_kernel = kernel_standard_NOALLOC!
######################################

res_path = joinpath(parentDirectory(@__DIR__), "results", "QM_work_precision")
function save_data(file, data)
  open(joinpath(res_path, file), "w") do io
    writedlm(io, data)
  end
end
function save_result(name, data)
  open(joinpath(res_path, name * ".json"), "w") do io
    write(io, JSON.json(data))
  end
  nothing
end

##########################################################
ref_result_RadauIIA5 = reference_entry(r_kernel, r_params, :(RadauIIA5()));
# ref_result_KenCarp58 = reference_entry(r_kernel, r_params, :(KenCarp58()));
# ref_result_QNDF = reference_entry(r_kernel, r_params, :(QNDF()));
# ref_result_Rodas4 = reference_entry(r_kernel, r_params, :(Rodas4()));

# save_data("discrete_x.dat", r_params.grid)
# save_data("reference_solution_RadauIIA5.dat", ref_result_RadauIIA5[end])
# save_data("reference_solution_KenCarp58.dat", ref_result_KenCarp58[end])
# save_data("reference_solution_QNDF.dat", ref_result_QNDF[end])
# save_data("reference_solution_Rodas4.dat", ref_result_Rodas4[end])

ref_result = ref_result_RadauIIA5;
##########################################################

function verify_solution(entry)
  vals = entry.sol[end]

  Λ = entry.entry.params.Λ
  tmax = entry.entry.params.maxt
  k_min = Λ * exp(-tmax)

  reg_sol = k_min^2 .+ vals

  all(reg_sol .> 0) && issorted(vals) && entry.sol.retcode == :Success
end

function table_WorkPrecision(kernel, stepper, list_odeargs, params)
  [ODEentry(stepper, kernel, odeargs, params) for odeargs in list_odeargs]
end

struct WorkPrecisionResult{T}
  success::Bool
  time::T
  abstol::T
  reltol::T
  norm_l1::T
  norm_l2::T
  norm_inf::T
  eval_stats
  result::Vector{T}
end

function populateWorkPrecisionResult(res_table, odeargs)
  [WorkPrecisionResult(verify_solution(x),
    x.time, odeargs[k][:abstol],
    odeargs[k][:reltol],
    norm_l1(r_params.grid, x.sol[end], ref_result[end]),
    norm_l2(r_params.grid, x.sol[end], ref_result[end]),
    norm_lINF(r_params.grid, x.sol[end], ref_result[end]),
    x.sol.destats,
    x.sol[end])
   for (k, x) in enumerate(res_table)]
end

function generateSet(stepper)
  tols = 10 .^ collect(range(-7.0, -12.0, step=-1.0))
  odeargs_list = [Dict(:abstol => atol, :reltol => rtol, :dt => 1e-4, :dtmin => 1e-10, :saveat => 0.1) for atol in tols for rtol in tols]

  prob_tbl = table_WorkPrecision(r_kernel, stepper, odeargs_list, r_params)
  res_tbl = run_table(prob_tbl, calc_ref=false, run_bench=true)
  populateWorkPrecisionResult(res_tbl, odeargs_list)
end

run_save(tag, method) = save_result(tag, generateSet(method))

######################### Run here #########################

# RadauII
run_save("RadauIIA5", :(RadauIIA5()))

# DIRK
run_save("ImplicitEuler", :(ImplicitEuler()))
run_save("TRBDF2", :(TRBDF2()))

# Rosenbrock
run_save("Rodas4", :(Rodas4()))

# Implicit Multistep
run_save("QNDF", :(QNDF()))

# external - Sundials
run_save("CVODE_BDF", :(CVODE_BDF()))
run_save("CVODE_BDF_band", :(CVODE_BDF(linear_solver=:Band, jac_upper=1, jac_lower=2)))
