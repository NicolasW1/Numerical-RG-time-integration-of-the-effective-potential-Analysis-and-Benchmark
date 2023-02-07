#############################################################################
# File contains analysis of time step controller choice
# Aim: Check which time step controller work / do not work
#############################################################################

using DifferentialEquations, Symbolics, ProgressMeter, DelimitedFiles
using LinearAlgebra, SparseArrays, BenchmarkTools, JSON

include("auxiliaries.jl")
include("kernels_scalar.jl")
include("run_util.jl")

######################################
r_params = params_paper_1
r_kernel = kernel_standard_NOALLOC!
######################################

res_path = joinpath(parentDirectory(@__DIR__), "results", "std_controller")
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
ref_result = reference_entry(r_kernel, r_params);
# save_data("discrete_x.dat", r_params.grid)
# save_data("reference_solution.dat", ref_result[end])
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
  tols = 10 .^ collect(range(-10.0, -10.0, step=-1.0))
  odeargs_list = [Dict(:abstol => atol, :reltol => rtol, :dt => 1e-4, :dtmin => 1e-10, :saveat => 0.1) for atol in tols for rtol in tols]

  prob_tbl = table_WorkPrecision(r_kernel, stepper, odeargs_list, r_params)
  res_tbl = run_table(prob_tbl, calc_ref=false, run_bench=true)
  populateWorkPrecisionResult(res_tbl, odeargs_list)
end

run_save(tag, method) = save_result(tag, generateSet(method))

######################### Run here #########################

run_save("TRBDF2_default", :(TRBDF2()))
run_save("TRBDF2_PI", :(TRBDF2(controller = :PI)))
run_save("TRBDF2_PID", :(TRBDF2(controller = :PID)))
run_save("TRBDF2_Standard", :(TRBDF2(controller = :Standard)))
run_save("TRBDF2_pred", :(TRBDF2(controller = :Predictive)))
run_save("TRBDF2_I", :(TRBDF2(controller = :I)))

run_save("RadauIIA5_default", :(RadauIIA5()))
run_save("RadauIIA5_PI", :(RadauIIA5(controller = :PI)))
run_save("RadauIIA5_PID", :(RadauIIA5(controller = :PID)))
run_save("RadauIIA5_Standard", :(RadauIIA5(controller = :Standard)))
run_save("RadauIIA5_pred", :(RadauIIA5(controller = :Predictive)))
run_save("RadauIIA5_I", :(RadauIIA5(controller = :I)))

run_save("QNDF_default", :(QNDF()))
run_save("QNDF_PI", :(QNDF(controller = :PI)))
run_save("QNDF_PID", :(QNDF(controller = :PID)))
run_save("QNDF_Standard", :(QNDF(controller = :Standard)))
run_save("QNDF_pred", :(QNDF(controller = :Predictive)))
run_save("QNDF_I", :(QNDF(controller = :I)))

# doesnt have a controller option....
# run_save("Rodas4_default", :(Rodas4()))
# run_save("Rodas4_PI", :(Rodas4(controller = :PI)))
# run_save("Rodas4_PID", :(Rodas4(controller = :PID)))
# run_save("Rodas4_Standard", :(Rodas4(controller = :Standard)))
# run_save("Rodas4_pred", :(Rodas4(controller = :Predictive)))
# run_save("Rodas4_I", :(Rodas4(controller = :I)))
