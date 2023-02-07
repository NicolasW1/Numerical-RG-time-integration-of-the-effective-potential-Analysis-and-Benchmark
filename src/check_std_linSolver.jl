#############################################################################
# File contains analysis of linear solver choice
# Aim: Check which linear solvers work / do not work
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

res_path = joinpath(parentDirectory(@__DIR__), "results", "std_linSolve")
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

run_save("TRBDF2_standard", :(TRBDF2()))
run_save("TRBDF2_BiCGStab", :(TRBDF2(linsolve=KrylovJL_BICGSTAB())))
run_save("TRBDF2_GMRES", :(TRBDF2(linsolve=KrylovJL_GMRES())))
run_save("TRBDF2_LU", :(TRBDF2(linsolve=LUFactorization())))
# run_save("TRBDF2_QR", :(TRBDF2(linsolve=QRFactorization())))
run_save("TRBDF2_KLU", :(TRBDF2(linsolve=KLUFactorization(; reuse_symbolic=true))))
run_save("TRBDF2_UMFPack", :(TRBDF2(linsolve=UMFPACKFactorization(; reuse_symbolic=true))))

run_save("RadauIIA5_standard", :(RadauIIA5()))
# run_save("RadauIIA5_BiCGStab", :(RadauIIA5(linsolve=KrylovJL_BICGSTAB())))
# run_save("RadauIIA5_GMRES", :(RadauIIA5(linsolve=KrylovJL_GMRES())))
run_save("RadauIIA5_LU", :(RadauIIA5(linsolve=LUFactorization())))
# run_save("RadauIIA5_QR", :(RadauIIA5(linsolve=QRFactorization())))
run_save("RadauIIA5_KLU", :(RadauIIA5(linsolve=KLUFactorization(; reuse_symbolic=true))))
run_save("RadauIIA5_UMFPack", :(RadauIIA5(linsolve=UMFPACKFactorization(; reuse_symbolic=true))))

run_save("ImplicitEuler_standard", :(ImplicitEuler()))
# run_save("ImplicitEuler_BiCGStab", :(ImplicitEuler(linsolve=KrylovJL_BICGSTAB())))
# run_save("ImplicitEuler_GMRES", :(ImplicitEuler(linsolve=KrylovJL_GMRES())))
run_save("ImplicitEuler_LU", :(ImplicitEuler(linsolve=LUFactorization())))
# run_save("ImplicitEuler_QR", :(ImplicitEuler(linsolve=QRFactorization())))
run_save("ImplicitEuler_KLU", :(ImplicitEuler(linsolve=KLUFactorization(; reuse_symbolic=true))))
run_save("ImplicitEuler_UMFPack", :(ImplicitEuler(linsolve=UMFPACKFactorization(; reuse_symbolic=true))))

run_save("QNDF_standard", :(QNDF()))
run_save("QNDF_BiCGStab", :(QNDF(linsolve=KrylovJL_BICGSTAB())))
run_save("QNDF_GMRES", :(QNDF(linsolve=KrylovJL_GMRES())))
run_save("QNDF_LU", :(QNDF(linsolve=LUFactorization())))
# run_save("QNDF_QR", :(QNDF(linsolve=QRFactorization())))
run_save("QNDF_KLU", :(QNDF(linsolve=KLUFactorization(; reuse_symbolic=true))))
run_save("QNDF_UMFPack", :(QNDF(linsolve=UMFPACKFactorization(; reuse_symbolic=true))))

run_save("Rodas4_standard", :(Rodas4()))
run_save("Rodas4_BiCGStab", :(Rodas4(linsolve=KrylovJL_BICGSTAB())))
run_save("Rodas4_GMRES", :(Rodas4(linsolve=KrylovJL_GMRES())))
run_save("Rodas4_LU", :(Rodas4(linsolve=LUFactorization())))
# run_save("Rodas4_QR", :(Rodas4(linsolve=QRFactorization())))
run_save("Rodas4_KLU", :(Rodas4(linsolve=KLUFactorization(; reuse_symbolic=true))))
run_save("Rodas4_UMFPack", :(Rodas4(linsolve=UMFPACKFactorization(; reuse_symbolic=true))))

run_save("ROS34PW2_standard", :(ROS34PW2()))
run_save("ROS34PW2_BiCGStab", :(ROS34PW2(linsolve=KrylovJL_BICGSTAB())))
run_save("ROS34PW2_GMRES", :(ROS34PW2(linsolve=KrylovJL_GMRES())))
run_save("ROS34PW2_LU", :(ROS34PW2(linsolve=LUFactorization())))
# run_save("ROS34PW2_QR", :(ROS34PW2(linsolve=QRFactorization())))
run_save("ROS34PW2_KLU", :(ROS34PW2(linsolve=KLUFactorization(; reuse_symbolic=true))))
run_save("ROS34PW2_UMFPack", :(ROS34PW2(linsolve=UMFPACKFactorization(; reuse_symbolic=true))))

