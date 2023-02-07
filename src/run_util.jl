mutable struct ODEentry{A,P}
  stepper::Expr
  kernel!::Function
  ODEargs::A
  params::P
end
  
mutable struct AnalysisEntry{E,S}
  entry::E
  sol::S
  time::Float64
  norm::Float64
end

function generate_ODEproblem(e::ODEentry)
  u0 = init(e.kernel!, e.params)
  jac_sparsity = get_sparsity_pattern(u0, e.kernel!, e.params)

  f = ODEFunction(e.kernel!; jac_prototype=Float64.(jac_sparsity))
  ODEProblem(f, u0, (0.0, e.params.maxt), e.params)
end

function run_entry(e::ODEentry)
  problem = generate_ODEproblem(e)

  solve(problem, eval(e.stepper); e.ODEargs...)
end

function timed_run_entry(e::ODEentry)
  timed_sol = @timed run_entry(e)

  (timed_sol.value, timed_sol.time)
end

function btimed_run_entry(e::ODEentry)
  pre_sol = @timed run_entry(e)
  sol=pre_sol.value
  if sol.retcode == :Success
    btime = @belapsed(run_entry($e), samples=10, evals=1, seconds=20)
  else
    btime = pre_sol.time
  end

  (sol, btime)
end

function reference_entry(kernel!::Function, params, stepper = :(RadauIIA5()))
  println("Generating Reference : $(String(Symbol(kernel!)))")
  odeargs = Dict(:abstol => 1e-15, :reltol => 1e-15, :dt => 1e-4, :dtmax => 1e-2, :saveat => 0.1)
  entry = ODEentry(stepper, kernel!, odeargs, params)
  
  run_entry(entry)
end

function analyze_entry(e::ODEentry; ref_sol=nothing, run_bench=false)

  sol, time = run_bench ? btimed_run_entry(e) : timed_run_entry(e)

  if sol.retcode != :Success || isnothing(ref_sol)
    # slightly hacky, but currently the definition of the norm cannot produce ∞ unless the simulation crashes
    norm = Inf
  else
    norm = maximum(abs.(ref_sol(e.params.maxt)[:] .- sol(e.params.maxt)[:]))
  end
  
  AnalysisEntry(e, sol, time, norm)
end

function norm_lINF(grid, sol1, sol2)
  maximum(abs.(sol1 .- sol2))
end
function norm_lp(p, grid, sol1, sol2)
  norm = zero(eltype(sol1))
  
  norm = norm + abs(grid[2]-grid[1]) * abs(sol1[1] - sol2[1])^p
  for i=2:length(grid)-1
    norm = norm + (abs(grid[i]-grid[i+1]) + abs(grid[i]-grid[i-1]))/2 * abs(sol1[i] - sol2[i])^p
  end
  norm = norm + abs(grid[end]-grid[end-1]) * abs(sol1[end] - sol2[end])^p

  norm^(1/p)
end
norm_l1(grid, sol1, sol2) = norm_lp(1, grid, sol1, sol2)
norm_l2(grid, sol1, sol2) = norm_lp(2, grid, sol1, sol2)

function run_table(table::AbstractArray{<:ODEentry}; calc_ref=true, run_bench=false)
  kernels = unique([e.kernel! for e in table])
  params = unique([e.params for e in table])
  
  if calc_ref
    references = Dict((kernel, param) => reference_entry(kernel, param) for kernel in kernels, param in params)
  end
  
  #this is also possible as a simple array comprehension, but @showprogress destroys the array structure and only returns a simple vector
  result = similar(table, AnalysisEntry)
  @showprogress for I in CartesianIndices(table)
    if calc_ref
      result[I] = analyze_entry(table[I], ref_sol=references[(table[I].kernel!, table[I].params)], run_bench=run_bench)
    else
      result[I] = analyze_entry(table[I], ref_sol=nothing, run_bench=run_bench)
    end
  end
  
  result
end