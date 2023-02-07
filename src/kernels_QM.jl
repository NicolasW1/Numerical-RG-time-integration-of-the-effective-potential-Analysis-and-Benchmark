struct ParameterQM
  grid::Vector{Float64}
  dim::Int64
  Λ::Float64
  λ::Float64
  m²::Float64
  hϕ::Float64
  μ::Float64
  maxt::Float64
end

function mesonFlux(e2_p, e2_s, k, p::ParameterQM)
  Nf = 2
  prefactor = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  return prefactor * ((Nf^2 - 1) / e2_p + 1 / e2_s)
end

function quarkFlux(rho, k, p::ParameterQM)
  prefactor = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  e = k^2 + rho * p.hϕ^2
  Nc = 3
  Nf = 2
  if (e >= p.μ)
    return -4Nc * Nf * prefactor / e
  end
  return 0.0
end

#####################################
############## STD ##################
#####################################

@inbounds @fastmath function kernel_standard_NOALLOC!(du, u, p::ParameterQM, t)
  k = exp(-t) * p.Λ
  Adkd = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  x = @view p.grid[:]
  n = length(x)

  du[1] = (Adkd * (-24 / k^2 + 4 / (k^2 + u[1]) + 1 / (-k^2 + 2 * u[1] - 3 * u[2]) - 3 / (k^2 + u[2]) + 24 / (k^2 + p.hϕ^2 * x[2]))) / x[2]

  for i in 2:n-1
    du[i] = (Adkd * (3 / (k^2 + u[i]) - 3 / (k^2 + u[i+1]) - 24 / (k^2 + p.hϕ^2 * x[i]) + 1 / (k^2 + u[i] + (2 * (u[i-1] - u[i]) * x[i]) / (x[i-1] - x[i])) + 24 / (k^2 + p.hϕ^2 * x[i+1]) - 1 / (k^2 + u[i+1] + (2 * (u[i] - u[i+1]) * x[i+1]) / (x[i] - x[i+1])))) / (-x[i] + x[i+1])
  end

  du[n] = (Adkd * (-3 / (k^2 + u[n-1]) + 3 / (k^2 + u[n]) + 24 / (k^2 + p.hϕ^2 * x[n-1]) - 1 / (k^2 + u[n-1] + (2 * (u[n-2] - u[n-1]) * x[n-1]) / (x[n-2] - x[n-1])) - 24 / (k^2 + p.hϕ^2 * x[n]) + 1 / (k^2 + u[n] + (2 * (u[n-1] - u[n]) * x[n]) / (x[n-1] - x[n])))) / (x[n-1] - x[n])
  nothing
end
init(::typeof(kernel_standard_NOALLOC!), p::ParameterQM) = map(x -> p.λ * x + p.m², p.grid)

@inbounds @fastmath function jac_standard!(J, u, p, t)
  k = exp(-t) * p.Λ
  Adkd = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  x = @view p.grid[:]
  n = length(x)

  J[1, 1] = (Adkd * (-4 / (k^2 + u[1])^2 - 2 / (k^2 - 2 * u[1] + 3 * u[2])^2)) / x[2]
  J[1, 2] = (3 * Adkd * ((ksq + u[2])^(-2) + (ksq - 2 * u[1] + 3 * u[2])^(-2))) / x[2]
  for i in 2:n-1
    J[i, i-1] = (2 * Adkd * (x[i-1] - x[i]) * x[i]) / (((k^2 + u[i]) * x[i-1] - (k^2 - 2 * u[i-1] + 3 * u[i]) * x[i])^2 * (x[i] - x[i+1]))
    J[i, i] = (Adkd * (-3 / (k^2 + u[i])^2 - (1 - (2 * x[i]) / (x[i-1] - x[i])) / (k^2 + u[i] + (2 * (u[i-1] - u[i]) * x[i]) / (x[i-1] - x[i]))^2 + (2 * (x[i] - x[i+1]) * x[i+1]) / ((k^2 + u[i+1]) * x[i] - (k^2 - 2 * u[i] + 3 * u[i+1]) * x[i+1])^2)) / (-x[i] + x[i+1])
    J[i, i+1] = (Adkd * (3 / (k^2 + u[i+1])^2 + ((x[i] - 3 * x[i+1]) * (x[i] - x[i+1])) / ((k^2 + u[i+1]) * x[i] - (k^2 - 2 * u[i] + 3 * u[i+1]) * x[i+1])^2)) / (-x[i] + x[i+1])
  end

  J[n, n-2] = (2 * Adkd * (x[n-2] - x[n-1]) * x[n-1]) / (((k^2 + u[n-1]) * x[n-2] - (k^2 - 2 * u[n-2] + 3 * u[n-1]) * x[n-1])^2 * (x[n-1] - x[n]))
  J[n, n-1] = (Adkd * (3 / (k^2 + u[n-1])^2 + ((x[n-2] - 3 * x[n-1]) * (x[n-2] - x[n-1])) / ((k^2 + u[n-1]) * x[n-2] - (k^2 - 2 * u[n-2] + 3 * u[n-1]) * x[n-1])^2 - (2 * (x[n-1] - x[n]) * x[n]) / ((k^2 + u[n]) * x[n-1] - (k^2 - 2 * u[n-1] + 3 * u[n]) * x[n])^2)) / (x[n-1] - x[n])
  J[n, n] = (Adkd * (-3 / (k^2 + u[n])^2 - (1 - (2 * x[n]) / (x[n-1] - x[n])) / (k^2 + u[n] + (2 * (u[n-1] - u[n]) * x[n]) / (x[n-1] - x[n]))^2)) / (x[n-1] - x[n])
  nothing
end
jacobian(::typeof(kernel_standard_NOALLOC!)) = jac_standard!

#####################################
########### Parameter ###############
#####################################

params_paper_QM = ParameterQM(generateGrid(7.5, 256), 3, 7.5, 1.0, 0.0, 1.5, 0.0, 6.0);