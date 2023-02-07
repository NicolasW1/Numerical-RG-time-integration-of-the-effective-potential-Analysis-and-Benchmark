struct ParameterScalar
	grid::Vector{Float64}
	dim::Int64
	Λ::Float64
	λ::Float64
	m²::Float64
	maxt::Float64
end

#####################################
############## STD ##################
#####################################

@inbounds @fastmath function kernel_standard_NOALLOC!(du, u, p::ParameterScalar, t)
  k = exp(-t) * p.Λ
  Adkd = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  x = @view p.grid[:]
  n = length(x)

  du[1] = -3 * Adkd * (u[1] - u[2]) / (x[2] * (k^2 + u[1]) * (k^2-2*u[1]+3*u[2]))

  for i in 2:n-1
    du[i] = Adkd / (x[i+1] - x[i]) * (1/(k^2 + u[i] + 2 * x[i] * (u[i-1] - u[i])/(x[i-1] - x[i])) - 1/(k^2 + u[i+1] + 2 * x[i+1] * (u[i] - u[i+1])/(x[i]-x[i+1])))
  end

  du[n] = Adkd/ (x[n-1] - x[n]) * (1/(k^2 + u[n] + 2 * x[n] * (u[n-1]-u[n])/(x[n-1]-x[n])) - 1/(k^2 + u[n-1] + 2 * x[n-1] * (u[n-2]-u[n-1])/(x[n-2]-x[n-1])))

  nothing
end
init(::typeof(kernel_standard_NOALLOC!), p::ParameterScalar) = map(x -> p.λ * x + p.m², p.grid)

@inbounds @fastmath function jac_standard!(J,u,p,t)
  k = exp(-t) * p.Λ
  Adkd = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  x = @view p.grid[:]
  n = length(x)

  J[1,1] = Adkd / x[2] * (-1/(k^2+u[1])^2 - 2/(k^2 - 2*u[1] + 3*u[2])^2)
  J[1,2] = 3 * Adkd /(x[2] * (k^2 - 2*u[1] + 3* u[2])^2)

  for i in 2:n-1
    J[i,i-1] = -2 * Adkd * x[i] / ((x[i-1] - x[i]) * (x[i+1] - x[i]) * (k^2 + u[i] + (2 * x[i] * (u[i-1] - u[i])/(x[i-1] - x[i])))^2)
    J[i,i] = Adkd / (x[i+1] - x[i]) * (-(1 - 2*x[i] / (x[i-1] - x[i])) / (k^2 + u[i] + 2 * x[i] * (u[i-1] - u[i])/(x[i-1] - x[i]))^2 + 2 * x[i+1] / ((x[i]-x[i+1]) * (k^2 + u[i+1] + 2*x[i+1] * (u[i]-u[i+1])/(x[i]-x[i+1]))^2))
    J[i,i+1] = Adkd * (1 - 2*x[i+1]/(x[i]-x[i+1]))/((x[i+1]-x[i]) * (k^2 + u[i+1] + 2*x[i+1]*(u[i]-u[i+1])/(x[i]-x[i+1]))^2)
  end

  J[n,n-2] = 2 * Adkd * x[n-1] / ((x[n-2] - x[n-1])*(x[n-1]-x[n])*(k^2+u[n-1]+2*x[n-1]*(u[n-2]-u[n-1])/(x[n-2]-x[n-1]))^2)
  J[n,n-1] = Adkd / (x[n-1] - x[n]) * ((1 - 2*x[n-1]/(x[n-2]-x[n-1]))/(k^2+u[n-1]+2*x[n-1]*(u[n-2]-u[n-1])/(x[n-2]-x[n-1]))^2 - 2*x[n]/((x[n-1]-x[n])*(k^2+u[n]+2*x[n]*(u[n-1]-u[n])/(x[n-1]-x[n]))^2))
  J[n,n] = -Adkd * (1 - 2*x[n]/(x[n-1]-x[n])) / ((x[n-1]-x[n])*(k^2 + u[n] + 2*x[n]*(u[n-1]-u[n])/(x[n-1]-x[n]))^2)

  nothing
end
jacobian(::typeof(kernel_standard_NOALLOC!)) = jac_standard!

#####################################
############## MASS #################
#####################################

@inbounds @fastmath function kernel_mass_NOALLOC!(du, u, p::ParameterScalar, t)
  k = exp(-t) * p.Λ
  ksq = k^2
  Adkd = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  x = @view p.grid[:]
  n = length(x)

  # du[1] = -Adkd * (u[1] - u[2]) / (x[2] * (k^2 + u[1]) * (k^2+u[2]))
  du[1] = (Adkd*(1/(ksq + u[1]) - 1/(ksq + u[2])))/x[2]

  for i in 2:n-1
    # du[i] = Adkd/((ksq + u[i]) * (x[i-1]-x[i])^2) * (2*x[i]*(u[i-1]-u[i])/(ksq+u[i-1]) + (u[i]-u[i+1])*(x[i-1]-3*x[i])*(x[i-1]-x[i])/((ksq+u[i+1])*(x[i]-x[i+1])))
    du[i] = (Adkd*(1/(ksq + u[i]) - 1/(ksq + u[i+1])))/(-x[i] + x[i+1]) + (2*x[i]*((Adkd*(1/(ksq + u[i-1]) - 1/(ksq + u[i])))/(-x[i-1] + x[i]) - (Adkd*(1/(ksq + u[i]) - 1/(ksq + u[i+1])))/(-x[i] + x[i+1])))/(x[i-1] - x[i])
  end

  du[n] = (-(Adkd/(ksq + u[n-1])) + Adkd/(ksq + u[n]))/(x[n-1] - x[n]) + (2*((-(Adkd/(ksq + u[n-2])) + Adkd/(ksq + u[n-1]))/(x[n-2] - x[n-1]) - (-(Adkd/(ksq + u[n-1])) + Adkd/(ksq + u[n]))/(x[n-1] - x[n]))*x[n])/(x[n-1] - x[n])

  nothing
end
init(::typeof(kernel_mass_NOALLOC!), p::ParameterScalar) = map(x -> 3.0 * p.λ * x + p.m², p.grid)

@inbounds @fastmath function jac_mass!(J,u,p,t)
  k = exp(-t) * p.Λ
  ksq = k^2
  Adkd = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  x = @view p.grid[:]
  n = length(x)

  J[1,1] = -(Adkd/((ksq + u[1])^2*x[2]))
  J[1,2] = Adkd/((ksq + u[2])^2*x[2])

  for i in 2:n-1
    J[i,i-1] = (2*Adkd*x[i])/((ksq + u[i-1])^2*(x[i-1] - x[i])^2)
    J[i,i] = (Adkd*(x[i-1]^2 - 4*x[i-1]*x[i] + x[i]*(x[i] + 2*x[i+1])))/((ksq + u[i])^2*(x[i-1] - x[i])^2*(x[i] - x[i+1]))
    J[i,i+1] = -((Adkd*(x[i-1] - 3*x[i]))/((ksq + u[i+1])^2*(x[i-1] - x[i])*(x[i] - x[i+1])))
  end

  J[n,n-2] = (2*Adkd*x[n])/((ksq + u[n-2])^2*(x[n-2] - x[n-1])*(x[n-1] - x[n]))
  J[n,n-1] = (Adkd*(-x[n-1]^2 + x[n-2]*(x[n-1] - 3*x[n]) + x[n-1]*x[n] + 2*x[n]^2))/((ksq + u[n-1])^2*(x[n-2] - x[n-1])*(x[n-1] - x[n])^2)
  J[n,n] = -((Adkd*(x[n-1] - 3*x[n]))/((ksq + u[n])^2*(x[n-1] - x[n])^2))

  nothing
end
jacobian(::typeof(kernel_mass_NOALLOC!)) = jac_mass!

#####################################
############## LOG ##################
#####################################

@inbounds @fastmath function kernel_log_NOALLOC!(du, u, p::ParameterScalar, t)
  k = exp(-t) * p.Λ
  ksq = k^2
  Adkd = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  x = @view p.grid[:]
  n = length(x)

  du[1] = exp(-u[1])*(-2*ksq + (Adkd*exp(-u[1]) - Adkd*exp(-u[2]))/x[2])

  for i in 2:n-1
    du[i] = exp(-u[i])*(-2*ksq + (Adkd*exp(-u[i]) - Adkd*exp(-u[i+1]))/(-x[i] + x[i+1]) + (2*x[i]*((Adkd*exp(-u[i-1]) - Adkd*exp(-u[i]))/(-x[i-1] + x[i]) - (Adkd*exp(-u[i]) - Adkd*exp(-u[i+1]))/(-x[i] + x[i+1])))/(x[i-1] - x[i]))
  end

  du[n] = exp(-u[n])*(-2*ksq + (Adkd*(-exp(-u[n-1]) + exp(-u[n])))/(x[n-1] - x[n]) + (2*((Adkd*(-exp(-u[n-2]) + exp(-u[n-1])))/(x[n-2] - x[n-1]) + (Adkd*(exp(-u[n-1]) - exp(-u[n])))/(x[n-1] - x[n]))*x[n])/(x[n-1] - x[n]))

  nothing
end
init(::typeof(kernel_log_NOALLOC!), p::ParameterScalar) = map(x -> log(p.Λ^2 + 3.0 * p.λ * x + p.m²), p.grid)

@inbounds @fastmath function jac_log!(J,u,p,t)
  k = exp(-t) * p.Λ
  ksq = k^2
  Adkd = 4π / (2π)^p.dim * k^(p.dim + 2) / p.dim
  x = @view p.grid[:]
  n = length(x)

  J[1,1] = exp(-2*u[1])*(2*ksq*exp(u[1]) - (Adkd*(2 - exp(u[1] - u[2])))/x[2])
  J[1,2] = (Adkd*exp(-u[1] - u[2]))/x[2]

  for i in 2:n-1
    J[i,i-1] = (2*Adkd*exp(-u[i-1] - u[i])*x[i])/(x[i-1] - x[i])^2
    J[i,i] = exp(-2*u[i])*((2*Adkd*x[i]*(-2 + exp(-u[i-1] + u[i]) - (2*(x[i-1] - x[i]))/(x[i] - x[i+1]) + (exp(u[i] - u[i+1])*(x[i-1] - x[i]))/(x[i] - x[i+1])))/(x[i-1] - x[i])^2 + (2*Adkd - Adkd*exp(u[i] - u[i+1]) + 2*ksq*exp(u[i])*(x[i] - x[i+1]))/(x[i] - x[i+1]))
    J[i,i+1] = -((Adkd*exp(-u[i] - u[i+1])*(x[i-1] - 3*x[i]))/((x[i-1] - x[i])*(x[i] - x[i+1])))
  end

  J[n,n-2] = (2*Adkd*exp(-u[n-2] - u[n])*x[n])/((x[n-2] - x[n-1])*(x[n-1] - x[n]))
  J[n,n-1] = (Adkd*exp(-u[n-1] - u[n])*(-x[n-1]^2 + x[n-2]*(x[n-1] - 3*x[n]) + x[n-1]*x[n] + 2*x[n]^2))/((x[n-2] - x[n-1])*(x[n-1] - x[n])^2)
  J[n,n] = (exp(-u[n])*((Adkd*exp(-u[n-1]) - 2*Adkd*exp(-u[n]) + 2*ksq*(x[n-1] - x[n]))*(x[n-1] - x[n]) + 2*Adkd*x[n]*(-exp(-u[n-1]) + 2*exp(-u[n]) + (exp(-u[n-2])*(x[n-1] - x[n]))/(x[n-2] - x[n-1]) + (exp(-u[n-1])*(-x[n-1] + x[n]))/(x[n-2] - x[n-1]))))/(x[n-1] - x[n])^2

  nothing
end
jacobian(::typeof(kernel_log_NOALLOC!)) = jac_log!


#####################################
########### Parameter ###############
#####################################

params_paper_1 = ParameterScalar(generateGrid(7.5, 256), 3, 7.5, 1.0, -2.5, 6.0);
