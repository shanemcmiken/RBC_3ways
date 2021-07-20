################################################################################
#
# RBC_VFI.jl solves the RBC model using value function iteration
#
# Shane McMiken
# June 2021
# Boston College
#
################################################################################

#--------------------------------#
#         PREAMBLE
#--------------------------------#

using Plots
using Distributions

#--------------------------------#
#     RBC MODEL
#--------------------------------#
#
# MODEL EQUATIONS
# 1. 1 - β [ (c^η/cp^η) (1 + rp - δ) ]
# 2. w - h^ψ c^η
# 3. (1-δ) k + i - kp
# 4. c + i - y
# 5. y - a * k^α * h^(1-α)
# 6. w - (1-α)*(y/h)
# 7. r - α * (y/k)
# 8. log(ap) - ρ*log(a)

# model parameters
β   = 0.95  # discount rate
η   = 0.8   # relative risk aversion
ψ   = 0.2   # inverse Frisch elasticity
α   = 0.3   # capital share of income
δ   = 0.05  # depreciation rate
ρ   = 0.9   # persistence parameter
σ   = 0.02  # Standard deviation of TFP shocks

# steady state
abar  = 1.0
rbar  = β^(-1)-(1-δ)
khbar = (rbar/α)^(1/(α-1))
chbar = abar*khbar^α-δ*khbar
hbar  = ((1-α)*abar*khbar^α/(abar*khbar^α-δ*khbar)^η)^(1/(ψ+η))
kbar  = khbar*hbar
cbar  = chbar*hbar
wbar  = (1-α)*abar*khbar^α
ybar  = kbar^α*hbar^(1-α)
ibar  = δ*kbar

#--------------------------------#
#  GRIDS and TRANSITION MATRIX
#--------------------------------#
# Steady state value function
Vbar = 1.0/(1.0-β)*(cbar^(1.0-η)/(1-η) - hbar^(1.0+ψ)/(1.0+ψ))

# Hours grid - in levels
nh = 150
hmin = 0.7*hbar
hmax = 1.3*hbar
vecsize = nh
step = (hmax - hmin)/(vecsize-1)
hgrid = [hmin + (i-1)*step for i∈1:nh]

# Capital grid - in logs
nk = 150
kmin = 0.7*kbar
kmax = 1.3*kbar
vecsize = nk
step = (kmax - kmin)/(vecsize-1)
kgrid  = [kmin + (i-1)*step for i∈1:nk]

# Technology grid - in logs - Tauchen (1986)
cover = 3
na = 2*cover + 1
m  = 1.5

function MyTauchen(σ_a::Float64,ρ_a::Float64,na::Int64,m::Float64)

  vecsize = na
  σ_y = sqrt(σ_a^2 / (1-(ρ_a^2))) # Stationary distribution variance
  step = 2 * m * σ_y / (vecsize-1)
  agrid  = [-m*σ_y + (i-1)*step for i∈1:na]

  # Calculate transition probabilities, AR(1) - Tauchen
  P = zeros(na, na)

  mm = agrid[2] - agrid[1]; # re-compute step
  for j = 1:na
    for k = 1:na
      if(k == 1) # first column
        P[j, k] = cdf(Normal(), (agrid[k] - ρ_a*agrid[j] + (mm/2))/σ_a)
      elseif(k == na) # second-last column
        P[j, k] = 1 - cdf(Normal(), (agrid[k] - ρ_a*agrid[j] - (mm/2))/σ_a)
      else
        P[j, k] = cdf(Normal(), (agrid[k] - ρ_a*agrid[j] + (mm/2))/σ_a) - cdf(Normal(), (agrid[k] - ρ_a*agrid[j] - (mm/2))/σ_a)
      end
    end #k
  end #j

  agrid = exp.(agrid) # Exponentiate grid

  return agrid, P #return productivity grid and Markov transition matrix
end

agrid, P = MyTauchen(σ,ρ,na,m)

#--------------------------------#
#    VALUE FUNCTION ITERATION
#--------------------------------#
# RECURSIVE PROBLEM:
#
# V(k,a) = max_{l,k_(+1)} log(c) - χ(l) + β*E[V(k_(+1),a_(t+1)]}
#
# subject to
#
# c + k - (1-δ)*k = a*k^α*l^(1-α

struct modelstate
  ind::Int64
  na::Int64
  nk::Int64
  nh::Int64
  P::Array
  agrid::Vector{Float64}
  kgrid::Vector{Float64}
  hgrid::Vector{Float64}
  β::Float64 # Discount rate
  η::Float64 # Relative risk aversion
  ψ::Float64 # inverse Frisch elasticity
  δ::Float64 # Depreication rate
  α::Float64 # Capital share of income
  V::Array
end

# computes value function given model state
function value(currentState::modelstate)
  ind     = currentState.ind
  na      = currentState.na
  nk      = currentState.nk
  nh      = currentState.nh
  P       = currentState.P
  agrid   = currentState.agrid
  kgrid   = currentState.kgrid
  hgrid   = currentState.hgrid
  β       = currentState.β
  η       = currentState.η
  ψ       = currentState.ψ
  δ       = currentState.δ
  α       = currentState.α
  VV      = currentState.V

  # Get states
  kk      = convert(Int, floor((ind-0.05)/na))+1
  aa      = convert(Int, floor(mod(ind-0.05,na))+1)

  # states
  kc      = kgrid[kk] # current capital
  ac      = agrid[aa] # current technology

  # hold capital kc constant, compute continuation value conditional on "ac" technology
  cont    = P[aa,:]' * VV
  cont    = vec(repeat(cont, nk)) # value is constant in the dimension of h-choice

  # compute model objects
  kkgr    = vec(repeat(kgrid',nh))
  hhgr    = vec(repeat(hgrid,nk,1))

  output      = ac.*kc.^α.*hhgr.^(1.0-α)
  investment  = kkgr .- (1.0-δ)*kc
  consumption = output - investment

  # compute new value funciton
  VVp = zeros(nh*nk)
  for (i,cons) in enumerate(consumption)
    if cons <= 0
      VVp[i] = -Inf
    else
      VVp[i] = cons^(1.0-η)/(1.0-η) - (hhgr[i])^(1.0+ψ)/(1.0+ψ) + β*cont[i]
    end
  end

  # find maximum
  Vnew, tempidx = findmax(vec(VVp))

  return Vnew, tempidx
end

#--------------------------------#
#           MAIN LOOP
#--------------------------------#

function VFI(Vbar::Float64, agrid::Array, kgrid::Array, hgrid::Array, P::Array; α = α, β = β, η = η, ψ = ψ, δ = δ, crit = Inf, maxiter = 1000, iter = 0, tol = 1.0e-8, nfix = 10)

  na = size(agrid,1)
  nh = size(hgrid,1)
  nk = size(kgrid,1)
  V_init = Vbar*ones(na,nk)
  V_next = Array{Float64,1}
  index = ones(na*nk,)

  # useful grids
  kagr = vec(repeat(kgrid',na))
  aagr = repeat(agrid,nk,1)
  hhgr = repeat(hgrid,nk,1)
  kkgr = vec(repeat(kgrid',nh))

  kidx    = collect(1:1:nh)
  kidx    = repeat(kidx,1,nh)'

  while (crit > tol && iter < maxiter)

    V_next = copy(V_init)

    if mod(iter,nfix) == 0 # speed-up: optimization every 10 iterations

      for ind = 1:(na*nk)
        State   = modelstate(ind,na,nk,nh,P,agrid,kgrid,hgrid,β,η,ψ,δ,α,V_next)
        tempVal, tempIndex  = value(State)

        V_next[ind] = tempVal
        index[ind] = tempIndex
      end

      else # on mod(iter,nfix) != 0, update the value function only.

        # Re-use conjectured index
        idx = convert(Array{Int64,1}, index)

        # E(V) condition on A(t) and each possible K(t+1)
        EVp = P*V_init

        # E(V) conditioned on A(t) and optimal K(t+1)
        ktmp = kidx[idx]
        evp_k = zeros(na*nk)
        for ind = 1:(na*nk)
          kk = convert(Int, floor((ind-0.05)/na))+1
          aa = convert(Int, floor(mod(ind-0.05,na))+1)
          evp_k[ind] = EVp[aa,ktmp[ind]]
        end

        #Recompute value using conjectured value function
        output = aagr.*kagr.^α.*hhgr[idx].^(1.0-α)
        investment = kkgr[idx] .- (1.0-δ)*kagr
        consumption = output - investment

        # Recompute value function under conjectured policy
        V_next = consumption.^(1.0-η)./(1.0-η) - hhgr[idx].^(1.0+ψ)./(1.0+ψ) + β*evp_k
        V_next = reshape(V_next,na,nk)
    end


    crit = maximum(abs.(V_next .- V_init))
    V_init = copy(V_next)
    iter += 1

    print("iteration: ",iter, ", critical val: ", crit*1.0e8," times 10^8", "\n")
  end #while

  return V_next, index, crit
end

@time Valuefunction, index, crit = VFI(Vbar,agrid,kgrid,hgrid,P)


#--------------------------------#
#    PLOT POLICY FUNCTIONS
#--------------------------------#

index   = convert(Array{Int64,1}, index)
pols    = reshape(index,(na,nk)) # policies

alow    = pols[cover-1,:] # fix productivity at low
amid    = pols[cover+1,:] # fix productivity at steady state
ahigh   = pols[cover+2,:] # fix productivity high

# Policies for low,mid,high realisations of "a"
hpol_low  = floor.(mod.(alow.-0.05,nh)) .+ 1.0
hpol_mid  = floor.(mod.(amid.-0.05,nh)) .+ 1.0
hpol_high = floor.(mod.(ahigh.-0.05,nh)) .+ 1.0
kpol_low  = floor.((alow.-0.05)./nk) .+ 1.0
kpol_mid  = floor.((amid.-0.05)./nk) .+ 1.0
kpol_high = floor.((ahigh.-0.05)./nk) .+ 1.0

# convert to integer arrays
hpol_low = convert(Array{Int64},hpol_low)
hpol_mid = convert(Array{Int64},hpol_mid)
hpol_high = convert(Array{Int64},hpol_high)
kpol_low = convert(Array{Int64},kpol_low)
kpol_mid = convert(Array{Int64},kpol_mid)
kpol_high = convert(Array{Int64},kpol_high)

# make plots
p1 = plot(kgrid,hgrid[hpol_low], ylims = (hmin,hmax), ylabel="low a", title="h policy")
p3 = plot(kgrid,hgrid[hpol_mid], ylims = (hmin,hmax), ylabel="steady state a")
p5 = plot(kgrid,hgrid[hpol_high], ylims = (hmin,hmax), ylabel="high a")

p2 = plot(kgrid,kgrid[kpol_low], ylims = (kmin,kmax), title="k policy")
p4 = plot(kgrid,kgrid[kpol_mid], ylims = (kmin,kmax))
p6 = plot(kgrid,kgrid[kpol_high], ylims = (kmin,kmax))

VFI_policyfunctions = plot(p1,p2,p3,p4,p5,p6, layout = (3,2), legend = false)
display(VFI_policyfunctions)
savefig(VFI_policyfunctions,"./VFI_policyfunc.png")
