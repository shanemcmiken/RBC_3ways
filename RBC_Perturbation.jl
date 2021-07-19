################################################################################
#
# RBC perturbation.jl solves the RBC model using Schmitt-Grohé & Uribe (2004) method
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
using ForwardDiff
using LinearAlgebra # schur(A::StridedMatrix, B::StridedMatrix) -> F::GeneralizedSchur

#--------------------------------#
#         RBC MODEL
#--------------------------------#

# 1 - β [ (c^η/cp^η) (1 + rp - δ) ]
# w - l^ψ c^η
# (1-δ) k + i - kp
# c + i - y
# y - a * k^α * l^(1-α)
# w - (1-α)*(y/l)
# r - α * (y/k)
# log(ap) - ρ*log(a)

# parameters
η   = 0.8   # relative risk aversion
ψ   = 0.2   # inverse Frisch elasticity
β   = 0.95  # discount rate
α   = 0.3   # capital share of income
δ   = 0.05  # depreciation rate
ρ   = 0.9   # persistence parameter
σ   = 0.02  # Standard deviation of TFP shocks


# steady state - levels
abar     = 1.0
rbar     = β^(-1)-(1-δ)
klbar    = (rbar/α)^(1/(α-1))
clbar    = abar*klbar^α-δ*klbar
lbar     = ((1-α)*abar*klbar^α/(abar*klbar^α-δ*klbar)^η)^(1/(ψ+η))
kbar     = klbar*lbar
cbar     = clbar*lbar
wbar     = (1-α)*abar*klbar^α
ybar     = kbar^α*lbar^(1-α)
ibar     = δ*kbar

# Define vectors of controls y, states x
Yss = [ybar, cbar, lbar, wbar, rbar, ibar]
Xss = [kbar, abar]

ny  = length(Yss) # Number of control variables
nx  = length(Xss) # Number of state variables
nyp = length(Yss)
nxp = length(Xss)

# take logs
lYss = log.(Yss)
lXss = log.(Xss)

xs  = vcat(lYss,lXss,lYss,lXss)


#--------------------------------#
#       MODEL EQUATIONS
#--------------------------------#

# 8 equations & 8 unknowns
#
#
# y c l w r i k a
# 1 2 3 4 5 6 7 8
#
# yp cp lp wp rp ip kp ap
# 9  10 11 12 13 14 15 16
#
# 1. c^-η - β [ cp^-η (1 + rp - δ) ]
# 2. l^ψ c^η - w
# 3. (1-δ) k + i - kp
# 4. c + i - y
# 5. y - a * k^α * l^(1-α)
# 6. w - (1-α)*(k/l)^α
# 7. r - α * (l/k)^(1-α)
# 8. log(ap) - ρ*log(a)

f1(x::AbstractArray) = exp(x[2])^(-η) - β*(exp(x[10])^(-η)*(1 + exp(x[13]) - δ))
f2(x::AbstractArray) = exp(x[4]) - exp(x[3])^ψ * exp(x[2])^η
f3(x::AbstractArray) = (1-δ)*exp(x[7]) + exp(x[6]) - exp(x[15])
f4(x::AbstractArray) = exp(x[2]) + exp(x[6]) - exp(x[1])
f5(x::AbstractArray) = exp(x[1]) - exp(x[8]) * exp(x[7])^α * exp(x[3])^(1-α)
f6(x::AbstractArray) = exp(x[4]) - (1-α)*(exp(x[7])/exp(x[3]))^α
f7(x::AbstractArray) = exp(x[5]) - α*(exp(x[3])/exp(x[7]))^(1-α)
f8(x::AbstractArray) = log(exp(x[16])) - ρ*log(exp(x[8]))

f = [f1;f2;f3;f4;f5;f6;f7;f8]


#--------------------------------#
#   GET NUMERICAL DERIVATIVES
#--------------------------------#

function Numerical_derivatives(f::Array{Function},y::Array,x::Array,yp::Array,xp::Array; ny=ny, nx=nx, nyp=nyp, nxp=nxp, xs=xs)

    # compute jacobian
    J   = zeros(length(f),length(xs))
    for (ii,func) in enumerate(f)
        J[ii,:] = ForwardDiff.gradient(func,xs)
    end

    # compute first order matices
    fy  = J[:,1:ny]
    fx  = J[:,(1+ny):(nx+ny)]
    fyp = J[:,(1+nx+ny):(nx+ny+nyp)]
    fxp = J[:,(1+nx+ny+nyp):(nx+ny+nyp+nxp)]

    return fy,fx,fyp,fxp
end

@time fy,fx,fyp,fxp = Numerical_derivatives(f,lYss,lXss,lYss,lXss)

#--------------------------------#
#   FIRST ORDER APPROXIMATION
#--------------------------------#

# compute the gx, hx matrix
function gx_hx(fy::Array,fx::Array,fyp::Array,fxp::Array; ny=ny, nx=nx, nyp=nyp, nxp=nxp)

    # Create system matrices A,B
    A   = [-fxp -fyp]
    B   = [fx fy]
    NK  = size(fx,2)

    # Complex Schur decomposition
    F = schur(A,B)

    # Pick non-explosive (stable) eigenvalues
    slt = abs.(diag(F.T)) .< abs.(diag(F.S))
    nk = sum(slt)

    # Reorder the system with stable eigs in upper-left
    S,T,Q,Z = ordschur(F,slt)

    # Split up results appropriately
    Z21 = Z[nk+1:end,1:nk]
    Z11 = Z[1:nk,1:nk]
    S11 = S[1:nk,1:nk]
    T11 = T[1:nk,1:nk]

    # Identify cases with no or multiple solutions
    if nk > NK
        print("Warning gx_hx: the equilibrium is locally indeterminate")
    elseif nk < NK
        print("Warning gx_hx: no local equilibrium exists")
    end

    # Check invertibility
    if rank(Z11)<nk
        print("Invertibility condition violated")
    end

    # Compute solution
    Z11i    = Z11\I(nk)
    gx      = real(Z21*Z11i)
    hx      = real(Z11*(S11\T11)*Z11i)

    return gx, hx
end

@time gx, hx = gx_hx(fy,fx,fyp,fxp)

#--------------------------------#
#   IMPULSE RESPONSES
#--------------------------------#

T   = 50
eta = [0,σ]

function ir(gx::Array,hx::Array,x0,T::Int)

  #x0=x0(:)
  pd=size(x0,1)
  MX=[gx;I(pd)]
  IR=zeros(T,size(MX,1))
  x=x0

  for t=1:T
  IR[t,:]=(MX*x)'
  x = hx * x
  end

  iry = IR[:,1:ny]
  irx = IR[:,ny+1:end]

  return iry,irx
end

irfy,irfx = ir(gx,hx,eta,T)

# Output
iry1 = plot(1:T,irfy[:,1], title="Output", label="IRF")
# Consumption
iry2 = plot(1:T,irfy[:,2], title="Consumption", label="IRF")
# Hours
iry3 = plot(1:T,irfy[:,3], title="Hours", label="IRF")
# Wages
iry4 = plot(1:T,irfy[:,4], title="Wages", label="IRF")
# Investment
iry5 = plot(1:T,irfy[:,5], title="Investment", label="IRF")
# Real Rate
iry6 = plot(1:T,irfy[:,6], title="Real Rate", label="IRF")

# Capital
irx1 = plot(1:T,irfx[:,1], title="Capital", label="IRF")
# TFP
irx2 = plot(1:T,irfx[:,2], title="Technology", label="IRF")

impulseresponses = plot(iry1,iry2,iry3,iry4,iry5,iry6,irx1,irx2, layout = (4,2), legend = false)
display(impulseresponses)
savefig(policyfunctions,"./IRs.png")
