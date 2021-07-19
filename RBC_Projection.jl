# RBC_Projection.jl solves the RBC model using approximation by Chebyshev polynomials at points collocated at chebyshev zeros. I use Gauss-Hermite method to approximate expectations

################################################################################
#
# RBC_Projection.jl solves the RBC model using approximation by Chebyshev polynomials using Smolyak's algorithm . I use Gauss-Hermite method to approximate expectations.
#
# NB: this code is based on MATLAB code authored by Grey Gordon.
#
# Shane McMiken
# June 2021
# Boston College
#
################################################################################


#--------------------------------#
#         PREAMBLE
#--------------------------------#
using Distributed
using SharedArrays
using Plots
using GroupSlices  #groupslices(A, dim) Returns a vector of integers where each integer element of the returned vector is a group number corresponding to the unique slices along dimension `dim` as returned from `unique(A, dim)`, where `A` can be a multidimensional array.
using FastGaussQuadrature # gausshermite()


# Number of workers - cores
#addprocs(8)
workers() # checking workers

#--------------------------------#
#      MISC FUNCTIONS
#--------------------------------#

# extract data from a dictionary object into the local objects using (d::Dict)
@everywhere function extract(d::Dict)
    expr = quote end
    for (k, v) in d
        push!(expr.args, :($(Symbol(k)) = $v))
    end
    eval(expr)
    return
end

# Cartesian Product
function cartprod(x,y)
    leny=length(y)
    lenx=length(x)
    m=leny*lenx
    OUT = zeros(Float64, m,2)
    c=1
    for i = 1:lenx
        for j = 1:leny
            OUT[c,1] = x[i]
            OUT[c,2] = y[j]
            c+=1
        end
    end
    return OUT
end

# "sortrows" orders arrays size of row elements. To sort array by column 1 then 2 is sortrows(A,[1,2])
sortrows(A, i, rev=false) = sortslices(A, dims=1, lt=(x,y)->isless(x[i],y[i]), rev=rev)

# "matchrow" finds the index for the row "a" in matrix "B".
matchrow(a,B) = findfirst(i->all(j->a[j] == B[i,j],1:size(B,2)),1:size(B,1))


#--------------------------------#
#      CHEBYSHEV FUNCTIONS
#--------------------------------#

# Evaluates Chebyshev polynomial at point x for degree n
function cheb_eval(degree::Int64,x::Float64)
    if (degree == 0)
        T = 1
        return T
    elseif (degree == 1)
        T = x
        return T
    else
        Tm2         = 1
        Tm1         = x
        for ind = 2:degree-1
            T       = 2*x.*Tm1 .- Tm2
            Tm2     = copy(Tm1)
            Tm1     = copy(T)
        end
    end
    T = 2*x.*Tm1 .- Tm2
    if abs(sum(T)) < 1e-12
        return zeros(size(x,1),1)
    else
        return T
    end
end

# Given an integer n, construct the set of n extrema of the Chebyshev polynomials: zeta_j = -cos(pi*(j-1)/(n-1)), j = 1,n
function chebyshev_extrema(n::Int64)
    if n<1
        print("number of extrema points n must be ≧ 1")
        return
    elseif n==1
        return set = 0
    else
        j = 1:1:n
        set = -cos.(pi.*(j.-1)./(n-1))
    end
    for ii =1:n
        if abs(set[ii]) <= 1e-15
            set[ii] = 0
        elseif abs(set[ii]-1) < 1e-15
            set[ii] = 1
        elseif abs(set[ii]+1) < 1e-15
            set[ii] = -1
        end
    end
    return set
end

#--------------------------------#
#      SMOLYAK ALGORITHM
#--------------------------------#

# Given an integer i, deliver the function m(i) = 2^(i-1) + 1
function smolyak_m(i::Int)
    if i<0
        print("i must be ≧ 1")
        return
    elseif i == 1
        o = 1
    elseif i>1
        o = 2^(i-1) + 1
    end
    return o
end

function smolyak_ennumerate(d::Int,μ::Int)
    if μ == 0
        enum = zeros(1,d)
    end
    if μ >= 1
        enum       = zeros(d^μ,d)
        enum_mum1   = smolyak_ennumerate(d,μ-1)
        m           = size(enum_mum1,1)
        for ii = 1:d
            enum[1+(ii-1)*m:ii*m,:]  = enum_mum1
            enum[1+(ii-1)*m:ii*m,ii] = enum[1+(ii-1)*m:ii*m,ii] .+ 1
        end
    end
    return enum
end

# Store the "choose" constant values step
function choose(n::Int,k::Int)
    if (k > n | k < 0)
        print("k must be in [0,n]")
        return
    end

    if (k >= n-k)
        val = prod(k+1:n)/factorial(n-k)
    else
        val = prod(n-k+1:n)/factorial(k)
    end
    return val
end

# Smolyak recipies part 1 & 2
function smolyak_1(d::Int,μ::Int,a::Vector,b::Vector)

    q       = max(d,μ+1)

    # store objects used smolyak_2 + smolyak_3
    s           = Dict()
    s["q"]      = q
    s["d"]      = d
    s["μ"]      = μ
    s["a"]      = lb[1:d]
    s["b"]      = ub[1:d]
    s["M_mup1"] = smolyak_m(μ+1)

    # Check dimensions of "d"
    if length(a) > d
        print("length of a ≧ d")
    end
    if length(b) > d
        print("length of b ≧ d")
    end

    #Construct necessary coefficients using fx values.
    # First, enumerate all necessary theta
    # -For all i satisfying q<=ibar<=d+mu,
    # -need { l | l(1) is in 1...m(i1), l(2) is in 1...m(i1)}
    enum = []
    for ibar = q:(d+μ)
            enum_temp   = smolyak_ennumerate(d,ibar-d)
            enum_temp   = enum_temp .+ 1.0
            if ibar == q
            enum        = enum_temp
        else
            enum        = vcat(enum_temp,enum)
        end
    end

    enum        = unique(enum,dims = 1)
    enum        = sortrows(enum,[1,2])
    leni        = size(enum,1)

    s["i"]      = convert(Array{Int64},enum)
    s["leni"]   = leni
    s["ibar"]   = sum(s["i"],dims = 2)
    s["k"]      = smolyak_m.(s["i"])

    Lb          = zeros(leni,1)
    Ub          = zeros(leni,1)
    tmpInt      = 0

    for iind = 1:leni
        Lb[iind] = tmpInt + 1;
        tmpInt = tmpInt + prod(s["k"][iind,:]);
        Ub[iind] = tmpInt;
    end

    s["Lb"]     = convert(Array{Int64},Lb)
    s["Ub"]     = convert(Array{Int64},Ub)

    z = zeros(s["Ub"][leni],d)
    j = zeros(s["Ub"][leni],d)

    # Define the set of extrema points for Chebyshev polynomial
    for iind = 1:leni
        ztmp = []
        jtmp = []
        for dind = 1:d
            if (dind == 1)
                ztmp = chebyshev_extrema(s["k"][iind,dind])
                jtmp = 1:s["k"][iind,dind]
            else
                ztmp = cartprod(ztmp,chebyshev_extrema(s["k"][iind,dind]))
                jtmp = cartprod(jtmp,1:s["k"][iind,dind])
            end
        end
        z[s["Lb"][iind]:s["Ub"][iind],:] = ztmp
        j[s["Lb"][iind]:s["Ub"][iind],:] = jtmp
    end

    s["z"]  = z
    s["j"]  = convert(Array{Int64},j)

    # Reverse the transformation from Chebyshev support [-1,1] into the support of states Ub, Lb

    x = zeros(s["Ub"][leni],d)
    for dind = 1:d
        x[:,dind] =  (z[:,dind].+1).*(b[dind]-a[dind])/2 .+ a[dind]
    end

    s["x"]  = x
    s["f"]  = zeros(size(z))
    s["l"]  = s["j"]
    s["T"]  = zeros(s["Ub"][leni],maximum(s["Ub"]-s["Lb"])+1)

    for iind = 1:leni
        for lind = s["Lb"][iind]:s["Ub"][iind]
            for jind = s["Lb"][iind]:s["Ub"][iind]
                tmpProd = 1.0
                for dind = 1:d
                    tmpProd = tmpProd*cheb_eval(s["l"][lind,dind] - 1, s["z"][jind, dind])
                end
                s["T"][jind, lind - s["Lb"][iind]+1] = tmpProd[1]
            end
        end
    end

    s["clprod"] = zeros(size(s["z"],1),1)
    for iind = 1:leni
        for lind = s["Lb"][iind]:s["Ub"][iind]
            s["clprod"][lind] = 1.0
            for dind = 1:d
                if (s["k"][iind, dind]>1) # Dimensions with only one dimensions are "dropped" according to Krueger and Kubler
                    if ((s["l"][lind, dind]==1) || (s["l"][lind,dind]==s["k"][iind,dind]))
                        s["clprod"][lind] = s["clprod"][lind]*2.0
                    end
                end
            end
        end
    end

    # cjprod is used differently but has the same values as clprod
    s["cjprod"] = s["clprod"]

    # for each i, there is a unique "const"
    s["constant"] = zeros(leni,1)

    for iind = 1:leni
        tmpProd = 1.0
        tmpSum = 0.0
        for dind = 1:d
            if(s["k"][iind,dind] >1)
                tmpSum += 1.0
                tmpProd = tmpProd*(s["k"][iind,dind]-1)
            end
        end
        s["constant"][iind] = (2.0^tmpSum)/tmpProd
    end

    # Store the "choose" constant values
    s["constVec"] = zeros(d+μ+1-q,1)
    for ibar = q:(d+μ)
        s["constVec"][ibar-q+1] = (-1.0)^(d+μ-ibar)
    end

    # Get Smolyak grid points
    tmp         = [s["x"] groupslices(s["x"],1)]
    tmp         = unique(tmp, dims = 1)
    tmp         = sortrows(tmp,[1,2])
    x           = tmp[:,1:2]

    # performs similiar function as matlab "unique": [C,redo,undo] = unique(A,'rows') where index vectors redo and undo are such that C = A(redo,:) and A = C(undo,:) that C = A(IA,:) and A = C(IC,:)
    s["redo"]   = convert(Vector{Int64},tmp[:,3])
    s["undo"]   = [matchrow(s["x"][ii,:],x) for ii =1:size(s["x"],1)]
    return x,s
end

function smolyak_2(fx::Vector,s::Dict)

    # Check dims
    if (size(fx,1) != size(s["redo"],1))
        print("fx dims are incorrect")
        return
    end

    s["f"] = fx[s["undo"],:] # Unpack fx
    s["theta"] = zeros(size(s["f"])) # Construct the polynomial coefficients

    extract(s) # extract dictionary

    for iind = 1:leni
        for lind = Lb[iind]:Ub[iind]
            tmp = transpose(T[Lb[iind]:Ub[iind],lind-Lb[iind]+1]./cjprod[Lb[iind]:Ub[iind]])
            s["theta"][lind,:] = constant[iind]/clprod[lind]*tmp*f[Lb[iind]:Ub[iind],:]
        end
    end

    # Precompute constant times the coefficient
    s["constTimesTheta"] = s["theta"]
    for iind = 1:leni
        s["constTimesTheta"][Lb[iind]:Ub[iind],:] = s["constVec"][ibar[iind]-q+1 ]*s["constTimesTheta"][Lb[iind]:Ub[iind],:]
    end
    return s
end

# "smolyak_3" feval = smolyak_3(pol,x) -- Given pol which defines an approximation to f, compute the approx value of f at the pts "x".
@everywhere function smolyak_3(s, x)

    extract(s) # extract dictionary

    # Check dims
    if (d != size(x,2))
        print("x dims are incorrect")
        return
    end

    # Convert x points to z via linear transformation
    z = zeros(size(x))
    for dind = 1:d
        z[:,dind] = 2.0*(x[:,dind].-a[dind])/(b[dind]-a[dind]) .- 1.0
    end

    # Evaluate the cheby values beforehand for all orders
    T = ones(size(x,1), M_mup1, d)
    for dind = 1:d
        T[:,2,dind] = z[:,dind]
    end

    twoz = 2*z
    for oind = 3:M_mup1
        for dind = 1:d
            T[:,oind,dind] = twoz[:,dind].*T[:,oind-1,dind] - T[:,oind-2,dind]
        end
    end

    # For each possible value of l, compute the product across i of T(:,li) where li is a component of l
    Tprod = ones(size(x,1),size(l,1))
    for dind = 1:d
        for lind = 1:size(l,1)
            if (l[lind,dind]>1)
                Tprod[:,lind] = Tprod[:,lind].*T[:,l[lind,dind],dind]
            end
        end
    end
    feval = Tprod*constTimesTheta
    return feval
end

#--------------------------------#
#         MODEL
#--------------------------------#

# model parameters
@everywhere β       = 0.994
@everywhere α       = 0.27
@everywhere δ       = 0.011
@everywhere σ       = 0.05
@everywhere ρ       = 0.9
@everywhere η       = 2.0
@everywhere ψ       = 2.0

# steady state
@everywhere Rss     = β^(-1)-(1-δ)
@everywhere klss    = (Rss/α)^(1/(α-1))
@everywhere clss    = klss^α-δ*klss
@everywhere lss     = ((1-α)*klss^α/(klss^α-δ*klss)^η)^(1/(ψ+η))
@everywhere kss     = klss*lss
@everywhere css     = clss*lss

# grids
@everywhere kmin    = 0.6*kss
@everywhere kmax    = 1.6*kss
@everywhere lmin    = 0.6*lss
@everywhere lmax    = 1.6*lss
@everywhere amax    = exp(5*σ/sqrt(1.0-ρ^2))
@everywhere amin    = 1.0/amax
@everywhere nk      = 101
@everywhere nl      = 101
@everywhere kgrid   = collect(range(kmin, kmax, length = nk))
#@everywhere lgrid   = collect(range(lmin, lmax, length = nl))
@everywhere lb      = [amin;kmin]
@everywhere ub      = [amax;kmax]

# other parameters
@everywhere ns      = 2 # number of states
@everywhere μ       = 3 # approximation finess (implies q=5 from lecture notes)
@everywhere mz      = 5 # five-point Gauss-Hermite quadrature rule
@everywhere λ       = 0.45

# STEP 1: GET SMOLYAK GRID POINTS

x_unique, ienumlist = smolyak_1(ns,μ,lb,ub)
ncol = size(x_unique,1)
print("number of collocation points is $ncol")


# STEP 2: INITIAL GUESS

consum0 = css.*ones(size(x_unique,1),1)

# Guess with mu = 3
# siga = 0.005
expect0 = [1.96718023530470, 2.02792258363725,2.16654587328216,2.29573864505488, 2.34692245920707,2.16794344861270,1.97248792334180,2.17191391633511, 2.35233339343833,2.17783018887064,1.98520076246493,2.00121393285854, 2.04601052827352,2.11098753086973,2.18476956722731,2.25579403695945, 2.31406902429019,2.35209923884793,2.36529046731483,2.19166717745200,  1.99777490410201,2.19748259302775,2.37810203766553,2.20135217581634, 2.00294384107665,2.06381845362820,2.20270795504085,2.33210987122065, 2.38336741025258]
expect0 = expect0.^(-η)

# STEP 3: CONSTRUCT POLYNOMIAL APPROXIMATION - GET THETAS
sexpect = smolyak_2(expect0,ienumlist)
resi    = SharedArray{Float64}(ncol,1)

#--------------------------------#
#   MAIN LOOP - POLICY FUNCITON
#--------------------------------#


function main(x_unique::Array, expect_init::Vector, s::Dict, resi::SharedArray; α = α, η = η, δ = δ, ncol = ncol, λ = λ, mz = mz, ρ = ρ, σ = σ, tol = 1.0e-6, maxiter = 10000)

    iter        = 1
    crit        = Inf
    expect0     = copy(expect_init)

    # hermite mz-point integration nodes
    int_node    = gausshermite(mz)

    while (crit > tol && iter < maxiter)

        sexpect = smolyak_2(expect0,s) # get thetas
        extract(sexpect) # extract dictionary

        # solve for consumption
        consum0 = expect0.^(-1/η)

        # solve for labor
        # -c_{t}^η l_{t}^ψ + (1-α) k_t^α l_t^(-α) A_t = 0
        labor  = ((1-α).*x_unique[:,1].*x_unique[:,2].^α.*expect0).^(1/(ψ+α))

        # solve for capital
        # k_{t+1} = A_t k_t^α + (1-δ)k_t - c_t
        capital = x_unique[:,1].*x_unique[:,2].^α.*labor.^(1-α) + (1-δ)*x_unique[:,2] - consum0

        # check non-negativity
        nonneg  = (capital .>= (1-δ)*x_unique[:,2]) #  equals 1 if I≧0
        capital = capital.*nonneg + (-nonneg.+1).*(1-δ).*x_unique[:,2]
        labor  = labor.*nonneg + (-nonneg.+1).*((1-α).*x_unique[:,1].^(1-η).*x_unique[:,2].^(α*(1-η))).^(1/(α*(1-η)+ψ+η))# Nb c_t = z_t k_t^α l_t^(1-α) & #intratemporal decision
        consum0 = consum0.*nonneg + (-nonneg.+1).*x_unique[:,1].*x_unique[:,2].^α.*labor.^(1-α)


        @sync @distributed for mx = 1:ncol # parallelize this, later
            aux = 0
            for zm = 1:mz # evaluate integral
                # 1. Compute tomorrow shock in each node
                atemp = x_unique[mx,1]^ρ*exp(2^0.5*σ*int_node[1][zm])
                # 2. Compute tomorrow's expectation
                expect1     = smolyak_3(sexpect, [atemp capital[mx]])
                # 2.1 Compute tomorrow's consumption
                consum1     = expect1.^(-1/η)
                # 2.2 Compute tomorrow's labor
                labor1      = ((1-α).*atemp.*capital[mx].^α.*expect1).^(1/(ψ+α))
                # 2.3 Compute tomorrow's capital
                capital1    = atemp.*capital[mx]^α*labor1^(1-α) .+ (1-δ)*capital[mx] .-consum1
                # 2.4 Check non-negativity
                nonneg      = (capital1 .>= (1-δ)*capital[mx])
                labor1      = labor1.*nonneg + (-nonneg.+1).*((1-α)*atemp^(1-η)*capital[mx]^(α*(1-η))).^(1/(α*(1-η)+ψ+η))
                consum1     = nonneg*consum1 + (-nonneg.+1)*atemp*capital[mx]^α*labor1^(1-α)
                mult1       = 0 .+ (-nonneg.+1).*(consum1.^(-η) .- expect1)
                # 3. Compute expectation
                aux = aux .+ int_node[2][zm]*(consum1^(-η)*(1-δ .+ α*capital[mx]^(α-1)*labor1^(1-α)*atemp) - (1-δ)*mult1)/pi^0.5
            end
            resi[mx] = β * aux[1]
        end

        crit     = maximum(abs.(resi[:] - expect0[:]))
        iter    += 1

        print("iteration: ",iter, ", critical val: ", crit*(1/tol)," times 10^-6", "\n")

        # Update solution
        expect0 = λ*resi[:,1] + (1-λ)*expect0[:,1]

    end
    return expect0
end

@time expect = main(x_unique,expect0,ienumlist,resi)


#--------------------------------#
#   COMPUTE AND PLOT SOLUTION
#--------------------------------#

shocks  = ones(length(kgrid),1).^ρ*exp(-4*σ)
sexpect = smolyak_2(expect,ienumlist)
consum0 = (smolyak_3(sexpect, [shocks kgrid])).^(-1/η)
labor0  = ((1-α).*shocks.*kgrid.^α.*consum0.^-η).^(1/(ψ+α))
capital = shocks.*kgrid.^α.*labor0.^(1-α) + (1-δ)*kgrid - consum0
nonneg  = (capital .>= (1-δ)*kgrid)
capital = capital.*nonneg + (-nonneg.+1).*(1-δ).*kgrid
labor0  = labor0.*nonneg + (-nonneg.+1).*((1-α).*shocks.^(1-η).*kgrid.^(α*(1-η))).^(1/(α*(1-η)+ψ+η))
consum0 = consum0.*nonneg + (-nonneg.+1).*shocks.*kgrid.^α.*labor0.^(1-α)
p1 = plot(kgrid,consum0, title = "consumption policy", xlabel = "assets", ylabel = "consumption", label = "policy")
scatter!([kss],[css], color =:red, Shape =:circle, label = "steady state")
p2 = plot(kgrid,labor0, title = "labor supply policy", xlabel = "assets", ylabel = "labor supply", label = "policy")
scatter!([kss],[lss], color =:red, Shape =:circle, label = "steady state")

policyfunctions = plot(p1,p2, layout = (2,1), legend = false)
display(policyfunctions)
savefig(policyfunctions,"./Projection_policyfunc.png")
