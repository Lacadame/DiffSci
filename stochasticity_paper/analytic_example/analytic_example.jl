using CairoMakie
using Random
using Statistics
using BenchmarkTools
using Distributions
using PrettyTables
using Test

# Diffusion rates
g_ve(t, p) = p.σ
g_edm(t, p) = p.σ * √(2t)

g(t, p) = p.schd == :edm ? g_edm(t, p) : p.schd == :ve ? g_ve(t, p) : zero(t)

# The variance of the solution \{X_t\}_t of the VE forward equation
# dX_t = σ dW_t
# with initial condition
# X_0 ∼ 𝒩(μ₀, σ₀^2)
# (the mean is always 𝔼[X_t] = μ₀)
sigmasq_ve(t, p) = p.σ₀^2 + p.σ^2 * t
sigmasq_prime_ve(t, p) = p.σ^2

# The variance of the solution \{X_t\}_t of the EDM forward equation
# dX_t = σ√(2t) dW_t
# with initial condition
# X_0 ∼ 𝒩(μ₀, σ₀^2)
# (the mean is always 𝔼[X_t] = μ₀)
sigmasq_edm(t, p) = p.σ₀^2 + p.σ^2 * t^2
sigmasq_prime_edm(t, p) = 2p.σ^2 * t

# The variance depending on the diffusion schedule

sigmasq(t, p) = p.schd == :edm ? 
    sigmasq_edm(t, p) : 
    p.schd == :ve ? 
    sigmasq_ve(t, p) : 
    zero(t) 

sigmabarsq(τ, p) = sigmasq(p.T - τ, p)

sigmasq_prime(t, p) = p.schd == :edm ? 
    sigmasq_prime_edm(t, p) : 
    p.schd == :ve ? 
    sigmasq_prime_ve(t, p) : 
    zero(t) 

sigmabarsq_prime(τ, p) = sigmasq_prime(p.T - τ, p)

# The mean square error in the score when using
# the approximate score
# s = - ( x - μθ ) / ( αθ σ(t)^2 )
score_error_mse(t, p) = ( p.μθ - p.μ₀ )^2 / ( p.αθ^2 * sigmasq(t, p)^2 ) + ( 1 - 1 / p.αθ )^2 / sigmasq(t, p)

# Mean of the approximate reverse equation with starting condition
# X̃_{τ = 0} ∼ 𝒩(μ_T, βT σ(T)^2)
mutilde(τ, p) = p.μθ + ( sigmabarsq(τ, p) / sigmabarsq(0, p) )^( (1 + p.γ)/2p.αθ ) * ( p.μT - p.μθ )

# Variance of the reverse approximate equation with starting condition
# X̃_{τ = 0} ∼ 𝒩(μT, βT σ(T)^2)
function sigmatildesq(τ, p) 
    (; αθ, γ, βT) = p
    s = sigmabarsq(τ, p)
    if s ≈ 0.0
        return 0.0
    else
        r = sigmabarsq(0.0, p) / s
        if 1 + γ ≈ αθ
            return s * ( βT + γ * log(r) )
        else
            return s * ( γ * αθ / ( 1 + γ - αθ ) + (βT  - γ * αθ / ( 1 + γ - αθ ) ) * ( r )^( 1 - (1 + γ)/αθ ) )
        end
    end
end

# KL divergence H(p | q) with p ∼ 𝒩(p.μ, p.σ²), q ∼ 𝒩(q.μ, q.σ²)
kldivergence_normals(p, q) = (1/2) * ( ( p.σ² + (p.μ - q.μ)^2 ) / q.σ² - log( p.σ² / q.σ² ) - 1 )

# mode-seeking KL divergence H(p̃_τ̃ | p̄_τ)
kldivergence_normals(τ̃, τ, p) = kldivergence_normals((μ=mutilde(τ̃, p), σ²=sigmatildesq(τ̃, p)), (μ=p.μ₀, σ²=sigmabarsq(τ, p)))
#kldivergence_normals(τ̃, τ, p) = (1/2) * ( ( sigmatildesq(τ̃, p) + ( p.μ₀ - mutilde(τ̃, p) )^2 ) / sigmabarsq(τ, p) - log( sigmatildesq(τ̃, p) / sigmabarsq(τ, p) ) - 1 )

# cover KL divergence H(p̄_τ | p̃_τ̃)
kldivergencerev_normals(τ̃, τ, p) = kldivergence_normals((μ=p.μ₀, σ²=sigmabarsq(τ, p)), (μ=mutilde(τ̃, p), σ²=sigmatildesq(τ̃, p)))

# KL evolution terms (Sec. evolution: dH/dτ = r_a - r_b, with r_a = ½ ḡ² r_e, r_b = ½ γ ḡ² (r_d - r_e)).
# Notation: sigmabarsq(τ) = σ̄(τ)², sigmatildesq(τ,p) = σ̃_θ(τ)², mutilde(τ,p) = μ̃_θ(τ).
# Denominators like sigmabarsq^2 mean σ̄⁴ because sigmabarsq already stores variance σ̄².
klrate_d(τ, p) = ( 1 / sigmatildesq(τ, p) - 1 / sigmabarsq(τ, p) )^2 * sigmatildesq(τ, p) + ( mutilde(τ, p) - p.μ₀ )^2 / sigmabarsq(τ, p)^2   # r_d(τ)

klrate_e(τ, p) = ( 1 / p.αθ - 1 ) * ( 1 / sigmatildesq(τ, p) - 1 / sigmabarsq(τ, p) ) * sigmatildesq(τ, p) / sigmabarsq(τ, p) - ( ( mutilde(τ, p) - p.μθ ) / p.αθ - mutilde(τ, p) + p.μ₀ ) * ( mutilde(τ, p) - p.μ₀ ) / sigmabarsq(τ, p)^2   # r_e(τ)

klrate_a(τ, p) = (1/2) * g(p.T - τ, p)^2 * klrate_e(τ, p)   # r_a(τ); g(T-τ,p) = ḡ(τ)

klrate_b(τ, p) = (1/2) * p.γ * g(p.T - τ, p)^2 * ( klrate_d(τ, p) - klrate_e(τ, p) )   # r_b(τ)

klrate(τ, p) = klrate_a(τ, p) - klrate_b(τ, p)   # dH(p̃_τ|p̄_τ)/dτ

# Bound

exp_alpha(s, τ, p) = ( sigmabarsq(τ, p) / sigmabarsq(s, p) )^p.γ

klbound_a(τ, p) = exp_alpha(zero(τ), τ, p) * kldivergence_normals(zero(τ), zero(τ), p)

klbound_b(τ, p) = (1/2) * sum(s -> g(p.T - s, p)^2 * (1 + p.γ) * klrate_e(s, p) * exp_alpha(s, τ, p), range(zero(τ), τ, step = p.T/1000)) * p.T / 1000

klbound(τ, p) = klbound_a(τ, p) + klbound_b(τ, p)

# Generalized mean and variance on a segment [τ₀, τ] with constant γ,
# starting from mean m₀ and variance v₀ at reverse time τ₀.
function mutilde_segment(τ, τ₀, m₀, γ, p)
    s  = sigmabarsq(τ, p)
    s₀ = sigmabarsq(τ₀, p)
    s₀ ≈ 0.0 && return p.μθ
    return p.μθ + (s / s₀)^((1 + γ) / (2p.αθ)) * (m₀ - p.μθ)
end

function sigmatildesq_segment(τ, τ₀, v₀, γ, p)
    (; αθ) = p
    s  = sigmabarsq(τ, p)
    s ≈ 0.0 && return 0.0
    s₀ = sigmabarsq(τ₀, p)
    s₀ ≈ 0.0 && return 0.0
    r = s₀ / s
    β = v₀ / s₀
    if 1 + γ ≈ αθ
        return s * (β + γ * log(r))
    else
        return s * (γ * αθ / (1 + γ - αθ) + (β - γ * αθ / (1 + γ - αθ)) * r^(1 - (1 + γ) / αθ))
    end
end

# Terminal (mean, variance) for a bang-bang schedule:
#   γ = γ₁ on [0, τs],  γ = γ₂ on [τs, T]
function bangbang_terminal(τs, γ₁, γ₂, p)
    # Initial conditions at τ = 0
    m₀ = p.μT
    v₀ = p.βT * sigmabarsq(0.0, p)

    # Phase 1: γ₁ from 0 to τs
    if τs ≤ 0
        m₁, v₁ = m₀, v₀
    else
        m₁ = mutilde_segment(τs, 0.0, m₀, γ₁, p)
        v₁ = sigmatildesq_segment(τs, 0.0, v₀, γ₁, p)
    end

    # Phase 2: γ₂ from τs to T
    if τs ≥ p.T
        return m₁, v₁
    else
        mf = mutilde_segment(p.T, τs, m₁, γ₂, p)
        vf = sigmatildesq_segment(p.T, τs, v₁, γ₂, p)
        return mf, vf
    end
end

# Optimal bang-bang KL for a given γ_max.
# Searches over both orderings (det→stoch and stoch→det)
# and all switching times τs ∈ [0, T].
# Returns:
#   result.fwd        (optimal forward KL value)
#   result.rev        (optimal reverse KL value)
#   result.τs_fwd     (switching time for forward KL optimum)
#   result.τs_rev     (switching time for reverse KL optimum)
#   result.order_fwd  ((γ₁, γ₂): which ordering, e.g. (0.0, γ_max) = det→stoch)
#   result.order_rev  (same for reverse KL)
function optimal_bangbang_kl(γ_max, p; npts = 100)
    target = (μ = p.μ₀, σ² = sigmabarsq(p.T, p))
    τ_grid = range(0.0, p.T, length = npts)

    best_fwd = Inf
    best_rev = Inf
    τs_fwd = 0.0
    τs_rev = 0.0
    order_fwd = (0.0, γ_max)
    order_rev = (0.0, γ_max)

    for τs in τ_grid
        for (γ₁, γ₂) in ((0.0, γ_max), (γ_max, 0.0))
            mf, vf = bangbang_terminal(τs, γ₁, γ₂, p)
            vf ≤ 0 && continue
            q = (μ = mf, σ² = vf)
            fwd = kldivergence_normals(q, target)
            rev = kldivergence_normals(target, q)
            if fwd < best_fwd
                best_fwd = fwd
                τs_fwd = τs
                order_fwd = (γ₁, γ₂)
            end
            if rev < best_rev
                best_rev = rev
                τs_rev = τs
                order_rev = (γ₁, γ₂)
            end
        end
    end

    return (fwd = best_fwd, rev = best_rev,
            τs_fwd = τs_fwd, τs_rev = τs_rev,
            order_fwd = order_fwd, order_rev = order_rev)
end

# ============================================================
# Final KL divergences for a time-dependent γ(τ)
# ============================================================
#
# γfun(τ) is a function giving the stochasticity parameter at reverse time τ.
# The integration is done by stepping through a fine grid in τ ∈ [0, T]
# and treating γ as constant on each subinterval, using the existing
# `mutilde_segment` and `sigmatildesq_segment` formulas.
#
# Returns (fwd_kl, rev_kl, mf, vf).

function kldivergences_timevarying(γfun, p; npts = 1000)
    τ_grid = range(0.0, p.T, length = npts)

    # Initial condition at τ = 0
    m = p.μT
    v = p.βT * sigmabarsq(0.0, p)

    # Step through the grid, treating γ as constant on each subinterval.
    # On [τₖ, τₖ₊₁], use γ = γfun at the midpoint for a midpoint-rule
    # approximation.
    for k in 1:length(τ_grid)-1
        τ₀ = τ_grid[k]
        τ₁ = τ_grid[k+1]
        γ  = γfun((τ₀ + τ₁) / 2)
        m_new = mutilde_segment(τ₁, τ₀, m, γ, p)
        v_new = sigmatildesq_segment(τ₁, τ₀, v, γ, p)
        m, v = m_new, v_new
    end

    target = (μ = p.μ₀, σ² = sigmabarsq(p.T, p))
    q      = (μ = m,    σ² = v)

    fwd = kldivergence_normals(q, target)
    rev = kldivergence_normals(target, q)

    return (fwd = fwd, rev = rev, mf = m, vf = v)
end

# KL-rate terms evaluated at an explicit Gaussian state (μ, σ²) for p̃_τ.
# These coincide with klrate_e / klrate_d when (μ, σ²) = (mutilde(τ,p), sigmatildesq(τ,p)).
function klrate_e_mv(τ, μ, σ², p)
    s̃ = σ²
    s̄ = sigmabarsq(τ, p)
    (1 / p.αθ - 1) * (1 / s̃ - 1 / s̄) * s̃ / s̄ -
        ( (μ - p.μθ) / p.αθ - μ + p.μ₀ ) * (μ - p.μ₀) / s̄^2
end

function klrate_d_mv(τ, μ, σ², p)
    s̃ = σ²
    s̄ = sigmabarsq(τ, p)
    (1 / s̃ - 1 / s̄)^2 * s̃ + (μ - p.μ₀)^2 / s̄^2
end

# r_e, r_d for dH(p̃_τ|p̄_τ)/dτ (expectations under p̃_τ).
klrate_instantaneous_expectation_fwd(τ, μ, σ², p) = klrate_d_mv(τ, μ, σ², p) - klrate_e_mv(τ, μ, σ², p)

# r_e, r_d for dH(p̄_τ|p̃_τ)/dτ (expectations under p̄_τ).
# With ε(x) = c0 (x - μ₀) + c1 and ∇log(p̃/p̄)(x) = -a (x - μ₀) + d/σ̃²,
# integrating against p̄ = 𝒩(μ₀, σ̄²) gives the closed forms below.
function klrate_rev_e_mv(τ, μ, σ², p)
    s̃ = σ²
    s̄ = sigmabarsq(τ, p)
    d = μ - p.μ₀
    a = 1 / s̃ - 1 / s̄
    c0 = (1 - 1 / p.αθ) / s̄
    c1 = (p.μθ - p.μ₀) / (p.αθ * s̄)
    -c0 * a * s̄ + c1 * d / s̃
end

function klrate_rev_d_mv(τ, μ, σ², p)
    s̃ = σ²
    s̄ = sigmabarsq(τ, p)
    a = 1 / s̃ - 1 / s̄
    d = μ - p.μ₀
    a^2 * s̄ + d^2 / s̃^2
end

klrate_instantaneous_expectation_rev(τ, μ, σ², p) =
    klrate_rev_d_mv(τ, μ, σ², p) - klrate_rev_e_mv(τ, μ, σ², p)

# Bang-bang γ*(τ): minimize dH/dτ at each τ; γ_max when r_d > r_e (same sign logic for fwd and rev).
optimal_instantaneous_gamma(τ, μ, σ², p, γ_max; direction = :fwd) =
    (direction == :fwd ? klrate_instantaneous_expectation_fwd : klrate_instantaneous_expectation_rev)(τ, μ, σ², p) > 0 ? γ_max : 0.0

# Piecewise-constant myopic schedule on a τ grid; propagates (μ, σ²) with the segment formulas.
function instantaneous_gamma_schedule(γ_max, p; npts = 1000, direction = :fwd)
    τ_grid = collect(range(0.0, p.T, length = npts))
    γ_piece = zeros(npts - 1)

    m = p.μT
    v = p.βT * sigmabarsq(0.0, p)

    for k in 1:npts - 1
        τ₀ = τ_grid[k]
        τ₁ = τ_grid[k + 1]
        τ_mid = (τ₀ + τ₁) / 2
        γ_piece[k] = optimal_instantaneous_gamma(τ_mid, m, v, p, γ_max; direction = direction)
        m = mutilde_segment(τ₁, τ₀, m, γ_piece[k], p)
        v = sigmatildesq_segment(τ₁, τ₀, v, γ_piece[k], p)
    end

    γfun(τ) = begin
        τ ≤ 0 && return γ_piece[1]
        τ ≥ p.T && return γ_piece[end]
        for k in 1:npts - 1
            τ ≤ τ_grid[k + 1] && return γ_piece[k]
        end
        γ_piece[end]
    end

    return (γfun = γfun, τ_grid = τ_grid, γ_piece = γ_piece, mf = m, vf = v)
end

# Bang-bang γ(τ) from optimal_bangbang_kl (single switch, :fwd or :rev objective).
function bangbang_gammafun(result_bb, direction::Symbol)
    τs = direction == :fwd ? result_bb.τs_fwd : result_bb.τs_rev
    order = direction == :fwd ? result_bb.order_fwd : result_bb.order_rev
    γfun(τ) = τ < τs ? order[1] : order[2]
    return γfun
end

# H(p̃_τ | p̄_τ) (or reverse) along a time-dependent γ(τ) schedule.
function kldivergence_evolution(γfun, p; npts = 1000, direction = :fwd)
    τ_grid = range(0.0, p.T, length = npts)
    target_at(τ) = (μ = p.μ₀, σ² = sigmabarsq(τ, p))

    m = p.μT
    v = p.βT * sigmabarsq(0.0, p)
    kl_vals = Vector{Float64}(undef, npts)

    q = (μ = m, σ² = v)
    kl_vals[1] = direction == :fwd ?
        kldivergence_normals(q, target_at(0.0)) :
        kldivergence_normals(target_at(0.0), q)

    for k in 1:length(τ_grid) - 1
        τ₀ = τ_grid[k]
        τ₁ = τ_grid[k + 1]
        γ  = γfun((τ₀ + τ₁) / 2)
        m = mutilde_segment(τ₁, τ₀, m, γ, p)
        v = sigmatildesq_segment(τ₁, τ₀, v, γ, p)
        q = (μ = m, σ² = v)
        kl_vals[k + 1] = direction == :fwd ?
            kldivergence_normals(q, target_at(τ₁)) :
            kldivergence_normals(target_at(τ₁), q)
    end

    return (τ = collect(τ_grid), kl = kl_vals)
end

# Monte-Carlo on forward diffustion
function montecarlo_fwd!(xfwd, p)
    @assert axes(xfwd) isa Tuple{Base.OneTo, Base.OneTo}
    (; μ₀, σ₀, T) = p
    n = size(xfwd, 2)
    dt = T / ( n - 1 )
    deltaw = view(xfwd, :, n)
    randn!(deltaw)
    xfwd[:, 1] .= μ₀ .+ σ₀ .* deltaw
    for j in 2:n
        t = (j - 1) * dt
        randn!(deltaw)
        xfwd[:, j] .= view(xfwd, :, j-1) .+ g(t, p) * √dt .* deltaw
    end
end

# Monte-Carlo on reverse diffusion
function montecarlo_reverse!(xtilde, p)
    @assert axes(xtilde) isa Tuple{Base.OneTo, Base.OneTo}
    (; μθ, αθ, μT, βT, γ, T) = p
    n = size(xtilde, 2)
    dt = T / ( n - 1 ) 
    deltaw = view(xtilde, :, n)
    randn!(deltaw)
    xtilde[:, 1] .= μT .+ √(βT * sigmabarsq(0, p)) .* deltaw
    for j in 2:n
        τ = (j - 1) * dt
        randn!(deltaw)
        a = g(T - τ, p)^2 * (1 + γ) / ( 2 * αθ * sigmabarsq(τ, p) )
        xtilde[:, j] .= view(xtilde, :, j-1) .- dt * a .* ( view(xtilde, :, j-1) .- μθ ) .+ g(T - τ, p) * √(γ * dt) .* deltaw
    end
end

# Parameters

## Fixed parameters
begin
    μ₀ = 0.0
    σ₀ = 1.0
    σ = 1.0
    T = 6.0
end

## Meshes and caches
begin
    n = 200
    m = 400

    xfwd = zeros(m, n)
    xtilde = zeros(m, n)

    tt = range(0.0, T, length=n)
end

## Parameter container
begin
    prms = (
        μ₀ = μ₀,
        σ₀ = σ₀,
        σ = σ,
        T = T,
        μθ = μ₀,
        αθ = 1.0,
        μT = μ₀,
        βT = 1.0,
        γ = 1.0,
        schd = :edm
    )
end

## parameter ranges
begin
    μθs = Tuple(μ₀ + δ for δ in (-0.3, 0.0, 0.3))
    αθs = (0.8, 1.0, 1.25)

    μTs = Tuple(μ₀ + δ for δ in (-0.3, 0.0, 0.3))
    βTs = (0.8, 1.0, 1.25)

    μθs = Tuple(μ₀ + δ for δ in (-0.3, 0.0, 0.3))
    αθs = (0.64, 1.0, 1.48)
    μTs = Tuple(μ₀ + δ for δ in (-0.22, 0.0, 0.22))
    βTs = (0.72, 1.0, 1.34)

    μθs = Tuple(μ₀ + δ for δ in (-1.0, 0.0, 1.0))
    αθs = (0.7, 1.0, 1.4)
    μTs = Tuple(μ₀ + δ for δ in (-0.5, 0.0, 0.5))
    βTs = (0.82, 1.0, 1.2)

    # Bernardo:
    μθs = Tuple(μ₀ + δ for δ in (-0.4, 0.0, 0.4))
    αθs = (0.9, 1.0, 1.15)
    μTs = Tuple(μ₀ + δ for δ in (-0.5, 0.0, 0.4))
    βTs = (0.7, 1.0, 1.3)

    # assymetric μTs and more dramatic variations
    μθs = Tuple(μ₀ + δ for δ in (-0.2, 0.0, 0.2))
    αθs = (0.7, 1.0, 1.3)
    μTs = Tuple(μ₀ + δ for δ in (0.0, 0.5))
    βTs = (0.6, 1.0, 1.4)

    # assymetric μTs and more dramatic variations 2
    μθs = Tuple(μ₀ + δ for δ in (-0.4, 0.0, 0.4))
    αθs = (0.7, 1.0, 1.3)
    μTs = Tuple(μ₀ + δ for δ in (0.0, 1.0))
    βTs = (0.6, 1.0, 1.4)

    γs = (0.0, 0.3, 1.0, 5.0)

    schds = (:ve, :edm)
end

## save figs
begin
    savepdfs = false
    savepngs = true
    alignment_onlysign = true   # musigma inset: true → sign(r_e); false → r_e
    mkpath(joinpath(@__DIR__, "figures"))
end

# run tests
begin
    runtests = false
end

# # KL Errors in score and starting distribution

# begin
#     tabthetat0 = pretty_table(
#         [collect(μθs) [kldivergence_normals((μ=μθ, σ²=αθ*prms.σ₀), (μ=prms.μ₀, σ²=prms.σ₀)) for μθ in μθs, αθ in αθs]];
#         column_labels = ["H(pθ₀ || p₀)"; ["αθ = $αθ" for αθ in αθs]]
#     )
#     tabthetaT = pretty_table(
#         [collect(μθs) [kldivergence_normals((μ=μθ, σ²=αθ*sigmasq(T, prms)), (μ=prms.μ₀, σ²=sigmasq(T, prms))) for μθ in μθs, αθ in αθs]];
#         column_labels = ["H(pθ_T || p_T)"; ["αθ = $αθ" for αθ in αθs]]
#     )
#     tabT = pretty_table(
#         [collect(μTs) [kldivergence_normals((μ=μT, σ²=βT*sigmasq(T, prms)), (μ=prms.μ₀, σ²=sigmasq(T, prms))) for μT in μTs, βT in βTs]];
#         column_labels = ["H(p̃_T || p_T)"; ["βT = $βT" for βT in βTs]]
#     )
# end

# # Figures - PDFs
# begin
#     let schd = :edm
#         diffusion_schedule = uppercase(String(schd))

#         local f = Figure(size = (1200, 1000))
#         local titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"PDFs", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"for assorted values of the parameters $\mu_\theta,$ $\alpha_\theta,$ $\beta_T,$ and $\mu_T$", halign = :left, fontsize = 20)
#         ax = Axis(
#             f[1, 1], 
#             title = L"PDF (%$(diffusion_schedule))",
#             xlabel = L"x",
#             ylabel = L"y"
#         )
#         lines!(ax, range(μ₀ - 5, μ₀ + 5, length=200), x -> pdf(Normal(prms.μ₀, sqrt(prms.σ₀^2)), x))
#         for (μθ, αθ) in Iterators.product(extrema(μθs), extrema(αθs))
#             lines!(ax, range(μ₀ - 5, μ₀ + 5, length=200), x -> pdf(Normal(μθ, sqrt(αθ*prms.σ₀^2)), x))
#         end
#         #Legend(titlelayout[3, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)
#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_pdfs_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_pdfs_$(diffusion_schedule).png"), f)
#         end
#     end
# end

# # Figures - forward and exact-score reverse sample paths
# begin
#     local f = Figure(size=(1000, 400 + 300 * length(filter(<(10), γs))))

#     local titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#     Label(titlelayout[1, 1], "Forward sample paths and exact-score reverse sample paths", halign = :left, fontsize = 24)

#     for (j, schd) in enumerate(schds)
#         diffusion_schedule = uppercase(String(schd))
#         p =(; prms..., schd = schd)
#         ax = Axis(
#             f[1, j], 
#             title = L"Forward sample paths $\sigma_0 = $ %$(σ₀), $\sigma = $ %$(σ) (%$(diffusion_schedule))",
#             limits = (0.0, T, - 2.8 * sqrt(sigmasq(T, p)), 2.8 * sqrt(sigmasq(T, p))),
#             xlabel = L"t",
#             ylabel = L"x"
#         )
#         montecarlo_fwd!(xfwd, p)
#         for k in axes(xfwd, 1)
#             lines!(ax, tt, view(xfwd, k, :), linewidth=0.2)
#         end
#         lines!(ax, tt, t -> μ₀ .+ √sigmasq(t, p), color=:black,label=L"theoretical $\pm$std")
#         lines!(ax, tt, t -> μ₀ .- √sigmasq(t, p), color=:black)
#         lines!(ax, tt, t -> μ₀ .+ 2√sigmasq(t, p), color=:red, label=L"theoretical $\pm2$std")
#         lines!(ax, tt, t -> μ₀ .- 2√sigmasq(t, p), color=:red)
#         samplemean = vec(mean(xfwd, dims=1))
#         samplestd = vec(sqrt.(var(xfwd, dims=1)))
#         lines!(ax, tt, samplemean .+ samplestd, color=:orange, label=L"sample paths $\pm$std")
#         lines!(ax, tt, samplemean .- samplestd, color=:orange)
#     end

#     local lks = Any[1, 2, 3]
#     for (j, schd) in enumerate(schds)
#         diffusion_schedule = uppercase(String(schd))
#         for (i, γ) in enumerate(filter(<(10), γs))
#             p =(
#                 ; 
#                 prms...,
#                 αθ = 1.0,
#                 βT = 1.0,
#                 μθ = prms.μ₀,
#                 μT = prms.μ₀,
#                 γ = γ,
#                 schd = schd
#             )
#             ax = Axis(
#                 f[i+1, j], 
#                 limits = (0.0, T, - 2.8 * sqrt(sigmabarsq(0, p)), 2.8 * sqrt(sigmabarsq(0, p))),
#                 title = L"Reverse sample paths with exact score and prior, $\gamma = $ %$(γ) (%$(diffusion_schedule))",
#                 xlabel = L"t",
#                 ylabel = L"x"
#             )

#             montecarlo_reverse!(xtilde, p)
#             for k in axes(xtilde, 1)
#                 lines!(ax, tt, view(xtilde, k, Iterators.reverse(axes(xtilde, 2))), linewidth=0.2)
#             end
#             lks[1] = lines!(ax, tt, t -> mutilde(T-t, p) + √sigmatildesq(T-t, p), color=:black)
#             lines!(ax, tt, t -> mutilde(T-t, p) - √sigmatildesq(T-t, p), color=:black)
#             lks[2] = lines!(ax, tt, t -> mutilde(T-t, p) + 2√sigmatildesq(T-t, p), color=:red)
#             lines!(ax, tt, t -> mutilde(T-t, p) - 2√sigmatildesq(T-t, p), color=:red)
#             samplemean = vec(collect(Iterators.reverse(mean(xtilde, dims=1))))
#             samplestd = vec(sqrt.(Iterators.reverse(var(xtilde, dims=1))))
#             lks[3] = lines!(ax, tt, samplemean .+ samplestd, color=:orange)
#             lines!(ax, tt, samplemean .- samplestd, color=:orange)
#         end
#     end

#     Legend(titlelayout[2, 1], lks, [L"theoretical $\pm$std", L"theoretical $\pm2$std", L"sample paths $\pm$std"], orientation = :horizontal, halign = :left, framevisible = false)

#     display(f)
#     if savepdfs
#         save(joinpath(@__DIR__(), "figures", "montecarlo_fwd_exactbwd.pdf"), f)
#     end
#     if savepngs
#         save(joinpath(@__DIR__(), "figures", "montecarlo_fwd_exactbwd.png"), f)
#     end
# end

# # Figures - approximate-score reverse sample paths
# begin
#     for schd in schds
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size=(200 + 300 * length(αθs), 100 + 200 * length(filter(<(10), γs))))

#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], "Approximate-score reverse sample paths", halign = :left, fontsize = 24)

#         lks = Any[1, 2, 3]
#         for (j, (βT, αθ)) in enumerate(Iterators.product(extrema(βTs), extrema(αθs)))
#             for (i, γ) in enumerate(filter(<(10), γs))
#                 p =(
#                     ; 
#                     prms...,
#                     αθ = αθ,
#                     βT = βT,
#                     μθ = prms.μ₀,
#                     μT = prms.μ₀,
#                     γ = γ,
#                     schd = schd
#                 )
#                 ax = Axis(
#                     f[i, j], 
#                     limits = (0.0, T, - 2.8 * sqrt(sigmabarsq(0, p)), 2.8 * sqrt(sigmabarsq(0, p))),
#                     title = L"$\alpha = $ %$(αθ), $\beta_T = $ %$(βT), $\gamma = $ %$(γ) (%$(diffusion_schedule))",
#                     xlabel = L"t",
#                     ylabel = L"x"
#                 )
#                 montecarlo_reverse!(xtilde, p)
#                 for i in axes(xtilde, 1)
#                     lines!(ax, tt, view(xtilde, i, Iterators.reverse(axes(xtilde, 2))), linewidth=0.2)
#                 end
#                 lks[1] = lines!(ax, tt, t -> mutilde(T-t, p) + √sigmatildesq(T-t, p), color=:black, label=L"theoretical $\pm$std")
#                 lines!(ax, tt, t -> mutilde(T-t, p) - √sigmatildesq(T-t, p), color=:black)
#                 lks[2] = lines!(ax, tt, t -> mutilde(T-t, p) + 2√sigmatildesq(T-t, p), color=:red, label=L"theoretical $\pm2$std")
#                 lines!(ax, tt, t -> mutilde(T-t, p) - 2√sigmatildesq(T-t, p), color=:red)
#                 samplemean = vec(collect(Iterators.reverse(mean(xtilde, dims=1))))
#                 samplestd = vec(sqrt.(Iterators.reverse(var(xtilde, dims=1))))
#                 lks[3] = lines!(ax, tt, samplemean .+ samplestd, color=:orange, label=L"sample paths $\pm$std")
#                 lines!(ax, tt, samplemean .- samplestd, color=:orange)
#             end
#         end

#         Legend(titlelayout[2, 1], lks, [L"theoretical $\pm$std", L"theoretical $\pm2$std", L"sample paths $\pm$std"], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "montecarlo_approxbwd_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "montecarlo_approxbwd_$(diffusion_schedule).png"), f)
#         end
#     end
# end

# # Figures - mse score
# begin
#     local f = Figure(size=(800, 300))
#     for (j, schd) in enumerate(schds)
#         diffusion_schedule = uppercase(String(schd))
#         p =(
#             ; 
#             prms...,
#             αθ = 1.0,
#             βT = 1.0,
#             μθ = prms.μ₀,
#             μT = prms.μ₀,
#             γ = 1.0,
#             schd = schd
#         )
#         ax = Axis(
#             f[1, j], 
#             title = L"Score mse with $\sigma_0 = $ %$(σ₀), $\sigma = $ %$(σ) (%$(diffusion_schedule))",
#             xlabel = L"t",
#             ylabel = L"\mathrm{mse}"
#         )
#         for αθ in αθs 
#             p =(
#                 ; 
#                 p...,
#                 αθ = αθ
#             )   
#             lines!(ax, tt, t -> score_error_mse(t, p), label=L"$\alpha = $ %$αθ")
#         end
#         axislegend( position = :rt)
#     end
#     display(f)
#     if savepdfs
#         save(joinpath(@__DIR__(), "figures", "analytic_mse_scores.pdf"), f)
#     end
#     if savepngs
#         save(joinpath(@__DIR__(), "figures", "analytic_mse_scores.png"), f)
#     end
# end
# nothing

# # Figures - KL divergence $H(\tilde{p}_t | \bar{p}_t)$
# begin
#     for schd in schds
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (1200, 1200))

#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"KL divergence $H(\tilde{p}_t | \bar{p}_t),$ with $\mu_0 = $ %$μ₀, $\sigma_0 = $ %$σ₀, $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 24)

#         lks = Any[γs...]

#         for (i, (μθ, αθ)) in enumerate(Iterators.product(extrema(μθs), extrema(αθs)))
#             for (j, (μT, βT)) in enumerate(Iterators.product(extrema(μTs), extrema(βTs)))
#                 #μθ = rand(μθs)
#                 #μT = rand(μTs)
#                 ax = Axis(
#                     f[i, j], 
#                     title = L"$\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ, $\mu_T = $ %$μT, $\beta_T = $ %$βT",
#                     limits = (0.0, T, -0.005, 0.2),
#                     xlabel = L"t",
#                     ylabel = L"\mathrm{entropy}",
#                     #yscale = log10
#                 )
#                 p = (
#                     ; 
#                     prms...,
#                     αθ = αθ,
#                     βT = βT,
#                     μθ = μθ,
#                     μT = μT,
#                     g = g,
#                     σsq = sigmasq,
#                     schd = schd
#                 )
#                 for (k, γ) in enumerate(γs)
#                     p =(; p..., γ = γ)
#                     lks[k] = lines!(f[i, j], tt, t -> kldivergence_normals(T - t, T - t, p), label=L"$\gamma = $ %$γ")
#                 end
#             end
#         end

#         Legend(titlelayout[2, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_KLdivergence_along_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_KLdivergence_along_$(diffusion_schedule).png"), f)
#         end
#     end
# end
# nothing

# # Figures - KL divergence $H(\tilde{p}_t | \bar{p}_t)$ with KL bound
# begin
#     for schd in schds
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (1200, 1200))

#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"KL divergence $H(\tilde{p}_t | \bar{p}_t),$ with $\mu_0 = $ %$μ₀, $\sigma_0 = $ %$σ₀, $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"KL divergence$$ (solid), \;\; KL bound$$ (dashed)", halign = :left, fontsize = 24)

#         local k0 = 0
#         local γs = (0.0, 1.0, 5.0,)
#         lks = Any[γs...]

#         for (i, (μθ, αθ)) in enumerate(Iterators.product(extrema(μθs), extrema(αθs)))
#             for (j, (μT, βT)) in enumerate(Iterators.product(extrema(μTs), extrema(βTs)))
#                 #μθ = rand(μθs)
#                 #μT = rand(μTs)
#                 ax = Axis(
#                     f[i, j], 
#                     title = L"$\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ, $\mu_T = $ %$μT, $\beta_T = $ %$βT",
#                     limits = (0.0, T, -0.005, 0.2),
#                     xlabel = L"t",
#                     ylabel = L"\mathrm{entropy}",
#                     #yscale = log10
#                 )
#                 p = (
#                     ; 
#                     prms...,
#                     αθ = αθ,
#                     βT = βT,
#                     μθ = μθ,
#                     μT = μT,
#                     g = g,
#                     σsq = sigmasq,
#                     schd = schd
#                 )
#                 for (k, γ) in enumerate(γs)
#                     p =(; p..., γ = γ)
#                     lks[k] = lines!(f[i, j], tt, t -> kldivergence_normals(T - t, T - t, p), label=L"$\gamma = $ %$γ", color=Cycled(k0 + k))
#                     lines!(f[i, j], tt, t -> klbound(T-t, p), linestyle = (:dash, :dense), color=Cycled(k0 + k))
#                 end
#             end
#         end

#         Legend(titlelayout[3, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_KLdivergence_with_klbound_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_KLdivergence_with_klbound_$(diffusion_schedule).png"), f)
#         end
#     end
# end
# nothing

# # Figures - KL divergence $H(\tilde{p}_t | \bar{p}_t)$ with KL bound with a single parameter set
# begin
#     let schd = :edm, σ = 1.0, T = 2.0, ττ = range(0.0, T, length=n)
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (800, 600))

#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"KL divergence $H(\tilde{p}_t | \bar{p}_t),$ with $\mu_0 = $ %$μ₀, $\sigma_0 = $ %$σ₀, $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 16)
#         Label(titlelayout[2, 1], L"KL divergence$$ (solid), \;\; KL bound$$ (dashed)", halign = :left, fontsize = 16)

#         k0 = 1
#         local γs = (2.0,)
#         lks = Any[γs...]

#         μθ = prms.μ₀
#         μT = prms.μ₀

#         αθ = 1.3
#         βT = 1.1

#         αθ = 0.7
#         βT = 0.5

#         ax = Axis(
#             f[1, 1], 
#             title = L"$\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ, $\mu_T = $ %$μT, $\beta_T = $ %$βT",
#             limits = (0.0, T, -0.005, 0.2),
#             xlabel = L"τ",
#             ylabel = L"\mathrm{entropy}",
#         )
#         p = (
#             ; 
#             prms...,
#             σ = σ,
#             αθ = αθ,
#             βT = βT,
#             μθ = μθ,
#             μT = μT,
#             T = T,
#             g = g,
#             σsq = sigmasq,
#             schd = schd
#         )
#         for (k, γ) in enumerate(γs)
#             p =(; p..., γ = γ)
#             lks[k] = lines!(f[1, 1], ττ, τ -> kldivergence_normals(τ, τ, p), label=L"$\gamma = $ %$γ", color=Cycled(k0 + k))
#             lines!(f[1, 1], ττ, τ -> klbound(τ, p), linestyle = (:dash, :dense), color=Cycled(k0 + k))
#         end

#         Legend(titlelayout[3, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#     end
# end
# nothing

# # Figures - KL divergence $H(\tilde{p}_t | \bar{p}_0)$
# begin
#     for schd in schds
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (1200, 1200))

#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"KL divergence $H(\tilde{p}_t | \bar{p}_0)$ with $\mu_0 = $ %$μ₀, $\sigma_0 = $ %$σ₀, $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 24)

#         lks = Any[γs...]

#         for (i, (μθ, αθ)) in enumerate(Iterators.product(extrema(μθs), extrema(αθs)))
#             for (j, (μT, βT)) in enumerate(Iterators.product(extrema(μTs), extrema(βTs)))
#                 ax = Axis(
#                     f[i, j], 
#                     title = L"$\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ, $\mu_T = $ %$μT, $\beta_T = $ %$βT",
#                     limits = (0.0, T, -0.005, 1.0),
#                     xlabel = L"t",
#                     ylabel = L"\mathrm{entropy}",
#                     #yscale = log10
#                 )
#                 p = (
#                     ; 
#                     prms...,
#                     αθ = αθ,
#                     βT = βT,
#                     μθ = μθ,
#                     μT = μT,
#                     schd = schd
#                 )
#                 for (k, γ) in enumerate(γs)
#                     p =(; p..., γ = γ)
#                     lks[k] = lines!(f[i, j], tt, t -> kldivergence_normals(T - t, T, p), label=L"$\gamma = $ %$γ")
#                 end
#             end
#         end

#         Legend(titlelayout[2, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_KLdivergence_data_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_KLdivergence_data_$(diffusion_schedule).png"), f)
#         end
#     end
# end
# nothing

# # Figures - specific stochasticity growth/decay rates
# begin
#     for schd in schds
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (1200, 1200))
#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"Specific growth/decay rates $$ (%$diffusion_schedule)", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"growth term $r_a(\tau) / r_d(\tau)$ (solid), \;\; decay term $r_b(\tau) / r_d(\tau)$ (dashed)", halign = :left, fontsize = 24) # ", \;\;and total (dotted)"
#         #rowgap!(titlelayout, 0)

#         lks = Any[γs...]
#         for (i, (μθ, αθ)) in enumerate(Iterators.product(extrema(μθs), extrema(αθs)))
#             for (j, (μT, βT)) in enumerate(Iterators.product(extrema(μTs), extrema(βTs)))
#                 ax = Axis(
#                     f[i, j], 
#                     title = L"$\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ, $\mu_T = $ %$μT, $\beta_T = $ %$βT",
#                     limits = (0.0, T, -50.0, 50.0),
#                     xlabel = L"t",
#                     ylabel = L"rate$$"
#                 )
#                 p = (
#                     ; 
#                     prms...,
#                     αθ = αθ,
#                     βT = βT,
#                     μθ = μθ,
#                     μT = μT,
#                     schd = schd
#                 )
#                 for (k, γ) in enumerate(γs)
#                     p =(; p..., γ = γ)
#                     lks[k] = lines!(f[i, j], tt, t -> klrate_a(T - t, p) / klrate_d(T - t, p), color=Cycled(k))
#                     lines!(f[i, j], tt, t -> klrate_b(T - t, p) / klrate_d(T - t, p), linestyle = (:dash, :dense), color=Cycled(k))
#                 end
#             end
#         end

#         Legend(titlelayout[3, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_specific_growth_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_specific_growth_$(diffusion_schedule).png"), f)
#         end
#     end
# end
# nothing

product_names_and_ranges = (
    #("full", (Iterators.product(αθs, μθs, βTs, μTs))),
    #("extrema", (Iterators.product(extrema(αθs), extrema(μθs), βTs, μTs))),
    #("zero_mean", (Iterators.product(αθs, 0.0, βTs, 0.0))),
    #("exact_score", (Iterators.product(1.0, 0.0, βTs, μTs))),
    ("minimal", (
        (first(αθs), last(μθs), round(first(βTs)/3, sigdigits=1), last(μTs)),
        (last(αθs), last(μθs), last(βTs), last(μTs)),
        (1.1, last(μθs), 0.1, last(μTs)),
        (last(αθs), last(μθs), first(βTs), last(μTs)),
        (0.9, first(μθs), 0.2, last(μTs)),
        (1.1, first(μθs), 1.2, last(μTs)),
        (1.1, first(μθs), 0.2, last(μTs)),
        (0.9, first(μθs), 1.2, last(μTs)),
    )
    ),
)

# # Figures - alignment sign(r_e) — now inset in musigma evolution figure below
# begin
#     ...
# end
# nothing

# # Figures - absolute growth/decay rate
# begin
#     for schd in schds
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (1200, 1200))
#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"(Absolute) decay/growth rates $$ (%$diffusion_schedule)", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"decay term $r_b(\tau)$ (solid), \;\; growth term $r_a(\tau)$ (dashed)", halign = :left, fontsize = 24)

#         lks = Any[γs...]
#         for (i, (μθ, αθ)) in enumerate(Iterators.product(extrema(μθs), extrema(αθs)))
#             for (j, (μT, βT)) in enumerate(Iterators.product(extrema(μTs), extrema(βTs)))
#                 ax = Axis(
#                     f[i, j], 
#                     title = L"$\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ, $\mu_T = $ %$μT, $\beta_T = $ %$βT",
#                     limits = (0.0, T, -2αθ*βT, 2αθ*βT),
#                     xlabel = L"t",
#                     ylabel = L"rate$$"
#                 )
#                 p = (
#                     ; 
#                     prms...,
#                     αθ = αθ,
#                     βT = βT,
#                     μθ = μθ,
#                     μT = μT,
#                     schd = schd
#                 )
#                 for (k, γ) in enumerate(γs)
#                     p =(; p..., γ = γ)
#                     lks[k] = lines!(f[i, j], tt, t -> klrate_a(T - t, p), color=Cycled(k))
#                     lines!(f[i, j], tt, t -> klrate_b(T - t, p), linestyle = (:dash, :dense), color=Cycled(k))
#                 end
#             end
#         end

#         Legend(titlelayout[3, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_absolute_growth_stochasticity_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_absolute_growth_stochasticity_$(diffusion_schedule).png"), f)
#         end
#     end
# end
# nothing

# # Figures - total KL evolution rate
# begin
#     for schd in schds
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (1200, 1200))
#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"Total rate $$ (%$diffusion_schedule)", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"rate term $dH(\tilde{p}_\tau | \bar{p}_\tau)/d\tau = r_b(\tau) - r_a(\tau)$", halign = :left, fontsize = 24)

#         lks = Any[γs...]
#         for (i, (μθ, αθ)) in enumerate(Iterators.product(extrema(μθs), extrema(αθs)))
#             for (j, (μT, βT)) in enumerate(Iterators.product(extrema(μTs), extrema(βTs)))
#                 ax = Axis(
#                     f[i, j], 
#                     title = L"$\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ, $\mu_T = $ %$μT, $\beta_T = $ %$βT",
#                     limits = (0.0, T, -αθ*βT, αθ*βT),
#                     xlabel = L"t",
#                     ylabel = L"rate$$"
#                 )
#                 p = (
#                     ; 
#                     prms...,
#                     αθ = αθ,
#                     βT = βT,
#                     μθ = μθ,
#                     μT = μT,
#                     schd = schd
#                 )
#                 for (k, γ) in enumerate(γs)
#                     p =(; p..., γ = γ)
#                     lks[k] = lines!(f[i, j], tt, t -> klrate_a(T - t, p) - klrate_b(T - t, p), color=Cycled(k))
#                 end
#             end
#         end

#         Legend(titlelayout[3, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_klevolutionrate_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_klevolutionrate_$(diffusion_schedule).png"), f)
#         end
#     end
# end
# nothing

# Figures - Evolution in parameter space

let trimarker = BezierPath([MoveTo(0, 0), LineTo(0.5, -1), LineTo(-0.5, -1), ClosePath()])
    for schd in schds
        for (product_name, product_range) in product_names_and_ranges
            diffusion_schedule = uppercase(String(schd))
            f = Figure(size = (1600, 80 * ( 2 + length(product_range))))

            titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
            Label(titlelayout[1, 1], L"Evolution in parameter space $(\mu,\,\sigma)$ with $\mu_0 = $ %$μ₀, $\sigma_0 = $ %$σ₀, $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 28)
            Label(titlelayout[2, 1], alignment_onlysign ?
                L"with the contour levels of $H(\tilde{p}_T | \bar{p}_0)$; \;\; inset: $\mathrm{sign}\, r_e(\tau)$" :
                L"with the contour levels of $H(\tilde{p}_T | \bar{p}_0)$; \;\; inset: $r_e(\tau)$", halign = :left, fontsize = 24)

            lks = Any[γs..., 1]
            γrange = LinRange(0.0, 100.0, 100)
            
            for (k, (αθ, μθ, βT, μT)) in enumerate(product_range)

                i = (k - 1) ÷ 4 + 1   # runs 1:4
                j = (k - 1) % 4 + 1   # runs 1:4

                μmin = min(first(μTs), first(μθs))
                μmax = max(last(μTs), last(μθs))
                μspan = max(last(μTs), last(μθs)) - μmin
                μs = LinRange(μmin - 0.5μspan, μmax + 0.5μspan, 200)
                σs = LinRange(0.1σ₀, 1.2√(σ₀^2 + σ^2 * max(T, T^2)), 200)
                ax_aspect = ( last(σs) - first(σs) ) / ( last(μs) - first(μs))
                ax = Axis(
                    f[i, j],
                    title = L"$\mu_T = $ %$μT, $\beta_T = $ %$βT, $\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ",
                    limits = (extrema(μs)...,  extrema(σs)...),
                    xlabel = L"\mu",
                    ylabel = L"\sigma",
                    xlabelsize = 16,
                    ylabelsize = 16,
                    titlesize = 19,
                    #yscale = log10
                )
                p = (
                    ; 
                    prms...,
                    αθ = αθ,
                    βT = βT,
                    μθ = μθ,
                    μT = μT,
                    g = g,
                    σsq = sigmasq,
                    schd = schd
                )

                inset_common = (
                    width = Relative(0.2),
                    height = Relative(0.2),
                    halign = 0.12,
                    valign = 0.84,
                    xlabel = L"t",
                    xlabelsize = 12,
                    ylabelsize = 12,
                    titlealign = :center,
                    titlesize = 12,
                    xticks = 0:prms.T / 3:prms.T,
                    xticklabelsize = 10,
                    yticklabelsize = 10,
                )
                ax_inset = if alignment_onlysign
                    Axis(
                        f[i, j];
                        inset_common...,
                        title = L"$\mathrm{sign}\, r_e(\tau)$",
                        yticks = [-1, 0, 1],
                        limits = (0.0, prms.T, -1.2, 1.2),
                    )
                else
                    Axis(
                        f[i, j];
                        inset_common...,
                        title = L"$r_e(\tau)$",
                        limits = (0.0, prms.T, nothing, nothing),
                    )
                end
                for (kγ, γ) in enumerate(γs)
                    pγ = (; p..., γ = γ)
                    let pγ = pγ, kγ = kγ
                        if alignment_onlysign
                            lines!(ax_inset, tt, t -> sign(klrate_e(T - t, pγ)), color = Cycled(kγ))
                        else
                            lines!(ax_inset, tt, t -> klrate_e(T - t, pγ), color = Cycled(kγ))
                        end
                    end
                end

                kldivs = [kldivergence_normals((μ=x, σ²=y^2),(μ=μ₀, σ²=σ₀^2)) for x in μs, y in σs]
                #heatmap!(ax, μs, σs, kldivs, alpha=0.7)
                contour!(ax, μs, σs, kldivs, labels=true, levels = exp.(-4.0:0.5:1.0), alpha=0.6)
                for (kγ, γ) in enumerate(γs)
                    pγ = (; p..., γ = γ)
                    lks[kγ] = lines!(ax, [mutilde(T - t, pγ) for t in tt], [sqrt(sigmatildesq(T - t, pγ)) for t in tt], color=Cycled(kγ), linewidth=3)
                    vecx = ax_aspect * ( mutilde(T, pγ) - mutilde(0.96T, pγ) )
                    vecy = sqrt(sigmatildesq(T, pγ)) - sqrt(sigmatildesq(0.96T, pγ))
                    vecnorm = sqrt(vecx^2 + vecy^2)
                    arrowangle = vecx ≈ 0.0 ? ( 1 - sign(vecy) ) * π/2 : - sign(vecx) * acos( vecy / vecnorm )
                    scatter!(ax, [mutilde(0, pγ)], [sqrt(sigmatildesq(0, pγ))], color=Cycled(kγ))
                    scatter!(ax, [mutilde(T, pγ)], [sqrt(sigmatildesq(T, pγ))], color=Cycled(kγ), marker=trimarker, rotation=arrowangle)
                    scatter!(ax, [mutilde(T, pγ)], [sqrt(sigmatildesq(T, pγ))], color=Cycled(kγ), markersize=4)
                end
                lks[end] = lines!(ax, [μ₀ for t in tt], [sqrt(sigmasq(T - t, p)) for t in tt], linestyle = (:dash, :dense), color=:black, linewidth=3)
                scatter!(ax, μ₀, sqrt(sigmasq(0, p)), color=:black)
                scatter!(ax, μ₀, sqrt(sigmasq(T, p)), color=:black, marker=trimarker)

                muaux(γ) = mutilde(T, (; p..., γ = γ))
                sigmasqaux(γ) = sigmatildesq(T, (; p..., γ = γ))

                lines!(ax, [muaux(γ) for γ in γrange], [sqrt(sigmasqaux(γ)) for γ in γrange], color=γrange, colorrange=(0,1), colormap=:redsblues, linestyle=(:dot,:dense), linewidth=2)
            end

            Legend(titlelayout[3, 1], lks, vcat([L"$\gamma = $ %$γ" for γ in γs], L"forward process$$"), orientation = :horizontal, halign = :left, framevisible = false, labelsize = 20)

            display(f)
            if savepdfs
                save(joinpath(@__DIR__(), "figures", "analytic_musigma_evolution_$(product_name)_$(diffusion_schedule).pdf"), f)
            end
            if savepngs
                save(joinpath(@__DIR__(), "figures", "analytic_musigma_evolution_$(product_name)_$(diffusion_schedule).png"), f)
            end
        end
    end
end
nothing

# # Figures - Evolution in parameter space

# for schd in schds
#     for (product_name, product_range) in product_names_and_ranges
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (1600, 80 * ( 2 + length(product_range))))

#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"Final sampling in parameter space $(\mu,\,\sigma)$ with $\mu_0 = $ %$μ₀, $\sigma_0 = $ %$σ₀, $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"with the contour levels of the sample quality $H(\tilde{p}_T | \bar{p}_0)$ (blues) and sample coverage $H(\bar{p}_0 | \tilde{p}_T)$ (reds)", halign = :left, fontsize = 20)

#         lks = Any[1]
#         γrange = LinRange(0.0, 100.0, 100)
        
#         for (k, (αθ, μθ, βT, μT)) in enumerate(product_range)

#             i = (k - 1) ÷ 4 + 1   # runs 1:4
#             j = (k - 1) % 4 + 1   # runs 1:4

#             μmin = min(first(μTs), first(μθs))
#             μmax = max(last(μTs), last(μθs))
#             μspan = max(last(μTs), last(μθs)) - μmin
#             μs = LinRange(μmin - 0.5μspan, μmax + 0.5μspan, 200)
#             σs = LinRange(0.1σ₀, 2σ₀, 200)
#             ax_aspect = ( last(σs) - first(σs) ) / ( last(μs) - first(μs))
#             ax = Axis(
#                 f[i, j],
#                 title = L"$\mu_T = $ %$μT, $\beta_T = $ %$βT, $\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ",
#                 limits = (extrema(μs)...,  extrema(σs)...),
#                 xlabel = L"\mu",
#                 ylabel = L"\sigma",
#                 xlabelsize = 16,
#                 ylabelsize = 16,
#                 titlesize = 19,
#                 #yscale = log10
#             )
#             p = (
#                 ; 
#                 prms...,
#                 αθ = αθ,
#                 βT = βT,
#                 μθ = μθ,
#                 μT = μT,
#                 g = g,
#                 σsq = sigmasq,
#                 schd = schd
#             )
#             kldivs = [kldivergence_normals((μ=x, σ²=y^2),(μ=μ₀, σ²=σ₀^2)) for x in μs, y in σs]
#             kldivrevs = [kldivergence_normals((μ=μ₀, σ²=σ₀^2),(μ=x, σ²=y^2)) for x in μs, y in σs]
#             #heatmap!(f[i, j], μs, σs, kldivs, alpha=0.7)
#             contour!(f[i, j], μs, σs, kldivs, labels=true,levels = exp.(-5.0:1.0:1.0), colormap=Reverse(:blues), linewidth=0.5, alpha=0.5)
#             contour!(f[i, j], μs, σs, kldivrevs, labels=true,levels = exp.(-5.0:1.0:1.0), colormap=Reverse(:reds), linewidth=0.5, alpha=0.5)
                    
#             muaux(γ) = mutilde(T, (; p..., γ = γ))
#             sigmasqaux(γ) = sigmatildesq(T, (; p..., γ = γ))

#             lks[1] = lines!(f[i, j], [muaux(γ) for γ in γrange], [sqrt(sigmasqaux(γ)) for γ in γrange], color=γrange, colorrange=(0,1), colormap=:lapaz, linewidth=2)

#             #lks[1] = lines!(f[i, j], [map(muaux, γrange)], [map(sqrt∘sigmasqaux,γrange)], color=Cycled(1), linewidth=3)
#         end

#         Legend(titlelayout[3, 1], [LineElement(color = cgrad(:greens)[0.5], linewidth = 2)], [L"Final parameters (from heavy color for $\gamma = 0$ to light color for $\gamma=100$)"], orientation = :horizontal, halign = :left, framevisible = false, labelsize = 18)
#         #Legend(titlelayout[3, 1], lks, vcat([L"reverse $\gamma = $ %$γ" for γ in γs], L"forward$$"), orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_musigma_finalKLongamma_$(product_name)_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_musigma_finalKLongamma_$(product_name)_$(diffusion_schedule).png"), f)
#         end
#     end
# end
# nothing

# Figures - Dependence of the final KL divergences on gamma
begin
    for schd in schds
        for (product_name, product_range) in product_names_and_ranges

            diffusion_schedule = uppercase(String(schd))

            f = Figure(size = (1600, 80 * ( 2 + length(product_range))))

            titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
            Label(titlelayout[1, 1], L"Dependence of the final KL divergences on $\gamma$, with $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 28)

            γrange = 0.0:0.1:5.0

            local lks = Any[1, 2, 3, 4]
            
            for (k, (αθ, μθ, βT, μT)) in enumerate(product_range)

                i = (k - 1) ÷ 4 + 1   # runs 1:4
                j = (k - 1) % 4 + 1   # runs 1:4

                ax = Axis(
                    f[i, j], 
                    title = L"$\mu_T = $ %$μT, $\beta_T = $ %$βT, $\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ",
                    limits = (first(γrange), last(γrange), 0, 0.2),
                    xlabel = L"\gamma",
                    ylabel = L"KL",
                    xlabelsize = 16,
                    ylabelsize = 16,
                    titlesize = 19,
                    #yscale = log10
                )

                ax_inset = Axis(
                    f[i, j],
                    width=Relative(0.4),
                    height=Relative(0.2),
                    halign=0.9,
                    valign=0.8,
                    title=L"bang-bang $\gamma(t)$",
                    xlabel = L"t",
                    ylabel = L"\gamma",
                    xlabelsize = 16,
                    ylabelsize = 16,
                    titlealign = :center,
                    titlesize = 16,
                    xticks = 0:prms.T/3:prms.T,
                    yticks = range(extrema(γs)..., length=2),
                    limits=(0.0, prms.T, first(γrange)-1.6, last(γrange)+1.6),
                    xticklabelsize = 12,
                    yticklabelsize = 12,
                )

                p = (
                    ; 
                    prms...,
                    αθ = αθ,
                    βT = βT,
                    μθ = μθ,
                    μT = μT,
                    g = g,
                    σsq = sigmasq,
                    schd = schd
                )

                kldivfun(γ) = kldivergence_normals(T, T, (; p..., γ = γ))
                kldivrevfun(γ) = kldivergencerev_normals(T, T, (; p..., γ = γ))
                lks[1] = lines!(ax, γrange,  kldivfun)
                lks[2] = lines!(ax, γrange,  kldivrevfun)
                
                result_bb = optimal_bangbang_kl(last(γrange), p)
                bb_fwd = result_bb.fwd
                bb_rev = result_bb.rev 
                bb_τs_fwd = result_bb.τs_fwd
                bb_order_fwd = result_bb.order_fwd
                bb_τs_rev= result_bb.τs_rev
                bb_order_rev = result_bb.order_rev

                bb_fwd_fun(τ) =  τ <   bb_τs_fwd ? bb_order_fwd[1] : bb_order_fwd[2]
                bb_rev_fun(τ) =  τ <   bb_τs_rev ? bb_order_rev[1] : bb_order_rev[2]

                lks[3] = hlines!(ax, [bb_fwd], color=Cycled(1), linestyle=(:dash,:loose))
                lks[4] = hlines!(ax, [bb_rev], color=Cycled(2), linestyle=(:dash,:loose))

                lines!(ax_inset, tt, t -> bb_fwd_fun(p.T - t), color=Cycled(1))
                lines!(ax_inset, tt, t -> bb_rev_fun(p.T - t), color=Cycled(2))
                #hlines!(ax_inset, 0, color = :black, linewidth = 0.2, style=:dot)
                #vlines!(ax_inset, 0, color = :black, linewidth = 0.2, style=:dot)

            end

            Legend(titlelayout[2, 1], lks, [L"$H(\tilde{p}_T | \bar{p}_0)$ with constant $\gamma$"; L"$H(\bar{p}_0 | \tilde{p}_T)$ with constant $\gamma$"; L"$H(\tilde{p}_T | \bar{p}_0)$ with optimal bang-bang $\gamma(t)$"; L"$H(\bar{p}_0 | \tilde{p}_T)$ with optimal bang-bang $\gamma(t)$"], orientation = :horizontal, halign = :left, framevisible = false, labelsize = 20)

            display(f)
            if savepdfs
                save(joinpath(@__DIR__(), "figures", "analytic_KL_in_gamma_$(product_name)_$(diffusion_schedule).pdf"), f)
            end
            if savepngs
                save(joinpath(@__DIR__(), "figures", "analytic_KL_in_gamma_$(product_name)_$(diffusion_schedule).png"), f)
            end
        end
    end
end
nothing

# Figures - entropy evolution: global bang-bang vs instantaneous bang-bang
begin
    γ_max = 5.0
    for schd in schds
        for (product_name, product_range) in product_names_and_ranges

            diffusion_schedule = uppercase(String(schd))

            f = Figure(size = (1600, 80 * (2 + length(product_range))))

            titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
            Label(titlelayout[1, 1], L"Entropy evolution with $\gamma_{\max} = $ %$γ_max (%$diffusion_schedule)", halign = :left, fontsize = 28)
            Label(titlelayout[2, 1], L"Instantaneous bang-bang optimal $\gamma(t)$ vs global bang-bang optimal $\gamma(t)$", halign = :left, fontsize = 24)

            local lks = Any[1, 2, 3, 4]

            for (k, (αθ, μθ, βT, μT)) in enumerate(product_range)

                i = (k - 1) ÷ 4 + 1
                j = (k - 1) % 4 + 1

                ax = Axis(
                    f[i, j],
                    title = L"$\mu_T = $ %$μT, $\beta_T = $ %$βT, $\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ",
                    limits = (0.0, T, 0.0, 0.25),
                    xlabel = L"t",
                    ylabel = L"KL",
                    xlabelsize = 16,
                    ylabelsize = 16,
                    titlesize = 19,
                )

                ax_inset = Axis(
                    f[i, j],
                    width = Relative(0.42),
                    height = Relative(0.22),
                    halign = 0.88,
                    valign = 0.82,
                    title = L"$\gamma(t)$ schedules",
                    xlabel = L"t",
                    ylabel = L"\gamma",
                    xlabelsize = 14,
                    ylabelsize = 14,
                    titlealign = :center,
                    titlesize = 16,
                    xticks = 0:prms.T / 3:prms.T,
                    yticks = [0.0, γ_max],   # must be a vector, not (0.0, γ_max): Makie reads tuples as (positions, labels)
                    limits = (0.0, prms.T, -0.2γ_max, 1.2γ_max),
                    xticklabelsize = 11,
                    yticklabelsize = 11,
                )

                p = (
                    ;
                    prms...,
                    αθ = αθ,
                    βT = βT,
                    μθ = μθ,
                    μT = μT,
                    g = g,
                    σsq = sigmasq,
                    schd = schd
                )

                result_bb = optimal_bangbang_kl(γ_max, p)
                γfun_global_fwd = bangbang_gammafun(result_bb, :fwd)
                γfun_global_rev = bangbang_gammafun(result_bb, :rev)
                sched_inst_fwd = instantaneous_gamma_schedule(γ_max, p; direction = :fwd)
                sched_inst_rev = instantaneous_gamma_schedule(γ_max, p; direction = :rev)
                γfun_inst_fwd = sched_inst_fwd.γfun
                γfun_inst_rev = sched_inst_rev.γfun

                n_evo = length(tt)
                evo_global_fwd = kldivergence_evolution(γfun_global_fwd, p; npts = n_evo, direction = :fwd)
                evo_inst_fwd   = kldivergence_evolution(γfun_inst_fwd, p; npts = n_evo, direction = :fwd)
                evo_global_rev = kldivergence_evolution(γfun_global_rev, p; npts = n_evo, direction = :rev)
                evo_inst_rev   = kldivergence_evolution(γfun_inst_rev, p; npts = n_evo, direction = :rev)
                # τ = 0 at t = T and τ = T at t = 0, so reverse the stored KL sequence for plotting vs tt
                kl_global_fwd = reverse(evo_global_fwd.kl)
                kl_inst_fwd   = reverse(evo_inst_fwd.kl)
                kl_global_rev = reverse(evo_global_rev.kl)
                kl_inst_rev   = reverse(evo_inst_rev.kl)

                lks[1] = lines!(ax, tt, kl_global_fwd, color = Cycled(1))
                lks[2] = lines!(ax, tt, kl_inst_fwd, color = Cycled(1), linestyle = (:dash, :dense))
                lks[3] = lines!(ax, tt, kl_global_rev, color = Cycled(2))
                lks[4] = lines!(ax, tt, kl_inst_rev, color = Cycled(2), linestyle = (:dash, :dense))

                bb_τs_fwd = result_bb.τs_fwd
                bb_order_fwd = result_bb.order_fwd
                bb_fwd_fun(τ) = τ < bb_τs_fwd ? bb_order_fwd[1] : bb_order_fwd[2]
                bb_τs_rev = result_bb.τs_rev
                bb_order_rev = result_bb.order_rev
                bb_rev_fun(τ) = τ < bb_τs_rev ? bb_order_rev[1] : bb_order_rev[2]

                lines!(ax_inset, tt, t -> bb_fwd_fun(p.T - t), color = Cycled(1))
                lines!(ax_inset, tt, t -> bb_rev_fun(p.T - t), color = Cycled(2))
                lines!(ax_inset, tt, t -> γfun_inst_fwd(p.T - t), color = Cycled(1), linestyle = (:dash, :dense))
                lines!(ax_inset, tt, t -> γfun_inst_rev(p.T - t), color = Cycled(2), linestyle = (:dash, :dense))
            end

            Legend(titlelayout[3, 1], lks, [
                L"$H(\tilde{p}_\tau | \bar{p}_\tau)$, global bang-bang $\gamma(t)$";
                L"$H(\tilde{p}_\tau | \bar{p}_\tau)$, instantaneous $\gamma(t)$";
                L"$H(\bar{p}_\tau | \tilde{p}_\tau)$, global bang-bang $\gamma(t)$";
                L"$H(\bar{p}_\tau | \tilde{p}_\tau)$, instantaneous $\gamma(t)$"
            ], orientation = :horizontal, halign = :left, framevisible = false, labelsize = 20)

            display(f)
            if savepdfs
                save(joinpath(@__DIR__(), "figures", "analytic_KL_evolution_bangbang_$(product_name)_$(diffusion_schedule).pdf"), f)
            end
            if savepngs
                save(joinpath(@__DIR__(), "figures", "analytic_KL_evolution_bangbang_$(product_name)_$(diffusion_schedule).png"), f)
            end
        end
    end
end
nothing

# # Figures - Dependence on gamma of the final KL divergence with Albergo gamma
# begin
#     for schd in schds
#         for (product_name, product_range) in product_names_and_ranges

#             diffusion_schedule = uppercase(String(schd))

#             f = Figure(size = (1600, 80 * ( 2 + length(product_range))))

#             titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#             Label(titlelayout[1, 1], L"Dependence of the final KL divergences on $\gamma$, with $\mu_0 = $ %$μ₀, $\sigma_0 = $ %$σ₀, $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 24)
#             Label(titlelayout[2, 1], L"for sample quality $H(\tilde{p}_T | \bar{p}_0)$ (blue) and for sample coverage $H(\bar{p}_0 | \tilde{p}_T)$ (orange)", halign = :left, fontsize = 20)

#             γrange = 0.0:0.1:5.0

#             local lks = Any[1, 2, 3, 4, 5, 6]
#             local legend_alternate = ""
            
#             for (k, (αθ, μθ, βT, μT)) in enumerate(product_range)

#                 i = (k - 1) ÷ 4 + 1   # runs 1:4
#                 j = (k - 1) % 4 + 1   # runs 1:4

#                     ax = Axis(
#                         f[i, j], 
#                         title = L"$\mu_T = $ %$μT, $\beta_T = $ %$βT, $\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ",
#                         limits = (first(γrange), last(γrange), 0, 0.2),
#                         xlabel = L"\gamma",
#                         ylabel = L"KL",
#                         xlabelsize = 16,
#                         ylabelsize = 16,
#                         titlesize = 19,
#                         #yscale = log10
#                     )
#                     p = (
#                         ; 
#                         prms...,
#                         αθ = αθ,
#                         βT = βT,
#                         μθ = μθ,
#                         μT = μT,
#                         g = g,
#                         σsq = sigmasq,
#                         schd = schd
#                     )

#                     kldivfun(γ) = kldivergence_normals(T, T, (; p..., γ = γ))
#                     kldivrevfun(γ) = kldivergencerev_normals(T, T, (; p..., γ = γ))
#                     lks[1] = lines!(f[i, j], γrange,  kldivfun)
#                     lks[2] = lines!(f[i, j], γrange,  kldivrevfun)
                    
#                     result_bb = optimal_bangbang_kl(4 * last(γrange), p)
#                     bb_fwd = result_bb.fwd
#                     bb_rev = result_bb.rev 
#                     lks[3] = hlines!(f[i, j], [bb_fwd], color=Cycled(1), linestyle=(:dash,:loose))
#                     lks[4] = hlines!(f[i, j], [bb_rev], color=Cycled(2), linestyle=(:dash,:loose))

#                     #result_alternate = kldivergences_timevarying(τ -> sin(π * τ / p.T)^2, p)
#                     #legend_alternate = L"$\gamma(t) = \sin(\pi t / T)^2$"
#                     #result_alternate = kldivergences_timevarying(τ -> sqrt(sigmabarsq(τ, p)), p)
#                     #legend_alternate = L"$\gamma(t) = \sigma(t)$"
#                     result_alternate = kldivergences_timevarying(τ -> sigmabarsq_prime(τ, p) / g(p.T - τ, p)^2, p)
#                     legend_alternate = L"$\gamma(t) = 2\sigma(t)\sigma'(t)$"
                    
#                     sn_fwd = result_alternate.fwd
#                     sn_rev = result_alternate.rev 
                    
#                     lks[5] = hlines!(f[i, j], [sn_fwd], color=Cycled(1), linestyle=:dot)
#                     lks[6] = hlines!(f[i, j], [sn_rev], color=Cycled(2), linestyle=:dot)
#                 end
#             #end

#             Legend(titlelayout[3, 1], lks, [L"$H(\tilde{p}_T | \bar{p}_0)$ with constant $\gamma$"; L"$H(\bar{p}_0 | \tilde{p}_T)$ with constant $\gamma$"; L"$H(\tilde{p}_T | \bar{p}_0)$ with optimal bang-bang $\gamma(t)$"; L"$H(\bar{p}_0 | \tilde{p}_T)$ with optimal bang-bang $\gamma(t)$"; L"$H(\tilde{p}_T | \bar{p}_0)$ with %$legend_alternate"; L"$H(\bar{p}_0 | \tilde{p}_T)$ with %$legend_alternate"], orientation = :horizontal, halign = :left, framevisible = false)

#             display(f)
#             if savepdfs
#                 save(joinpath(@__DIR__(), "figures", "analytic_KL_in_gamma_albergo_$(product_name)_$(diffusion_schedule).pdf"), f)
#             end
#             if savepngs
#                 save(joinpath(@__DIR__(), "figures", "analytic_KL_in_gamma_albergo_$(product_name)_$(diffusion_schedule).png"), f)
#             end
#         end
#     end
# end
# nothing

# Figures - Dependence on gamma of the final KL divergence multidimensional

multidim_product_names_and_ranges = (
    #("full", (Iterators.product(αθs, μθs, βTs, μTs))),
    #("extrema", (Iterators.product(extrema(αθs), last(μθs), extrema(βTs), last(μTs), extrema(αθs), last(μθs), extrema(βTs), last(μTs), 4, (6, 40)))),
    #("zero_mean", (Iterators.product(extrema(αθs), 0.0, extrema(βTs), 0.0, extrema(αθs), 0.0, extrema(map(b -> 10b, βTs)), 0.0, 4, (6, 40)))),
    #("off_mean", (Iterators.product(extrema(αθs), 0.1, extrema(βTs), -0.1, extrema(αθs), -0.05, extrema(map(b -> 10b, βTs)), 0.02, 4, (6, 40)))),
    #("exact_score", (Iterators.product(1.0, 0.0, βTs, μTs))),
    # ("minimal", (
    #     (1.0, 0.1, 1.0, 0.0, 1.0, 0.02, 1.0, 0.0, 4, 6),
    #     (1.4, 0.5, 0.1, 0.2, 0.2, 0.05, 0.1, 0.02, 4, 6),
    #     (1.4, 0.5, 0.1, 0.2, 1.1, 0.05, 0.1, 0.02, 4, 6),
    #     (1.3, 0.1, 0.6, 0.0, 0.1, 0.02, 0.6, 0.0, 4, 6),
    #     (1.0, 0.1, 1.0, 0.0, 1.0, 0.02, 1.0, 0.0, 4, 20),
    #     (1.4, 0.5, 0.1, 0.2, 0.1, 0.05, 0.1, 0.02, 4, 20),
    #     (0.2, 0.1, 2.4, 0.0, 0.6, 0.02, 0.8, 0.0, 4, 20),
    #     (1.4, 0.1, 0.6, 0.0, 1.3, 0.02, 0.6, 0.0, 4, 20),
    # )
    # ),
    # ("minimal", (
    #     (1.4, -0.2, 0.1, 1.0, 1.1, -0.05, 0.1, 1.0, 4, 6),
    #     (1.4, -0.2, 0.1, 1.0, 1.1, -0.05, 0.1, 1.0, 4, 20),
    #     (1.4, -0.2, 0.1, 1.0, 1.1, 0.05, 0.1, 1.0, 4, 6),
    #     (1.4, -0.2, 0.1, 1.0, 1.1, 0.05, 0.1, 1.0, 4, 20),
    # )
    ("minimal", (
        (1.4, -0.2, 0.1, 1.0, 1.1, -0.05, 1.0, 4, 6),
        (1.4, -0.2, 0.1, 1.0, 1.1, -0.05, 1.0, 4, 20),
        (1.4, -0.2, 0.1, 1.0, 1.1, 0.05, 1.0, 4, 6),
        (1.4, -0.2, 0.1, 1.0, 1.1, 0.05, 1.0, 4, 20),
    )
    ),
)
# (αθ1, μθ1, βT1, μT1, αθ2, μθ2, μT2, d, n)

begin
    for schd in schds
        for (product_name, product_range) in multidim_product_names_and_ranges

            diffusion_schedule = uppercase(String(schd))

            f = Figure(size = (1600, 80 * ( 2 + length(product_range))))

            titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
            Label(titlelayout[1, 1], L"Dependence of the final KL divergences on $\gamma$, $T = $ %$T (%$diffusion_schedule)", halign = :left, fontsize = 26)

            γrange = 0.0:0.02:5.0
            μ₀2 = 0.0
            σ₀2 = 0.2

            local lks = Any[1, 2, 3, 4]
            
            for (k, (αθ1, μθ1, βT1, μT1, αθ2, μθ2, μT2, d, n)) in enumerate(product_range)

                i = (k - 1) ÷ 4 + 1   # runs 1:4
                j = (k - 1) % 4 + 1   # runs 1:4

                ax = Axis(
                    f[i, j], 
                    title = L"$\mu_\theta:$ %$μθ1, %$μθ2, $\alpha_\theta:$ %$αθ1, %$αθ2, d = %$d, n = %$n",
                    limits = (first(γrange), last(γrange), -0.05, 2.0),
                    xlabel = L"\gamma",
                    ylabel = L"KL",
                    xlabelsize = 16,
                    ylabelsize = 16,
                    titlesize = 16,
                    #yscale = log10
                )

                p1 = (
                    ; 
                    prms...,
                    αθ = αθ1,
                    βT = βT1,
                    μθ = μθ1,
                    μT = μT1,
                    g = g,
                    σsq = sigmasq,
                    schd = schd
                )
                # β''_T = β'_T σ'(T) / σ''(T); at τ = 0, σ̄(0)² = sigmabarsq(0, p) = σ(T)²
                βT2 = βT1 * sqrt(sigmabarsq(0.0, p1) / sigmabarsq(0.0, (; prms..., σ₀ = σ₀2, schd = schd)))
                p2 = (
                    ; 
                    prms...,
                    μ₀ = μ₀2,
                    σ₀ = σ₀2,
                    αθ = αθ2,
                    βT = βT2,
                    μθ = μθ2,
                    μT = μT2,
                    g = g,
                    σsq = sigmasq,
                    schd = schd
                )

                kldivfun_p1(γ)    = d       * kldivergence_normals(T, T, (; p1..., γ = γ))
                kldivfun_p2(γ)    = (n - d) * kldivergence_normals(T, T, (; p2..., γ = γ))
                kldivfun(γ)       = kldivfun_p1(γ) + kldivfun_p2(γ)
                kldivrevfun_p1(γ) = d       * kldivergencerev_normals(T, T, (; p1..., γ = γ))
                kldivrevfun_p2(γ) = (n - d) * kldivergencerev_normals(T, T, (; p2..., γ = γ))
                kldivrevfun(γ)    = kldivrevfun_p1(γ) + kldivrevfun_p2(γ)

                lks[1] = lines!(ax, γrange, kldivfun,       color = :blue)
                lks[3] = lines!(ax, γrange, kldivfun_p1,    color = :blue,   linestyle = :dash)
                lks[4] = lines!(ax, γrange, kldivfun_p2,    color = :blue,   linestyle = :dot)
                lks[2] = lines!(ax, γrange, kldivrevfun,    color = :orange)
                lines!(ax, γrange, kldivrevfun_p1, color = :orange, linestyle = :dash)
                lines!(ax, γrange, kldivrevfun_p2, color = :orange, linestyle = :dot)

            end

            Legend(titlelayout[2, 1], lks, [
                L"$H(\tilde{p}_T | \bar{p}_0)$ with constant $\gamma$";
                L"$H(\bar{p}_0 | \tilde{p}_T)$ with constant $\gamma$";
                L"manifold contribution $d \cdot h'$";
                L"off-manifold contribution $(n - d) \cdot h''$"
            ], orientation = :horizontal, halign = :left, framevisible = false, labelsize = 22)

            display(f)
            if savepdfs
                save(joinpath(@__DIR__(), "figures", "analytic_KL_in_gamma_multidim_$(product_name)_$(diffusion_schedule).pdf"), f)
            end
            if savepngs
                save(joinpath(@__DIR__(), "figures", "analytic_KL_in_gamma_multidim_$(product_name)_$(diffusion_schedule).png"), f)
            end
        end
    end
end
nothing

# # Final KL divergence with assorted parameters
# begin
#     let schd = :edm
#         diffusion_schedule = uppercase(String(schd))
#         dd_abs = Dict(γ => Float64[] for γ in γs)
#         dd_rel = Dict(γ => Float64[] for γ in γs)
#         #for (μθ, αθ, βT, μT) in Iterators.product(μθs, αθs, βTs, μTs)
#         for (μθ, αθ, βT) in Iterators.product(μθs, αθs, βTs)
#             for μT in μTs
#                 for γ in γs
#                     p = (
#                         ; 
#                         prms...,
#                         αθ = αθ,
#                         βT = βT,
#                         μθ = μθ,
#                         μT = μT,
#                         γ = γ,
#                         schd = schd
#                     )
#                     p0 = (
#                         ; 
#                         p...,
#                         γ = 0.0,
#                     )
#                     kldiv = kldivergence_normals(T, T, p)
#                     kldiv0 = kldivergence_normals(T, T, p0)
#                     push!(dd_abs[γ], kldiv)
#                     if kldiv0 ≈ 0.0
#                         @info "Ops: μθ=$μθ, αθ=$αθ, βT=$βT, μT=$μT"
#                     end
#                     push!(dd_rel[γ], γ == 0.0 ? 1.0 : kldiv / kldiv0)
#                 end
#             end
#             for γ in γs
#                 push!(dd_abs[γ], NaN)
#                 push!(dd_rel[γ], γ == 0.0 ? 1.0 : NaN)
#             end
#         end
#         local f = Figure(size = (1000, 400))
#         local titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"Final KL divergence $H(\tilde{p}_0 | \bar{p}_0)$ (%$diffusion_schedule)", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"for assorted values $p_n$ of the parameters $\mu_\theta,$ $\alpha_\theta,$ $\beta_T,$ and $\mu_T$, from outer to inner loops:", halign = :left, fontsize = 20)
#         Label(titlelayout[3, 1], L"$\mu_\theta \in $%$μθs, $\alpha_\theta \in$%$(αθs), $\beta_T \in$%$βTs, and $\mu_T \in$%$μTs,", halign = :left, fontsize = 20)

#         local lks = Any[γs...]
#         ax_abs = Axis(
#             f[1, 1],
#             title=L"KL divergences $$ (%$diffusion_schedule)",
#             xlabel = L"n",
#             ylabel = L"\mathrm{entropy}"
#         )
#         ax_rel = Axis(
#             f[1, 2],
#             title=L"KL divergences relative to that of $\gamma = 0$ (%$diffusion_schedule)",
#             xlabel = L"p",
#             ylabel = L"\mathrm{adimensional}"
#         )
#         for (k, γ) in enumerate(γs)
#             lks[k] = barplot!(ax_abs, dd_abs[γ])
#             if γ == 0.0
#                 lines!(ax_rel, dd_rel[γ], color=Cycled(k))
#             else
#                 barplot!(ax_rel, dd_rel[γ], color=Cycled(k))
#             end
#         end
#         Legend(titlelayout[4, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_assorted_final_KLdivergence_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_assorted_final_KLdivergence_$(diffusion_schedule).png"), f)
#         end
#     end
# end

# # Final KL divergence with assorted parameters
# begin
#     let schd = :edm
#         diffusion_schedule = uppercase(String(schd))

#         local f = Figure(size = (1200, 1000))
#         local titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"Final KL divergence $H(\tilde{p}_0 | \bar{p}_0)$ (%$diffusion_schedule)", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"for assorted values of the parameters $\mu_\theta,$ $\alpha_\theta,$ $\beta_T,$ and $\mu_T$", halign = :left, fontsize = 20)

#         local lks = Any[γs...]

#         for (j, αθ) in enumerate(αθs)
#             for (i, βT) in enumerate(βTs)
#                 dd_abs = Dict(γ => Float64[] for γ in γs)
#                 dd_rel = Dict(γ => Float64[] for γ in γs)
#                 for μθ in μθs
#                     for μT in μTs
#                         for γ in γs
#                             p = (
#                                 ; 
#                                 prms...,
#                                 αθ = αθ,
#                                 βT = βT,
#                                 μθ = μθ,
#                                 μT = μT,
#                                 γ = γ,
#                                 schd = schd
#                             )
#                             p0 = (
#                                 ; 
#                                 p...,
#                                 γ = 0.0,
#                             )
#                             kldiv = kldivergence_normals(T, T, p)
#                             kldiv0 = kldivergence_normals(T, T, p0)
#                             push!(dd_abs[γ], kldiv)
#                             if kldiv0 ≈ 0.0
#                                 println("Ops: μθ=$μθ, αθ=$αθ, βT=$βT, μT=$μT")
#                             end
#                             push!(dd_rel[γ], γ == 0.0 ? 1.0 : kldiv / kldiv0)
#                         end
#                     end
#                     for γ in γs
#                         push!(dd_abs[γ], NaN)
#                         push!(dd_rel[γ], γ == 0.0 ? 1.0 : NaN)
#                     end
#                 end

#                 ax_abs = Axis(
#                     f[i, j],
#                     title=L"KL divergences $\alpha_\theta = $ %$αθ, $\beta_T = $ %$βT",
#                     limits = (0, 12, 0.0, 0.35),
#                     xticks = (2:4:10, [L"$\mu_\theta = $ %$μθ" for μθ in μθs]),
#                     ylabel = L"\mathrm{entropy}"
#                 )
#                 for (k, γ) in enumerate(γs)
#                     lks[k] = barplot!(ax_abs, collect(1:9) .+ 0.22( k-length(γs)/2 ), dd_abs[γ], bar_labels = γ == 0.0 ? repeat([[L"$\mu_T = $ %$μT" for μT in μTs]; ""], 3) : nothing, label_rotation = π/2, label_offset = 2 .+ 200 * [maximum([dd_abs[γ][i] for γ in γs]) - dd_abs[0.0][i] for i in eachindex(dd_abs[γ])], label_size=12, width=0.25)
#                 end
#             end
#         end

#         Legend(titlelayout[3, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)
#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_assorted_final_KLdivergence_alt_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_assorted_final_KLdivergence_alt_$(diffusion_schedule).png"), f)
#         end
#     end
# end

# # Check KL divergence and KL evolution terms
# begin
#     @info "dt: $(Float64(tt.step))"
#     for schd in schds
#         diffusion_schedule = uppercase(String(schd))
#         f = Figure(size = (1000, 1200))

#         titlelayout = GridLayout(f[0, 1], halign = :left, tellwidth = false)
#         Label(titlelayout[1, 1], L"Check with KL divergence $H(\tilde{p}_t | \bar{p}_t),$ with $\mu_0 = $ %$μ₀, $\sigma_0 = $ %$σ₀ (%$diffusion_schedule)", halign = :left, fontsize = 24)
#         Label(titlelayout[2, 1], L"from formula (solid), $$ and from integrated evolution equation (dashed)", halign = :left, fontsize = 24)

#         lks = Any[γs...]

#         for (i, αθ) in enumerate(αθs)
#             for (j, βT) in enumerate(βTs)
#                 μθ = rand(μθs)
#                 μT = rand(μTs)
#                 ax = Axis(
#                     f[i, j], 
#                     title = L"$\mu_\theta = $ %$μθ, $\alpha_\theta = $ %$αθ, $\mu_T = $ %$μT, $\beta_T = $ %$βT (%$diffusion_schedule)",
#                     limits = (0.0, T, -0.005, 0.2),
#                     xlabel = L"t",
#                     ylabel = L"\mathrm{entropy}",
#                 )
#                 p = (
#                     ; 
#                     prms...,
#                     αθ = αθ,
#                     βT = βT,
#                     μθ = μθ,
#                     μT = μT,
#                     g = g,
#                     σsq = sigmasq
#                 )
#                 for (k, γ) in enumerate(γs)
#                     p =(; p..., γ = γ)
#                     klformula = [kldivergence_normals(τ, τ, p) for τ in tt]
#                     klintegrated = accumulate(+, [klrate(τ, p) * Float64(tt.step) for τ in tt], init = kldivergence_normals(0.0, 0.0, p))
#                     @info "Maximum error: $(maximum(abs, klformula - klintegrated)) (error fraction: $(maximum(abs, klformula - klintegrated) / √(1+γ) / Float64(tt.step)))"
#                     @assert maximum(abs, klformula - klintegrated) ≤ ( 2√(1+γ) ) * Float64(tt.step)
#                     lks[k] = lines!(f[i, j], Iterators.reverse(tt), klformula, label=L"KL formula $\gamma = $ %$γ", color=Cycled(k))
#                     lines!(f[i, j], Iterators.reverse(tt), klintegrated, label=L"KL integrated $\gamma = $ %$γ", linestyle = (:dash, :dense), color=Cycled(k))
#                 end
#             end
#         end

#         Legend(titlelayout[3, 1], lks, [L"$\gamma = $ %$γ" for γ in γs], orientation = :horizontal, halign = :left, framevisible = false)

#         display(f)
#         if savepdfs
#             save(joinpath(@__DIR__(), "figures", "analytic_KLdivergence_check_$(diffusion_schedule).pdf"), f)
#         end
#         if savepngs
#             save(joinpath(@__DIR__(), "figures", "analytic_KLdivergence_check_$(diffusion_schedule).png"), f)
#         end
#     end
# end
# nothing

# Tests and benchmarks

if runtests

    prms_tst = (
        μ₀ = 2.0,
        σ₀ = 1.0,
        σ = 1.0,
        T = 80.0,
        μθ = 2.1,
        αθ = 1.1,
        μT = 0.8,
        βT = 1.2,
        γ = 5.0,
        schd = :edm
    )

    @test kldivergence_normals(prms_tst.T, prms_tst.T, (; prms_tst..., γ = 1.0)) ≈ kldivergences_timevarying(τ -> 1.0, prms_tst).fwd

    res_timevarying = kldivergences_timevarying(τ -> τ < 0.4 ? 0.0 : 5.0, prms_tst)
    res_bangbang = bangbang_terminal(0.4, 0.0, 5.0, prms_tst)
    @test res_timevarying.mf ≈ res_bangbang[1] (rtol = 1e-5)
    @test res_timevarying.vf ≈ res_bangbang[2] (rtol = 1e-5)

    sched_inst = instantaneous_gamma_schedule(5.0, prms_tst)
    res_inst = kldivergences_timevarying(sched_inst.γfun, prms_tst)
    evo_inst = kldivergence_evolution(sched_inst.γfun, prms_tst)
    @test res_inst.fwd ≈ evo_inst.kl[end] (rtol = 1e-6)
    @test res_inst.fwd ≈ kldivergence_normals((μ = res_inst.mf, σ² = res_inst.vf), (μ = prms_tst.μ₀, σ² = sigmabarsq(prms_tst.T, prms_tst)))

    p_const = (; prms_tst..., γ = 1.0)
    @test klrate_e_mv(0.5, mutilde(0.5, p_const), sigmatildesq(0.5, p_const), prms_tst) ≈ klrate_e(0.5, p_const) (rtol = 1e-10)
    @test klrate_d_mv(0.5, mutilde(0.5, p_const), sigmatildesq(0.5, p_const), prms_tst) ≈ klrate_d(0.5, p_const) (rtol = 1e-10)

    # Check the closed-form rates against direct quadrature of the defining
    # expectations, for both measures (p̃ for fwd, p̄ for rev).
    let τ = 0.5, μ̃ = mutilde(0.5, p_const), σ̃² = sigmatildesq(0.5, p_const)
        s̄ = sigmabarsq(τ, prms_tst)
        ε(x) = -(x - prms_tst.μθ) / (prms_tst.αθ * s̄) + (x - prms_tst.μ₀) / s̄
        ∇logh(x) = -(x - μ̃) / σ̃² + (x - prms_tst.μ₀) / s̄
        quadrature(f, dist) = begin
            lo, hi = mean(dist) - 12std(dist), mean(dist) + 12std(dist)
            xs = range(lo, hi, length = 200_001)
            sum(x -> f(x) * pdf(dist, x), xs) * step(xs)
        end
        p̃ = Normal(μ̃, √σ̃²)
        p̄ = Normal(prms_tst.μ₀, √s̄)
        @test klrate_e_mv(τ, μ̃, σ̃², prms_tst) ≈ quadrature(x -> ε(x) * ∇logh(x), p̃) (rtol = 1e-6)
        @test klrate_d_mv(τ, μ̃, σ̃², prms_tst) ≈ quadrature(x -> ∇logh(x)^2, p̃) (rtol = 1e-6)
        @test klrate_rev_e_mv(τ, μ̃, σ̃², prms_tst) ≈ quadrature(x -> ε(x) * ∇logh(x), p̄) (rtol = 1e-6)
        @test klrate_rev_d_mv(τ, μ̃, σ̃², prms_tst) ≈ quadrature(x -> ∇logh(x)^2, p̄) (rtol = 1e-6)
    end
end

if runtests 
    @btime g_ve(t, p) setup=(t=0.5; p=(σ₀ = 1.0, σ = 1.0))

    @btime g_ve(t, p) setup=(t=0.5; p=$prms_tst)
    @btime g_edm(t, p) setup=(t=0.5; p=$prms_tst)
    @btime g(t, p) setup=(t=0.5; p=$prms_tst)

    @btime sigmasq_ve(t, p) setup=(t=0.5; p=$prms_tst)
    @btime sigmasq_edm(t, p) setup=(t=0.5; p=$prms_tst)
    @btime sigmasq(t, p) setup=(t=0.5; p=$prms_tst)
    @btime sigmabarsq(τ, p) setup=(τ=0.5; p=$prms_tst)

    @btime score_error_mse(t, p) setup=(t=0.5; p=$prms_tst)

    @btime mutilde(τ, p) setup=(τ=0.5; p=$prms_tst)

    @btime sigmatildesq(τ, p) setup=(τ=0.5; p=$prms_tst)

    @btime kldivergence_normals(τ̃, τ, p) setup=(τ̃=0.1; τ=0.5; p=$prms_tst)

    @btime klrate(τ, p) setup=(τ=0.1; p=$prms_tst)

    @btime klbound(τ, p) setup=(τ=0.1; p=$prms_tst)

    @btime klbound(τ, p) setup=(τ=0.1; p=$prms_tst)

    @btime mutilde_segment(τ, τ₀, m₀, γ, p) setup=(τ=0.5; τ₀=0.3; m₀=1.5; γ=5.0; p=$prms_tst)

    @btime sigmatildesq_segment(τ, τ₀, m₀, γ, p) setup=(τ=0.5; τ₀=0.3; m₀=1.5; γ=5.0; p=$prms_tst)

    @btime bangbang_terminal(τs, γ₁, γ₂, p) setup=(τs=0.5; γ₁=0.0; γ₂=5.0; p=$prms_tst)

    @btime optimal_bangbang_kl(γ_max, p) setup=(γ_max=5.0; p=$prms_tst)

    @btime kldivergences_timevarying(γfun, p) setup=(γfun = τ -> τ < 0.4 ? 0.0 : 5.0; p=$prms_tst)

    @btime montecarlo_fwd!(xfwd, p) setup=(xfwd=zeros(10, 20); p=$prms_tst)

    @btime montecarlo_reverse!(xtilde, p) setup=(xtilde=zeros(10, 20); p=$prms_tst)

    @code_warntype sigmabarsq(0.1, prms_tst)

    @code_warntype kldivergence_normals(0.1, 0.2, prms_tst)

    @code_warntype klrate(0.5, prms_tst)

    @code_warntype klbound(0.5, prms_tst)

    @code_warntype exp_alpha(0.2, 0.5, prms_tst)
end

nothing
