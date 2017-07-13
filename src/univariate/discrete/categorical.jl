doc"""
    Categorical(p)
A *Categorical distribution* is parameterized by a probability vector `p` (of length `K`).
$P(X = k) = p[k]  \quad \text{for } k = 1, 2, \ldots, K.$
```julia
Categorical(p)   # Categorical distribution with probability vector p
params(d)        # Get the parameters, i.e. (p,)
probs(d)         # Get the probability vector, i.e. p
ncategories(d)   # Get the number of categories, i.e. K
```
Here, `p` must be a real vector, of which all components are nonnegative and sum to one.
**Note:** The input vector `p` is directly used as a field of the constructed distribution, without being copied.
External links:
* [Categorical distribution on Wikipedia](http://en.wikipedia.org/wiki/Categorical_distribution)
"""

Categorical{T} = Generic{Int,T,UnitRange{Int}}

(::Type{Categorical})(p::Vector{P}, ::NoArgCheck) where P =
    Generic{Int,P,UnitRange{Int}}(1:length(p), p, NoArgCheck())

function (::Type{Categorical})(p::Vector{P}) where P
    @check_args(Generic, isprobvec(p))
    Generic{Int,P,UnitRange{Int}}(1:length(p), p, NoArgCheck())
end

function (::Type{Categorical})(k::Integer)
    @check_args(Generic, k >= 1)
    Generic{Int,Float64,UnitRange{Int}}(1:k, fill(1/k, k), NoArgCheck())
end


@distr_support Categorical 1 support(d).stop

### Conversions

convert{T<:Real, S<:Real}(::Type{Categorical{T}}, p::Vector{S}) = Categorical(Vector{T}(p))

### Parameters

ncategories(d::Categorical) = support(d).stop
params(d::Categorical) = (probs(d),)
@inline partype{T<:Real}(d::Categorical{T}) = T

### Statistics

function median{T<:Real}(d::Categorical{T})
    k = ncategories(d)
    p = probs(d)
    cp = zero(T)
    i = 0
    while cp < 1/2 && i <= k
        i += 1
        @inbounds cp += p[i]
    end
    i
end

function mgf{T<:Real}(d::Categorical{T}, t::Real)
    k = ncategories(d)
    p = probs(d)
    s = zero(T)
    for i = 1:k
        @inbounds s += p[i] * exp(t)
    end
    s
end

function cf{T<:Real}(d::Categorical{T}, t::Real)
    k = ncategories(d)
    p = probs(d)
    s = zero(T) + zero(T)*im
    for i = 1:k
        @inbounds s += p[i] * cis(t)
    end
    s
end

### Evaluation

function cdf{T<:Real}(d::Categorical{T}, x::Int)
    k = ncategories(d)
    p = probs(d)
    x < 1 && return zero(T)
    x >= k && return one(T)
    c = p[1]
    for i = 2:x
        @inbounds c += p[i]
    end
    return c
end

pdf{T<:Real}(d::Categorical{T}, x::Int) = insupport(d, x) ? probs(d)[x] : zero(T)

logpdf(d::Categorical, x::Int) = insupport(d, x) ? log(probs(d)[x]) : -Inf

function _pdf!{T<:Real}(r::AbstractArray, d::Categorical{T}, rgn::UnitRange)
    vfirst = round(Int, first(rgn))
    vlast = round(Int, last(rgn))
    vl = max(vfirst, 1)
    vr = min(vlast, ncategories(d))
    p = probs(d)
    if vl > vfirst
        for i = 1:(vl - vfirst)
            r[i] = zero(T)
        end
    end
    fm1 = vfirst - 1
    for v = vl:vr
        r[v - fm1] = p[v]
    end
    if vr < vlast
        for i = (vr - vfirst + 2):length(rgn)
            r[i] = zero(T)
        end
    end
    return r
end


# sampling

sampler(d::Categorical) = AliasTable(d.p)


### sufficient statistics

immutable CategoricalStats <: SufficientStats
    h::Vector{Float64}
end

function add_categorical_counts!{T<:Integer}(h::Vector{Float64}, x::AbstractArray{T})
    for i = 1 : length(x)
        @inbounds xi = x[i]
        h[xi] += 1.   # cannot use @inbounds, as no guarantee that x[i] is in bound
    end
    h
end

function add_categorical_counts!{T<:Integer}(h::Vector{Float64}, x::AbstractArray{T}, w::AbstractArray{Float64})
    n = length(x)
    if n != length(w)
        throw(ArgumentError("Inconsistent array lengths."))
    end
    for i = 1 : n
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        h[xi] += wi   # cannot use @inbounds, as no guarantee that x[i] is in bound
    end
    h
end

function suffstats{T<:Integer}(::Type{Categorical}, k::Int, x::AbstractArray{T})
    CategoricalStats(add_categorical_counts!(zeros(k), x))
end

function suffstats{T<:Integer}(::Type{Categorical}, k::Int, x::AbstractArray{T}, w::AbstractArray{Float64})
    CategoricalStats(add_categorical_counts!(zeros(k), x, w))
end

const CategoricalData = Tuple{Int, AbstractArray}

suffstats(::Type{Categorical}, data::CategoricalData) = suffstats(Categorical, data...)
suffstats(::Type{Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = suffstats(Categorical, data..., w)

# Model fitting

function fit_mle(::Type{Categorical}, ss::CategoricalStats)
    Categorical(pnormalize!(ss.h))
end

function fit_mle{T<:Integer}(::Type{Categorical}, k::Integer, x::AbstractArray{T})
    Categorical(pnormalize!(add_categorical_counts!(zeros(k), x)), NoArgCheck())
end

function fit_mle{T<:Integer}(::Type{Categorical}, k::Integer, x::AbstractArray{T}, w::AbstractArray{Float64})
    Categorical(pnormalize!(add_categorical_counts!(zeros(k), x, w)), NoArgCheck())
end

fit_mle(::Type{Categorical}, data::CategoricalData) = fit_mle(Categorical, data...)
fit_mle(::Type{Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = fit_mle(Categorical, data..., w)

fit_mle{T<:Integer}(::Type{Categorical}, x::AbstractArray{T}) = fit_mle(Categorical, maximum(x), x)
fit_mle{T<:Integer}(::Type{Categorical}, x::AbstractArray{T}, w::AbstractArray{Float64}) = fit_mle(Categorical, maximum(x), x, w)

fit(::Type{Categorical}, data::CategoricalData) = fit_mle(Categorical, data)
fit(::Type{Categorical}, data::CategoricalData, w::AbstractArray{Float64}) = fit_mle(Categorical, data, w)
