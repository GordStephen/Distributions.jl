struct GenericSampler{T<:Real, S<:AbstractVector{T}} <: Sampleable{Univariate,Discrete}
    support::S
    aliastable::AliasTable

    GenericSampler{T,S}(support::S, probs::Vector{<:Real}) where {T<:Real,S<:AbstractVector{T}} =
        new(support, AliasTable(probs))
end

GenericSampler(support::S, probs::Vector{<:Real}) where {T<:Real,S<:AbstractVector{T}} =
    GenericSampler{T,S}(support, probs)

sampler(d::Generic) =
    GenericSampler(d.support, d.p)

rand(rng::MersenneTwister, s::GenericSampler) =
    (@inbounds v = s.support[rand(rng, s.aliastable)]; v)

rand(s::GenericSampler) = rand(GLOBAL_RNG, s)
