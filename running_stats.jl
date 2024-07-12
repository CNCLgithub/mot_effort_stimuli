using Statistics

struct Metrics
    funcs::Vector{Function}
    names::Vector{Symbol}

    function Metrics(fs, ns)
        @assert length(fs) == length(ns)
        new(fs, ns)
    end
end

function (ms::Metrics)(args::Tuple)
    n = length(ms.funcs)
    vs = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        vs[i] = ms.funcs[i](args...)
    end
    return vs
end

mutable struct RunningStats
    mus::Vector{Float64}
    vrs::Vector{Float64}
    c::Int64
end

function RunningStats(ms::Metrics)
    n = length(ms.names)
    RunningStats(zeros(n), zeros(n), 1)
end

function stride!(stats::RunningStats, ms::Metrics, args::Tuple)
    for mi = eachindex(ms.funcs)
        f = ms.funcs[mi]
        x = mean(f, args...)
        m_prev = stats.mus[mi]
        m = m_prev + ((x - m_prev) / stats.c)
        stats.vrs[mi] += (x - m_prev) * (x - m)
        stats.mus[mi] = m
    end
    stats.c += 1
end

function report(stats::RunningStats, ms::Metrics)
    n = length(ms.funcs)
    mus = Vector{Float64}(undef, n)
    sds = Vector{Float64}(undef, n)
    for mi = 1:n
        mus[mi] = stats.mus[mi]
        sds[mi] = sqrt(stats.vrs[mi] / (stats.c - 1))
    end

    Dict{Symbol, Tuple}(zip(ms.names, zip(mus, sds)))
end

function smooth(xs::Vector{Float64}, w::Int64)
    n = length(xs)
    ys = zeros(length(xs))
    w = isodd(w) ? w : w + 1
    hw = Int64((w + 1) / 2)
    @inbounds for i = 1:n
        c = 0
        for j = max(1, i - hw + 1):min(n, i + hw - 1)
            ys[i] += xs[j]
            c += 1
        end
        ys[i] *= 1.0 / c
    end
    return ys
end
