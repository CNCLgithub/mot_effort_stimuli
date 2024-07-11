using Gen
using CSV
using JSON
using MOTCore
using MOTCore: scholl_delta
using DataFrames
using UnicodePlots
using FillArrays
using Statistics: mean
using LinearAlgebra: norm
using Accessors: setproperties

include("running_stats.jl")

function trial_to_dict(states::Vector)
    positions = map(states) do state
        map(state.objects) do dot
            get_pos(dot)
        end
    end
    Dict(:positions => positions)
end

function write_dataset(dataset::Vector, json_path::String)
    out = map(trial_to_dict, dataset)
    open(json_path, "w") do f
        write(f, json(out))
    end
    return nothing
end

function write_condlist(condlist::Vector, path::String)
    open(path, "w") do f
        write(f, json(condlist))
    end
    return nothing
end

function gen_trial(wm::SchollWM, targets,
                   metrics::Metrics,
                   max_steps::Int64 = 3000)
    nsteps = length(targets)
    sofar = Vector{SchollState}(undef, nsteps)
    nmetrics = length(metrics.funcs)
    vals = Matrix{Float64}(undef, nmetrics, nsteps)
    current_step = 1
    block_idx = 1
    prev = burnin(wm, 12)
    for i = 1:nsteps
        m = targets[i]
        best_st = cur_st = scholl_kernel(1, prev, wm)
        best_ms = cur_ms = metrics((cur_st,))
        best_r = cur_r = resid = sum(abs.(m - cur_ms))
        t = 1
        while t < max_steps && !isapprox(resid, 0.)
            sample_st = scholl_kernel(1, prev, wm)
            sample_ms = metrics((sample_st,))
            sample_r = sum(abs.(sample_ms - m))
            # MH accept
            if log(rand()) < log(cur_r) - log(sample_r)
                cur_st = sample_st
                cur_ms = sample_ms
                cur_r = sample_r
                if sample_r < best_r
                    best_st = sample_st
                    best_ms = sample_ms
                    best_r = sample_r
                end
            end
            t += 1
        end
        prev = sofar[i] = best_st
        vals[:, i] = best_ms
    end
    return sofar, vals
end

function nearest_obj(state::SchollState)
    objects = state.objects
    d = Inf
    @inbounds for i = 1:7
        tpos = get_pos(objects[i])
        for j = (i+1):8
            d = min(norm(tpos - get_pos(objects[j])), d)
        end
    end
    clamp(d, 0., 45.) # prevent over-correction
end

function tddensity(state::SchollState)
    # first 4 are targets
    objects = state.objects
    avg_tdd = 0.0
    @inbounds for i = 1:4
        tdd = Inf
        tpos = get_pos(objects[i])
        for j = 5:8
            d = norm(tpos - get_pos(objects[j]))
            avg_tdd += d
        end
    end
    avg_tdd / 16
end


function tdmin(state::SchollState)
    # first 4 are targets
    objects = state.objects
    tdd = Inf
    @inbounds for i = 1:4
        tpos = get_pos(objects[i])
        for j = 5:8
            # REVIEW: consider l1 distance
            d = norm(tpos - get_pos(objects[j]))
            tdd = min(tdd, d)
        end
    end
    tdd
end

function eccentricity(state::SchollState)
    # first 4 are targets
    objects = state.objects
    no = length(objects)
    val = 0.0
    @inbounds for i = 1:no
        # tdd = Inf
        val += norm(get_pos(objects[i]))
    end
    2 * (val / no)
end

function burnin(wm::SchollWM, steps::Int64)
    state = scholl_init(wm)
    for t = 1:steps
        state = scholl_kernel(t, state, wm)
    end
    return state
end

function warmup(wm::SchollWM, metrics::Metrics,
                k::Int64=240, n::Int64=1000)
    stats = RunningStats(metrics)
    for _ = 1:n
        state = burnin(wm, 12)
        part = scholl_chain(k, state, wm)
        stride!(stats, metrics, (part,))
    end
    report(stats, metrics)
end

@gen static function peak_kernel(t::Int, prev::SchollState,
                                 wm::SchollWM, metrics::Metrics)
    deltas ~ Gen.Map(scholl_delta)(Fill(wm, wm.n_dots))
    next::SchollState = step(wm, prev_st, deltas)
    mus = metrics(next)
    metrics ~ broadcasted_normal(mus, 1.0)
    return next
end

function test()

    dname = "peak"
    version = "1"

    wm = SchollWM(;
                  n_dots=8,
                  area_width = 720.0,
                  area_height = 480.0,
                  dot_radius = 20.0,
                  vel=2.75,
                  vel_min = 2.0,
                  vel_max = 3.75,
                  vel_step = 2.00,
                  vel_prob = 0.20
    )

    # dataset parameters
    ntrials = 4
    nexamples = 0
    epoch_dur = 3 # seconds
    fps = 24 # frames per second
    epoch_frames = epoch_dur * fps
    tot_frames = 5 * epoch_frames # 3 epochs total


    metrics = Metrics(
        [tddensity, eccentricity, nearest_obj],
        [:tdd, :ecc, :nobj]
    )
    nm = length(metrics.funcs)

    @time stats = warmup(wm, metrics, tot_frames, 1000)
    display(stats)

    tdd_mu, tdd_sd = stats[:tdd]
    ecc_mu, _ = stats[:ecc]
    nd_mu = 50.0

    delta_d = -2.5
    delta_e = 5.0
    delta_m = 0.5 * (delta_d + delta_e)

    D = [max(0.0, tdd_mu + delta_d * tdd_sd), ecc_mu, nd_mu]
    M = [tdd_mu + delta_m * tdd_sd, ecc_mu, nd_mu]
    E = [tdd_mu + delta_e * tdd_sd, ecc_mu, nd_mu]

    @show stats
    @show D
    @show M
    @show E

    dataset = []
    cond_list = []
    examples = []
    base = "output/$(dname)_$(version)"
    isdir(base) || mkdir(base)

    perms = [
        [M, M, E, M, M],
        [M, M, D, M, M],
    ]

    df = DataFrame(:scene => Int32[],
                   :epoch => Int32[],
                   :tdd => Float32[],
                   :ecc => Float32[],
                   :nobj => Float32[]
                   )
    # total of 12 trials
    for i = 1:ntrials
        perm = perms[i % 2 + 1]
        ne = length(perm)
        targets = repeat(perm, inner = epoch_frames)
        trial, vals = gen_trial(wm, targets, metrics, 2000)
        push!(dataset, trial)
        push!(cond_list, [i, false])
        # d = Dict(:scene => i,
        #          :epoch => 1:ne)

        @show map(first, perm)
        for (mi, m) = enumerate(metrics.names)
            plt = lineplot(1:tot_frames,
                           map(x -> x[mi], targets),
                           title = String(m),
                           xlabel = "time",
                           name = "target"
                           )
            lineplot!(plt, 1:tot_frames,
                      vec(vals[mi, :]),
                      name = "measured"
                      )
            display(plt)
            # vs = reshape(vals[mi, :], (epoch_frames, ne))
            # d[m] = vec(mean(vs, dims = 1))
        end
        # display(d)
        # append!(df, d)
    end
    write_dataset(dataset, "$(base)/dataset.json")
    write_condlist(cond_list, "$(base)/trial_list.json")
    CSV.write("$(base)/tdd.csv", df)

    for i = 1:nexamples
        perm = perms[i % 2 + 1]
        ne = length(perm)
        targets = repeat(perm, inner = epoch_frames)
        trial, vals = gen_trial(wm, targets, metrics, 1000)
        push!(examples, trial)
    end
    write_dataset(examples, "$(base)/examples.json")
end

test();
