using Gen
using CSV
using JSON
using MOTCore
using DataFrames
using UnicodePlots
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
        # println("$(i): $(first(m))")
        st = i == 1 ? burnin(wm, 12) : scholl_kernel(1, prev, wm)
        xs = metrics((st,))
        rxs = sum(abs.(xs - m))
        t = 1
        while t < max_steps && !isapprox(rxs, 0.)
            _st = i == 1 ? burnin(wm, 12) : scholl_kernel(1, prev, wm)
            vs = metrics((_st,))
            rvs = sum(abs.(vs - m))
            if rxs > rvs # log(1) < log(rxs) - log(rvs)
                st = _st
                xs = vs
                rxs = rvs
            end
            t += 1
        end
        sofar[i] = st
        vals[:, i] = xs
        # println("  $(first(xs))")
        prev = st
    end
    return sofar, vals
end

function extend_trial(wm::SchollWM,
                      prev::Vector{SchollState},
                      targets,
                      metrics::Metrics,
                      max_steps::Int64 = 3000)
    nsteps = length(targets)
    sofar = Vector{SchollState}(undef, nsteps)
    nmetrics = length(metrics.funcs)
    vals = Matrix{Float64}(undef, nmetrics, nsteps)
    current_step = 1
    block_idx = 1
    prev_st = last(prev)
    for i = 1:nsteps
        m = targets[i]
        st = scholl_kernel(1, prev_st, wm)
        xs = metrics((st,))
        rxs = sum(abs.(xs - m))
        t = 1
        while t < max_steps && !isapprox(rxs, 0.)
            _st = scholl_kernel(1, prev_st, wm)
            vs = metrics((_st,))
            rvs = sum(abs.(vs - m))
            if rxs > rvs # log(1) < log(rxs) - log(rvs)
                st = _st
                xs = vs
                rxs = rvs
            end
            t += 1
        end
        sofar[i] = st
        vals[:, i] = xs
        prev_st = st
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

function test()

    dname = "colonoscopy_0.1"
    pairs = 10
    nexamples = 3

    wm = SchollWM(;
                  n_dots=8,
                  area_width = 720.0,
                  area_height = 480.0,
                  dot_radius = 20.0,
                  vel=3.75,
                  vel_min = 3.0,
                  vel_max = 4.75,
                  vel_step = 0.75,
                  vel_prob = 0.30
    )

    # dataset parameters
    epoch_dur = 3 # seconds | x3 for total
    fps = 24 # frames per second
    epoch_frames = epoch_dur * fps


    metrics = Metrics(
        [tddensity, eccentricity, nearest_obj],
        [:tdd, :ecc, :nobj]
    )
    nm = length(metrics.funcs)

    @time stats = warmup(wm, metrics, 4 * epoch_frames, 2000)
    display(stats)

    tdd_mu, tdd_sd = stats[:tdd]
    ecc_mu, _ = stats[:ecc]
    nd_mu = 50.0

    delta_d = -2.5
    delta_e = 5.0

    D = [max(0.0, tdd_mu + delta_d * tdd_sd), ecc_mu, nd_mu]
    E = [tdd_mu + delta_e * tdd_sd, ecc_mu, nd_mu]

    @show stats
    @show D
    @show E

    dataset = []
    cond_list = []
    base = "output/$(dname)"
    isdir(base) || mkdir(base)


    df = DataFrame(:scene => Int32[],
                   :epoch => Int32[],
                   :tdd => Float32[],
                   :ecc => Float32[],
                   :nobj => Float32[]
                   )

    perm = [E, E, D]
    ne = length(perm)
    itargets = repeat(perm, inner = epoch_frames)
    ftargets = repeat(E, inner = epoch_frames)

    trial = 1
    for i = 1:pairs
        targets = repeat(perm, inner = epoch_frames)
        part_a, vals_a = gen_trial(wm, targets, metrics, 5000)
        # extend with two Easy segments
        part_b, vals_b = extend_trial(wm, part_a,
                                      fill(E, 2 * epoch_frames),
                                      metrics, 5000)

        push!(dataset, part_a)
        push!(cond_list, [trial, false])

        push!(dataset, [part_a; part_b])
        push!(cond_list, [trial + 1, false])

        trial += 2

        d = Dict(:scene => i,
                 :epoch => 1:5)

        vals = hcat(vals_a, vals_b)
        for (mi, m) = enumerate(metrics.names)
            vs = reshape(vals[mi, :], (epoch_frames, 5))
            d[m] = vec(mean(vs, dims = 1))
        end
        display(d)
        append!(df, d)
    end
    write_dataset(dataset, "$(base)/dataset.json")
    write_condlist(cond_list, "$(base)/trial_list.json")
    CSV.write("$(base)/tdd.csv", df)

    examples = []
    for i = 1:pairs
        targets = repeat(perm, inner = epoch_frames)
        part_a, vals_a = gen_trial(wm, targets, metrics, 5000)
        push!(examples, part_a)
    end
    write_dataset(examples, "$(base)/examples.json")
end

test();
