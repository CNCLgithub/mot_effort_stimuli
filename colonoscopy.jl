using Gen
using CSV
using JSON
using MOTCore
using FillArrays
using DataFrames
using UnicodePlots
using Statistics: mean
using LinearAlgebra: norm
using Accessors: setproperties

include("running_stats.jl")

function gen_trial(wm::SchollWM, targets,
                   metrics::Metrics,
                   rejuv_steps::Int64 = 100,
                   particles = 10)
    nsteps = size(targets, 2)
    pf = initialize_particle_filter(col_chain, (0, wm, metrics),
                                    choicemap(), particles)
    obj = 1
    for i = 1:nsteps
        obs = choicemap((:states => i => :metrics,  targets[:, i]))
        particle_filter_step!(pf,
                              (i, wm, metrics),
                              (UnknownChange(), NoChange(), NoChange()),
                              obs)
        maybe_resample!(pf)
        if i % 5 == 0
            obj = categorical(Fill(1.0 / wm.n_dots, wm.n_dots))
        end
        Threads.@threads for p = 1:particles
            for _ = 1:rejuv_steps
                new_tr, w = regenerate(pf.traces[p],
                                    Gen.select(:states => i => :deltas => obj))
                if log(rand()) < w
                    pf.traces[p] = new_tr
                    pf.log_weights[p] += w
                end
            end
        end
    end
    # Extract MAP
    best_idx = argmax(get_log_weights(pf))
    best_tr = pf.traces[best_idx]
    (_, sofar) = get_retval(best_tr)
    nmetrics = length(metrics.funcs)
    vals = Vector{SVector{nmetrics, Float64}}(undef, nsteps)
    for i = 1:nsteps
        vals[i] = SVector{nmetrics, Float64}(metrics((sofar[i],)))
    end
    return best_tr, sofar, vals
end


function extend_trial(trace::Gen.Trace,
                      new_phase::Int,
                      wm::SchollWM, targets,
                      metrics::Metrics,
                      rejuv_steps::Int64 = 100,
                      particles = 10)
    nsteps = size(targets, 2)
    # choices = choicemap(get_choices(trace))
    choices = get_choices(trace)
    init_cm = get_selected(choices, Gen.select(:init_state))
    pf = initialize_particle_filter(col_chain, (0, wm, metrics),
                                    init_cm, particles)
    obj = 1
    for i = 1:nsteps
        obs = choicemap()
        # Stage 1: match initial positions
        addr = :states => i => :deltas
        if has_submap(choices, addr)
            set_submap!(obs, addr, get_submap(choices, addr))
        else
            # Stages 2 + 3 metrics
            obs[:states => i => :metrics] = targets[:, i]
        end
        particle_filter_step!(pf,
                              (i, wm, metrics),
                              (UnknownChange(), NoChange(), NoChange()),
                              obs)
        i < new_phase && continue # don't optimize Stage 1
        maybe_resample!(pf)

        if i % 5 == 0
            obj = categorical(Fill(1.0 / wm.n_dots, wm.n_dots))
        end
        Threads.@threads for p = 1:particles
            for _ = 1:rejuv_steps
                new_tr, w = regenerate(pf.traces[p],
                                    Gen.select(:states => i => :deltas => obj))
                if log(rand()) < w
                    pf.traces[p] = new_tr
                    pf.log_weights[p] += w
                end
            end
        end
    end
    # Extract MAP
    best_idx = argmax(get_log_weights(pf))
    best_tr = pf.traces[best_idx]
    (_, sofar) = get_retval(best_tr)
    nmetrics = length(metrics.funcs)
    vals = Vector{SVector{nmetrics, Float64}}(undef, nsteps)
    for i = 1:nsteps
        vals[i] = SVector{nmetrics, Float64}(metrics((sofar[i],)))
    end
    return sofar, vals
end

@gen static function col_kernel(t::Int, prev::SchollState,
                                 wm::SchollWM, metrics::Metrics)
    deltas ~ Gen.Map(scholl_delta)(Fill(wm, wm.n_dots))
    next::SchollState = MOTCore.step(wm, prev, deltas)
    mus = metrics((next,))
    metrics ~ broadcasted_normal(mus, 1.0)
    return next
end

@gen static function col_chain(k::Int, wm::SchollWM, metrics::Metrics)
    init_state ~ scholl_init(wm)
    states ~ Gen.Unfold(col_kernel)(k, init_state, wm, metrics)
    result = (init_state, states)
    return result
end

function test()

    dname = "colonoscopy_0.3"
    pairs = 10
    nexamples = 3

    wm = SchollWM(;
                  n_dots=8,
                  area_width = 720.0,
                  area_height = 480.0,
                  dot_radius = 20.0,
                  vel=2.75,
                  vel_min = 2.0,
                  vel_max = 3.75,
                  vel_step = 1.00,
                  vel_prob = 0.20
    )

    # dataset parameters
    epoch_dur = 3 # seconds | x3 for total
    fps = 24 # frames per second
    epoch_frames = epoch_dur * fps
    tot_frames = 5 * epoch_frames # 5 epochs total


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
    delta_e = 3.0

    D = [max(0.0, tdd_mu + delta_d * tdd_sd), ecc_mu, nd_mu]
    E = [tdd_mu + delta_e * tdd_sd, ecc_mu, nd_mu]

    @show stats
    @show D
    @show E

    dataset = []
    cond_list = []
    base = "output/$(dname)"
    isdir(base) || mkdir(base)


    df = DataFrame(:scene => UInt8[],
                   :frame => UInt32[],
                   :tdd => Float32[],
                   :ecc => Float32[],
                   :nobj => Float32[]
                   )


    part_a_end = Int(4 * epoch_frames)

    targets = repeat(hcat(E,E,D,E,E), inner = (1, epoch_frames))
    targets[1, :] = smooth(targets[1, :], fps)

    trial = 1
    for i = 1:pairs
        trace, part_a, _ = gen_trial(wm, targets[:, 1:part_a_end],
                                          metrics, 20, 300)
        # extend with two Easy segments
        part_b, vals = extend_trial(trace, part_a_end, wm,
                                      targets,
                                      metrics, 20, 300)

        push!(dataset, part_a)
        push!(cond_list, [trial, false])

        push!(dataset, part_b)
        push!(cond_list, [trial + 1, false])

        trial += 2

        d = Dict(:scene => i,
                 :frame => 1:tot_frames)

        for (mi, m) = enumerate(metrics.names)
            vs = map(x -> x[mi], vals)
            d[m] = vs
            ylim = (min(minimum(vs), D[mi]), E[mi])
            plt = lineplot(1:tot_frames,
                           targets[mi, :],
                           title = String(m),
                           xlabel = "time",
                           name = "target",
                           ylim = ylim
                           )
            lineplot!(plt, 1:tot_frames, vs, name = "measured")
            vline!(plt, part_a_end)
            display(plt)
        end
        append!(df, d)
    end
    write_dataset(dataset, "$(base)/dataset.json")
    write_condlist(cond_list, "$(base)/trial_list.json")
    CSV.write("$(base)/tdd.csv", df)

    examples = []
    for i = 1:pairs
        _, part_a, _ = gen_trial(wm, targets[:, 1:part_a_end],
                                   metrics, 20, 100)
        push!(examples, part_a)
    end
    write_dataset(examples, "$(base)/examples.json")
end

test();
