using Gen
using CSV
using JSON
using MOTCore
using DataFrames
using FillArrays
using UnicodePlots
using StaticArrays
using Statistics: mean
using LinearAlgebra: norm
using Accessors: setproperties
using MOTCore: scholl_delta, scholl_init

include("running_stats.jl")

function gen_trial(wm::SchollWM, targets,
                   metrics::Metrics,
                   rejuv_steps::Int64 = 100,
                   particles = 10)
    nsteps = size(targets, 2)
    pf = initialize_particle_filter(peak_chain, (0, wm, metrics),
                                    choicemap(), particles)
    obj = 1
    for i = 1:nsteps
        obs = choicemap((:states => i => :metrics,  targets[:, i]),
                        (:states => i => :pos_var, 1.0))
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

function extract_positions(st::SchollState)
    n = length(st.objects)
    result = Vector{Float64}(undef, 2 * n)
    @inbounds for i = 1:n
        x, y = get_pos(st.objects[i])
        result[(2 * (i - 1) + 1)] = x
        result[(2 * i)] = y
    end
    return result
end

@gen static function peak_kernel(t::Int, prev::SchollState,
                                 wm::SchollWM, metrics::Metrics)
    deltas ~ Gen.Map(scholl_delta)(Fill(wm, wm.n_dots))
    next::SchollState = MOTCore.step(wm, prev, deltas)
    mus = metrics((next,))
    pos = extract_positions(next)
    metrics ~ broadcasted_normal(mus, 1.0)
    pos_var ~ uniform(0.0, 1000.0)
    positions ~ broadcasted_normal(pos, pos_var)
    return next
end

@gen static function peak_chain(k::Int, wm::SchollWM, metrics::Metrics)
    init_state ~ scholl_init(wm)
    states ~ Gen.Unfold(peak_kernel)(k, init_state, wm, metrics)
    result = (init_state, states)
    return result
end

function test()

    dname = "shifting_peak"
    version = "3"

    wm = SchollWM(;
                  n_dots=8,
                  area_width = 720.0,
                  area_height = 480.0,
                  dot_radius = 20.0,
                  vel=3.0,
                  vel_min = 2.0,
                  vel_max = 4.0,
                  vel_step = 1.00,
                  vel_prob = 0.20
    )

    # dataset parameters
    nscenes = 10
    ntrials = nscenes * 2
    nexamples = 3
    fps = 24 # frames per second
    epoch_frames = 24 # each epoch = 1s
    tot_frames = 20 * epoch_frames # 20s blue print - cut to 15s trials


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

    delta_h = -3.5
    delta_e = 3.0
    delta_m = 0.50 * delta_h

    H = [max(0.0, tdd_mu + delta_h * tdd_sd); ecc_mu; nd_mu]
    M = [tdd_mu + delta_m * tdd_sd; ecc_mu; nd_mu]
    E = [tdd_mu + delta_e * tdd_sd; ecc_mu; nd_mu]

    @show stats
    @show H
    @show M
    @show E

    dataset = []
    cond_list = []
    examples = []
    base = "output/$(dname)_$(version)"
    isdir(base) || mkdir(base)


    # 1 second bin example:
    # E E E E E E E H E E E E E E E E E E E E
    #           E E H E E E E E E E E E E E E
    # E E E E E E E H E E E E E E E

    cutoff = fps * 5 # remove 5s from start or end

    window = 12
    hard_epochs = hcat(
        E, E, E, E, E, E, E, H, H, H, E, E, E, E, E, E, E, E, E, E
    )
    hard_targets = repeat(hard_epochs, inner = (1, epoch_frames))
    hard_targets[1, :] = smooth(hard_targets[1, :], window)

    moderate_epochs = hcat(
        E, E, E, E, E, E, E, M, M, M, E, E, E, E, E, E, E, E, E, E
    )
    moderate_targets = repeat(moderate_epochs, inner = (1, epoch_frames))
    moderate_targets[1, :] = smooth(moderate_targets[1, :], window)



    df = DataFrame(:scene => UInt8[],
                   :frame => UInt32[],
                   :tdd => Float32[],
                   :ecc => Float32[],
                   :nobj => Float32[]
                   )

    for i = 1:nscenes

        targets = iseven(i) ? moderate_targets : hard_targets

        trace, src_frames, vals =
            gen_trial(wm, targets, metrics, 20, 100)

        earlier = src_frames[(cutoff+1):end]
        push!(dataset, earlier)
        push!(cond_list, [(i - 1) * 2 + 1, false])

        later = src_frames[1:(tot_frames - cutoff)]
        push!(dataset, later)
        push!(cond_list, [i * 2, false])



        d = Dict(:scene => i,
                 :frame => 1:tot_frames)

        for (mi, m) = enumerate(metrics.names)
            vs = map(x -> x[mi], vals)
            d[m] = vs
            plt = lineplot(1:tot_frames,
                           targets[mi, :],
                           title = String(m),
                           xlabel = "time",
                           name = "target",
                           ylim = (min(minimum(vs), H[mi]), E[mi])
                           )
            lineplot!(plt, 1:tot_frames, vs, name = "measured")

            if m == :tdd
                display(plt)
            end
        end
        append!(df, d)

    end
    write_dataset(dataset, "$(base)/dataset.json")
    write_condlist(cond_list, "$(base)/trial_list.json")
    CSV.write("$(base)/tdd.csv", df)

    diffs = combine(groupby(df, Cols(:scene)),
                    :tdd => mean)
    display(diffs)

    for i = 1:nexamples
        _, trial, vals = gen_trial(wm, hard_targets, metrics, 20, 100)
        trial = iseven(i) ? trial[1:(tot_frames - epoch_frames)] :
            trial[epoch_frames:end]
        push!(examples, trial)
    end
    write_dataset(examples, "$(base)/examples.json")

    cp(@__FILE__, "$(base)/script.jl"; force = true)
end

test();
