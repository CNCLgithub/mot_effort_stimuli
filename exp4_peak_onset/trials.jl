using Gen
using CSV
using JSON
using Term
using MOTCore
using DataFrames
using FillArrays
using UnicodePlots
using StaticArrays
using Statistics: mean
using LinearAlgebra: norm
using Accessors: setproperties
using MOTCore: scholl_delta, scholl_init

include("../running_stats.jl")
include("motion.jl")
include("iou.jl")

################################################################################
# Bayesian Optimizer
################################################################################

@gen function init_proposal(trace, obj_idx)
    xaddr = :dots => obj_idx => :x
    yaddr = :dots => obj_idx => :y
    mu_x = trace[xaddr]
    mu_y = trace[yaddr]
    @trace(normal(mu_x, 50.0), xaddr)
    @trace(normal(mu_y, 50.0), yaddr)
    return nothing
end

function gen_init_state(wm::MagneticWM, targets,
                        metrics::Metrics,
                        steps::Int64 = 10000)
    constraints = choicemap()
    constraints[:metrics => :obs] = targets
    trace, best_score = generate(magnetic_init, (wm,metrics), constraints)
    best_trace = trace
    for i = 1:steps
        obj_idx = ((i-1) % wm.n_dots) + 1
        trace, accept = mh(trace, init_proposal, (obj_idx,))
        if accept && get_score(trace) > best_score
            best_trace = trace
        end
    end
    get_retval(best_trace)
end

function gen_epoch(wm::MagneticWM,
                   istate::MagneticState,
                   metrics::Metrics,
                   targets::Vector{Float64},
                   polarity::Float64,
                   max_time_steps::Int64,
                   rejuv_steps::Int64 = 100,
                   particles = 1,
                   tol = 0.05)

    pf = initialize_particle_filter(magnetic_chain,
                                    (0, istate, wm, metrics),
                                    choicemap(), particles)

    for time_step = 1:max_time_steps
        constraints = choicemap(
            (time_step => :polarity, polarity),
            (time_step => :metrics => :obs,  targets)
        )
        particle_filter_step!(
            pf,
            (time_step, istate, wm, metrics),
            (UnknownChange(), NoChange(), NoChange(), NoChange()),
            constraints
        )
        maybe_resample!(pf)
        Threads.@threads for p = 1:particles
            for s = 1:rejuv_steps
                obj = categorical(Fill(1.0 / wm.n_dots, wm.n_dots))
                selections = Gen.select(time_step => :deltas => obj)
                new_tr, w = regenerate(pf.traces[p], selections)
                if log(rand()) < w
                    pf.traces[p] = new_tr
                    pf.log_weights[p] += w
                end
            end
        end
    end

    best_idx = argmax(get_log_weights(pf))
    best_tr = pf.traces[best_idx]
    states = get_retval(best_tr)
    return states
end

function reverse_frame(state::MagneticState)
    n = length(state.objects)
    new_objects = Vector{Dot}(undef, n)
    @inbounds for i = 1:n
        obj = state.objects[i]
        new_vel = get_vel(obj) .* S2V(-1.0, -1.0)
        new_objects[i] =
            Dot(obj.radius, get_pos(obj), new_vel)
    end
    MagneticState(new_objects, state.targets)
end

function package_targets(metrics::Metrics; kwargs...)
    @assert length(metrics) == length(kwargs) "Metric - Target length missmatch"
    n = length(metrics)
    targets = Vector{Float64}(undef, n)
    for (i, k) = enumerate(metrics.names)
        targets[i] = kwargs[k]
    end
    return targets
end

function visualize_frame(wm::MagneticWM, trace, frame::Int64, show=true)
    init, steps = get_retval(trace)
    state = frame == 0 ? init : steps[frame]
    visualize_frame(state, show, wm.area_width, wm.area_height)
end

function visualize_frame(wm::MagneticWM, state::MagneticState, show=true)
    visualize_frame(state, show, wm.area_width, wm.area_height)
end

function visualize_frame(state::MagneticState, show=true,
                         width = 720, height = 480)
    n = length(state.objects)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    colors = Vector{Symbol}(undef, n)
    @inbounds for i = 1:n
        xs[i], ys[i] = get_pos(state.objects[i])
        colors[i] = i <= 4 ? :blue : :red
    end
    plt = scatterplot(xs, ys, marker = :circle, color=colors,
                      xlim = (-0.5*width, 0.5*width),
                      ylim = (-0.5*height, 0.5*height),
                      )
    show && display(plt)
    return plt
end


function visualize_frame(wm::MagneticWM, a::MagneticState, b::MagneticState)
    UnicodePlots.panel(
        visualize_frame(wm, a, false)
    ) * UnicodePlots.panel(
        visualize_frame(wm, b, false)
    ) |> display
end

function check_tolerance(states::Vector,
                         ms::Metrics,
                         target::Vector{Float64},
                         tol::Float64 = 0.1)
    stats = report(states, ms)
    n = length(ms)
    i = 1
    passed = true
    while passed && i <= n
        mu, _ = stats[ms.names[i]]
        passed = (abs(mu - target[i]) / target[i]) < tol
        i += 1
    end
    return passed
end

function inspect_tolerance(state, ms::Metrics,
                           target::Vector{Float64},
                           tol::Float64 = 0.1)
    stats = ms((state,))
    n = length(ms)
    pct_error = Vector{Float64}(undef, n)
    for i = 1:n
        mu = stats[i]
        pct_error[i] = abs(mu - target[i]) / target[i]
    end
    failed = map(>(tol), pct_error)
    passed = !any(failed)
    if !passed
        @warn "Sample did not converge to targets"
    end
    colors = map(x -> x ? :red : :green, failed)
    plt = barplot(
        ms.names, pct_error, color = colors,
        xlabel = "|Error|",
        ylabel = "Metric",
        xlim = (0.0, max(tol, maximum(pct_error))),
        title = "Target Convergence",
    )
    display(plt)
    df = DataFrame(
        :metric => ms.names,
        :target => target,
        :actual => stats,
        Symbol("% Error") => pct_error,
    )
    display(df)
    return passed
end

function inspect_tolerance(
    states::AbstractVector,
    ms::Metrics,
    target::Vector{Float64},
    tol::Float64 = 0.15
    )
    n = length(ms)
    raw_error = Vector{Float64}(undef, n)
    pct_error = Vector{Float64}(undef, n)
    passed = true
    local stats
    for chunk = chunk_array(states, 10)
        stats = report(chunk, ms)
        passed =
            inspect_chunk_tolerance!(raw_error, pct_error,
                                     stats, ms , target,
                                     tol)
        if !passed
            @warn "Sample did not converge to targets"
            visualize_frame(chunk[end])
            break
        end
    end

    df = DataFrame(
        :metric => ms.names,
        :target => target,
        :actual => map(x -> first(stats[x]), ms.names),
        Symbol("% Error") => pct_error,
    )
    display(df)
    return passed
end

function inspect_chunk_tolerance!(
    raw_error::Vector{Float64},
    pct_error::Vector{Float64},
    stats,
    ms::Metrics,
    target::Vector{Float64},
    tol::Float64 = 0.15
    )
    n = length(ms)
    for i = 1:n
        mu, _ = stats[ms.names[i]]
        raw_error[i] = mu - target[i]
        pct_error[i] = abs(mu - target[i]) / target[i]
    end
    passed = all(<(tol), pct_error)
    return passed
end

function package_stats(trial::Vector, ms::Metrics)
    nm = length(ms)
    nt = length(trial)
    stats = Dict{Symbol, Vector{Float64}}()
    for m = ms.names
        stats[m] = Vector{Float64}(undef, nt)
    end
    for t = 1:nt
        state = trial[t]
        _stats = ms((state,))
        for i = 1:nm
            stats[ms.names[i]][t] = _stats[i]
        end
    end
    return stats
end

function gen_trial(wm::MagneticWM,
                   polarity::Float64,
                   peak_targets::Vector{Float64},
                   trough_targets::Vector{Float64},
                   metrics::Metrics,
                   peak_onset::Int64,
                   peak_ramp_window::Int64,
                   trial_frames::Int64;
                   flat::Bool = false
                   )
    init_state = gen_init_state(wm, trough_targets, metrics)
    println("Init frame:")
    visualize_frame(wm, init_state)
    passed_init =
        inspect_tolerance(init_state, metrics, trough_targets)
    if !passed_init
        println("Init frame malformed...restarting")
        return MagneticState[]
    end

    trial = fill(init_state, trial_frames)

    println("Pre-Peak")
    pre_peak = gen_epoch(wm, init_state, metrics, trough_targets, polarity,
                         peak_onset-1)
    pre_peak_passed =
        inspect_tolerance(pre_peak, metrics, trough_targets)
    visualize_frame(wm, pre_peak[end])
    pre_peak_passed || return MagneticState[]
    trial[1:(peak_onset-1)] = pre_peak

    println("Peak-Ascend")
    peak_ascend = gen_epoch(wm, pre_peak[end], metrics, peak_targets,
                            flat ? polarity : -polarity, # objects now attract
                            peak_ramp_window)
    peak_ascend_passed =
        inspect_tolerance(peak_ascend[end], metrics, peak_targets)
    visualize_frame(wm, peak_ascend[end])
    peak_ascend_passed || return MagneticState[]
    peak_ascend_end = peak_onset+peak_ramp_window-1
    trial[peak_onset:peak_ascend_end] = peak_ascend

    println("Peak-Descend")
    peak_descend = gen_epoch(wm, peak_ascend[end], metrics,
                             trough_targets,
                             polarity, # objects now repel
                             peak_ramp_window)
    peak_descend_passed =
        inspect_tolerance(peak_descend[end], metrics, trough_targets)
    visualize_frame(wm, peak_descend[end])
    peak_descend_passed || return MagneticState[]
    peak_descend_end = peak_ascend_end + peak_ramp_window
    trial[(peak_ascend_end+1):peak_descend_end] = peak_descend

    println("Post-Peak")
    post_peak = gen_epoch(wm, peak_descend[end], metrics, trough_targets,
                          polarity,
                          trial_frames - peak_descend_end)
    post_peak_passed =
        inspect_tolerance(post_peak, metrics, trough_targets)
    visualize_frame(wm, post_peak[end])
    post_peak_passed || return MagneticState[]
    trial[(peak_descend_end+1):end] = post_peak

    return trial
end


function common_fate(a::Dot, b::Dot)
    dpos = norm(get_pos(a) - get_pos(b))
    angle = vec2_angle(get_vel(a), get_vel(b))
    dvel = abs(sin(0.5 * angle))
    dpos * (dvel + 1)
end

function avg_common_fate(state::MagneticState)
    objects = state.objects
    mu = 0.0
    @inbounds for i = 1:4
        _mu = Inf
        for j = 5:8
            _mu =
                min(_mu, common_fate(objects[i], objects[j]))
        end
        mu += _mu
    end
    mu *= 0.25
    return mu
end

function tdminavg(state::MagneticState)
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

function warmup(wm::MagneticWM, metrics::Metrics,
                k::Int64=240, n::Int64=1000)
    stats = RunningStats(metrics)
    for _ = 1:n
        state = magnetic_init(wm, metrics)
        part = wm_magnetic(k, wm, state, metrics)
        stride!(stats, metrics, (part,))
    end
    report(stats, metrics)
end

function main()

    dname = "peak_onset"
    version = "2"

    wm = MagneticWM(;
        n_dots=8,
        area_width = 720.0,
        area_height = 480.0,
        dot_radius = 20.0,
        vel=4.25,
        vel_min = 3.5,
        vel_max = 5.0,
        vel_step = 1.00,
        mag_power = 2.0,
        mag_scale = 0.05,
        rep_power = 15.0,
        rep_scale = 0.15,
    )

    # dataset parameters
    npeaked = 12
    nflat   = 12
    nexamples = 3
    fps = 24 # frames per second
    trial_frames = round(Int64, 15 * fps)

    peak_ramp_window = round(Int64, fps * 1.5)

    tdd_scale = 1.00
    ecc_scale = 1.00
    cf_scale = 0.1

    no_max = 45.0
    nd_max = 175.0

    polarity = 0.8

    metrics = Metrics(;
        # keep targets and distractors close
        avg_distractor_distance = tddensity,
        nearest_distractor = x -> min(nd_max, tdminavg(x)),
        # target_distractor_iou = target_distractor_iou,
        # distract_density = x -> tdd_scale * tddensity(x),
        # common_fate = avg_common_fate,
        # but prevent occlusion
        # nearest_object = x -> min(no_max, nearest_obj(x)),
        # and encourage centering
        eccentricity = eccentricity,
    )

    @time stats = warmup(wm, metrics, trial_frames, 1000)
    display(stats)

    tdd_mu, tdd_sd = stats[:avg_distractor_distance]
    nd_mu, nd_sd = stats[:nearest_distractor]
    ecc_mu, ecc_sd = stats[:eccentricity]

    delta_h = -3.00 # Hard
    delta_m = -2.00 # Moderate
    delta_e =  3.00 # Easy

    H = package_targets(metrics;
        avg_distractor_distance = tdd_mu + (-3.50) * tdd_sd,
        nearest_distractor = nd_mu  + (-3.5) * nd_sd,
        eccentricity = ecc_mu - ecc_sd,
    )
    M = package_targets(metrics;
        avg_distractor_distance = tdd_mu + (-2.00) * tdd_sd,
        nearest_distractor = nd_mu  + (-2.00) * nd_sd,
        eccentricity = ecc_mu -  ecc_sd,
    )
    E = package_targets(metrics;
        avg_distractor_distance = tdd_mu + 2.0 * tdd_sd,
        nearest_distractor = nd_max,
        eccentricity = ecc_mu + ecc_sd,
    )

    @show H
    @show M
    @show E

    dataset = []
    cond_list = []
    examples = []
    base = "output/$(dname)_$(version)"
    if isdir(base)
    #     @warn "Dataset for version $(version) already exists!"
    #     return
    else
        mkdir(base)
    end



    df_schema = Dict(:scene => UInt8[],
                     :frame => UInt32[])
    for m = metrics.names
        df_schema[m] = Float32[]
    end
    df = DataFrame(df_schema)

    println("PEAKED TRIALS")
    complete = 1
    while complete <= npeaked
        ishard  = isodd(complete)
        peak_targets = H

        islate = complete <= 0.5 * npeaked
        peak_onset = round(Int64, 3.0 * fps)
        if islate
            peak_onset += round(Int64, 2.0 * fps)
        end

        trial = gen_trial(
            wm, polarity, peak_targets, E,
            metrics,
            peak_onset,
            peak_ramp_window,
            trial_frames
        )

        isempty(trial) && continue

        d = Dict(:scene => complete,
                 :frame => 1:trial_frames)
        merge!(d, package_stats(trial, metrics))

        for (mi, m) = enumerate(metrics.names)
            plt = lineplot(
                d[:frame],
                d[m],
                name = "measured",
                xlabel = "Time",
                ylabel = "Metric",
                title = "Convergence Trace: $(m)",
                # ylim = (H[mi]-1.0, E[mi]+1.0)
            )
            hline!(plt, E[mi], name = "target")
            vline!(plt, peak_onset, name = "onset")
            peak_end = peak_onset + 2*peak_ramp_window
            vline!(plt, peak_end, name = "end")
            display(plt)
        end
        complete += 1
        append!(df, d)
        push!(dataset, trial)
        push!(cond_list, [complete-1, false])
    end

    println("FLAT TRIALS")
    while complete <= npeaked + nflat
        peak_onset = round(Int64, 3.0 * fps) # not used
        trial = gen_trial(
            wm, polarity, E, E,
            metrics,
            peak_onset,
            peak_ramp_window,
            trial_frames;
            flat = true
        )

        isempty(trial) && continue

        d = Dict(:scene => complete,
                 :frame => 1:trial_frames)
        merge!(d, package_stats(trial, metrics))

        for (mi, m) = enumerate(metrics.names)
            plt = lineplot(
                d[:frame],
                d[m],
                name = "measured",
                xlabel = "Time",
                ylabel = "Metric",
                title = "Convergence Trace: $(m)",
                # ylim = (H[mi]-1.0, E[mi]+1.0)
            )
            hline!(plt, E[mi], name = "target")
            vline!(plt, peak_onset, name = "onset")
            peak_end = peak_onset + 2*peak_ramp_window
            vline!(plt, peak_end, name = "end")
            display(plt)
        end
        complete += 1
        append!(df, d)
        push!(dataset, trial)
        push!(cond_list, [complete-1, false])
    end

    write_dataset(dataset, "$(base)/dataset.json")
    write_condlist(cond_list, "$(base)/trial_list.json")
    CSV.write("$(base)/metrics.csv", df)

    while length(examples) < nexamples
        peak_onset = round(Int64, 4 * fps)
        trial = gen_trial(
            wm, polarity,
            H, E,
            metrics,
            peak_onset,
            peak_ramp_window,
            trial_frames
        )
        if !isempty(trial)
            push!(examples, trial)
        end
    end
    write_dataset(examples, "$(base)/examples.json")

    cp(@__FILE__, "$(base)/script.jl"; force = true)
end

main();
