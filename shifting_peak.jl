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

include("running_stats.jl")

@gen function init_proposal(trace, obj_idx)
    xaddr = :state => :dots => obj_idx => :x
    yaddr = :state => :dots => obj_idx => :y
    mu_x = trace[xaddr]
    mu_y = trace[yaddr]
    @trace(normal(mu_x, 20.0), xaddr)
    @trace(normal(mu_y, 20.0), yaddr)
    return nothing
end

function gen_peak(wm::SchollWM, targets,
                  metrics::Metrics,
                  steps::Int64 = 10000)
    constraints = choicemap()
    constraints[:metrics] = targets
    trace, best_score = generate(peak_init, (wm,metrics), constraints)
    best_state = get_retval(trace)
    for i = 1:steps
        obj_idx = ((i-1) % wm.n_dots) + 1
        trace, accept = mh(trace, init_proposal, (obj_idx,))
        if accept && get_score(trace) > best_score
            best_state = get_retval(trace)
        end
    end
    best_state
end

function descend_peak(wm::SchollWM, istate::SchollState,
                      metrics::Metrics, targets::Vector{Float64},
                      max_time_steps::Int64,
                      rejuv_steps::Int64 = 100,
                      particles = 1,
                      tol = 0.05)
    pf = initialize_particle_filter(peak_chain, (0, wm, istate, metrics),
                                    choicemap(), particles)
    obj = 1
    time_step = 1
    converged = false
    states = SchollState[]
    while !converged && time_step <= max_time_steps
        constraints = choicemap(
            (:states => time_step => :metrics,  targets),
        )
        particle_filter_step!(
            pf,
            (time_step, wm, istate, metrics),
            (UnknownChange(), NoChange(), NoChange(), NoChange()),
            constraints
        )
        maybe_resample!(pf)
        # if time_step % 5 == 0
        #     obj = categorical(Fill(1.0 / wm.n_dots, wm.n_dots))
        # end
        # selections = Gen.select(
        #     ((:states => t => :deltas => obj)
        #      for t = (max(1, time_step-3):time_step))...
        #          )
        # selections =
        #     Gen.select(
        #         ((:states => time_step => :deltas => obj) for obj = 4:8)...)
        Threads.@threads for p = 1:particles
            obj = categorical(Fill(1.0 / wm.n_dots, wm.n_dots))
            selections = Gen.select(
                ((:states => t => :deltas => obj)
                 for t = (max(1, time_step-3):time_step))...
                     )
            for s = 1:rejuv_steps
                # obj = ((s - 1) % 8) + 1
                new_tr, w = regenerate(pf.traces[p], selections)
                if log(rand()) < w
                    pf.traces[p] = new_tr
                    pf.log_weights[p] += w
                end
            end
        end
        time_step += 1
        time_step <= 6 && continue
        # check for convergence
        best_idx = argmax(get_log_weights(pf))
        best_tr = pf.traces[best_idx]
        (_, states) = get_retval(best_tr)
        converged = check_tolerance(
            states[(end-5):end],
            metrics, targets, tol
        )
    end
    return states
end

function extend_frame(wm::SchollWM, istate::SchollState,
                      metrics::Metrics, targets::Vector{Float64},
                      max_time_steps::Int64,
                      rejuv_steps::Int64 = 100,
                      particles = 10)
    pf = initialize_particle_filter(peak_chain, (0, wm, istate, metrics),
                                    choicemap(), particles)
    obj = 1
    for time_step = 1:max_time_steps
        constraints = choicemap(
            (:states => time_step => :metrics,  targets),
        )
        particle_filter_step!(
            pf,
            (time_step, wm, istate, metrics),
            (UnknownChange(), NoChange(), NoChange(), NoChange()),
            constraints
        )
        maybe_resample!(pf)
        # if time_step % 5 == 0
        #     obj = categorical(Fill(1.0 / wm.n_dots, wm.n_dots))
        # end
        Threads.@threads for p = 1:particles
            obj = categorical(Fill(1.0 / wm.n_dots, wm.n_dots))
            for s = 1:rejuv_steps
                # obj = ((s - 1) % 8) + 1
                selections = Gen.select(
                    ((:states => t => :deltas => obj)
                     for t = (max(1, time_step-3):time_step))...
                         )
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
    (_, states) = get_retval(best_tr)
    return states
end

@gen static function peak_init(wm::SchollWM, ms::Metrics)
    state ~ scholl_init(wm)
    mus = ms((state,))
    metrics ~ broadcasted_normal(mus, 1.0)
    return state
end

@gen static function peak_kernel(t::Int, prev::SchollState,
                                 wm::SchollWM, ms::Metrics)
    deltas ~ Gen.Map(scholl_delta)(Fill(wm, wm.n_dots))
    next::SchollState = MOTCore.step(wm, prev, deltas)
    mus = ms((next,))
    metrics ~ broadcasted_normal(mus, 1.0)
    return next
end

@gen static function peak_chain(k::Int, wm::SchollWM,
                                init_state::SchollState,
                                metrics::Metrics)
    states ~ Gen.Unfold(peak_kernel)(k, init_state, wm, metrics)
    result = (init_state, states)
    return result
end

function reverse_frame(state::SchollState)
    n = length(state.objects)
    new_objects = Vector{Dot}(undef, n)
    @inbounds for i = 1:n
        obj = state.objects[i]
        new_vel = get_vel(obj) .* S2V(-1.0, -1.0)
        new_objects[i] =
            Dot(obj.radius, get_pos(obj), new_vel)
    end
    SchollState(new_objects, state.targets)
end


function tdminavg(state::SchollState)
    # first 4 are targets
    objects = state.objects
    tdd = Inf
    @inbounds for i = 1:4
        tpos = get_pos(objects[i])
        _tdd = Inf
        for j = 5:8
            # REVIEW: consider l1 distance
            d = norm(tpos - get_pos(objects[j]))
            _tdd = min(_tdd, d)
        end
        tdd = min(_tdd, tdd)
    end
    tdd
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

function visualize_frame(wm::SchollWM, state::SchollState, show=true)
    n = length(state.objects)
    xs = Vector{Float64}(undef, n)
    ys = Vector{Float64}(undef, n)
    colors = Vector{Symbol}(undef, n)
    @inbounds for i = 1:n
        xs[i], ys[i] = get_pos(state.objects[i])
        colors[i] = i <= 4 ? :blue : :red
    end
    plt = scatterplot(xs, ys, marker = :circle, color=colors,
                      xlim = (-0.5*wm.area_width, 0.5*wm.area_width),
                      ylim = (-0.5*wm.area_height, 0.5*wm.area_height),
                      )
    show && display(plt)
    return plt
end


function visualize_frame(wm::SchollWM, a::SchollState, b::SchollState)
    UnicodePlots.panel(
        visualize_frame(wm, a, false)
    ) * UnicodePlots.panel(
        visualize_frame(wm, b, false)
    ) |> display
end

function check_tolerance(states::Vector{SchollState},
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

function inspect_tolerance(state::SchollState, ms::Metrics,
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
        xlabel = "% Error",
        ylabel = "Metric",
        xlim = (0.0, max(tol, maximum(pct_error))),
        title = "Target Convergence",
    )
    display(plt)
    return passed
end

function inspect_tolerance(states::Vector{SchollState}, ms::Metrics, target::Vector{Float64},
                           tol::Float64 = 0.15)
    stats = report(states, ms)
    n = length(ms)
    raw_error = Vector{Float64}(undef, n)
    pct_error = Vector{Float64}(undef, n)
    for i = 1:n
        mu, _ = stats[ms.names[i]]
        raw_error[i] = mu - target[i]
        pct_error[i] = abs(mu - target[i]) / target[i]
    end
    failed = map(>(tol), pct_error)
    passed = !any(failed)
    if !passed
        @warn "Sample did not converge to targets"
    end
    df = DataFrame(
        :metric => ms.names,
        :target => target,
        :actual => map(x -> first(stats[x]), ms.names),
        Symbol("% Error") => pct_error,
    )
    display(df)
    # colors = map(x -> x < 0 ? :red : :blue, raw_error)
    # names = map((name, failed) -> "$(name)\n$(failed)",
    #             ms.names, failed)
    # plt = barplot(
    #     ms.names, pct_error, color = colors,
    #     xlabel = "Error",
    #     ylabel = "Metric",
    #     xlim = (0.0, max(tol, maximum(pct_error))),
    #     title = "Target Convergence",
    # )
    # display(plt)
    return passed
end

function package_stats(trial::Vector{SchollState}, ms::Metrics)
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

function gen_trial(wm::SchollWM,
                   peak_targets::Vector{Float64},
                   trough_targets::Vector{Float64},
                   metrics::Metrics,
                   peak_onset::Int64,
                   peak_max_window::Int64,
                   peak_half_window::Int64,
                   trial_frames::Int64,
                   )
    state = gen_peak(wm, peak_targets, metrics)
    println("Peak frame:")
    visualize_frame(wm, state)
    passed_init =
        inspect_tolerance(state, metrics, peak_targets)
    if !passed_init
        println("Peak frame malformed...restarting")
        return SchollState[]
    end

    trial = fill(state, trial_frames)
    peak = descend_peak(wm, reverse_frame(state), metrics,
                        trough_targets, peak_half_window,
                        8, 200)
    platue = extend_frame(
        wm,
        state,
        metrics, peak_targets,
        peak_max_window,
        8, 200
    )

    println("Expanding before peak")
    before = descend_peak(wm, reverse_frame(state), metrics,
                          trough_targets, peak_half_window,
                          8, 200)
    nbefore = length(before)
    # println("Converged in $(nbefore) frames")
    before_passed =
        inspect_tolerance(before[end-5:end], metrics,
                          trough_targets)
    visualize_frame(wm, state, before[end])
    if !before_passed
        println("Peak expansion malformed...restarting")
        return SchollState[]
    end

    println("Expanding after peak")
    after  = descend_peak(wm, platue[end], metrics,
                          trough_targets, peak_half_window,
                          8, 200)
    nafter = length(after)
    after_passed =
        inspect_tolerance(after[end-5:end], metrics,
                          trough_targets)

    visualize_frame(wm, state, after[end])
    if !after_passed
        println("Peak expansion malformed...restarting")
        return SchollState[]
    end

    peak_top = peak_onset + nbefore - 1
    peak_platue = peak_top + peak_max_window
    peak_descend = peak_platue + 1
    peak_end = peak_descend + nafter - 1

    trial[peak_onset:peak_top] = reverse(before)
    trial[(peak_top+1):peak_platue] = platue
    trial[peak_descend:peak_end] = after

    println("Completing rest of trial")
    trial[1:(peak_onset-1)] =
        reverse(extend_frame(
            wm,
            trial[peak_onset],
            metrics, trough_targets,
            peak_onset-1,
            8, 100
        ))
    trial[(peak_end+1):end] = extend_frame(
        wm,
        trial[peak_end],
        metrics, trough_targets,
        trial_frames - peak_end,
        8, 100
    )
    return trial
end

function common_fate(a::Dot, b::Dot)
    dpos = norm(get_pos(a) - get_pos(b))
    angle = vec2_angle(get_vel(a), get_vel(b))
    dvel = abs(sin(0.5 * angle))
    dpos * (dvel + 1)
end

function avg_common_fate(state::SchollState)
    objects = state.objects
    @assert length(objects) === 8
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

function main()

    dname = "shifting_peak"
    version = "8"

    wm = SchollWM(;
        n_dots=8,
        area_width = 720.0,
        area_height = 480.0,
        dot_radius = 20.0,
        vel=4.0,
        vel_min = 3.80,
        vel_max = 4.20,
        vel_step = 0.50,
        vel_prob = 0.50
    )

    # dataset parameters
    nscenes = 4
    nexamples = 0
    fps = 24 # frames per second
    trial_frames = round(Int64, 15 * fps)

    tdd_scale = 1.00
    ecc_scale = 1.00
    cf_scale = 0.1

    metrics = Metrics(;
        # keep targets and distractors close
        avg_distractor_distance = tddensity,
        nearest_distractor = x -> tdminavg(x),
        # distract_density = x -> tdd_scale * tddensity(x),
        # common_fate = x -> cf_scale * avg_common_fate(x),
        # but prevent occlusion
        nearest_object = x -> min(80.0, nearest_obj(x)),
        # and encourage centering
        # eccentricity = x -> ecc_scale * eccentricity(x),
    )

    @time stats = warmup(wm, metrics, trial_frames, 1000)
    display(stats)

    tdd_mu, tdd_sd = stats[:avg_distractor_distance]
    nd_mu, nd_sd = stats[:nearest_distractor]
    # dd_mu, dd_sd = stats[:distract_density]
    # cf_mu, cf_sd = stats[:common_fate]
    # ecc_mu, ecc_sd = stats[:eccentricity]
    # no_mu, no_sd = stats[:nearest_object]
    no_hard = 55.0
    no_easy = 80.0

    delta_h = -3.00 # Hard
    delta_m = -2.00 # Moderate
    delta_e =  3.00 # Easy

    H = package_targets(metrics;
        avg_distractor_distance = tdd_mu + (-2) * tdd_sd,
        nearest_distractor = nd_mu + (-1.5) * nd_sd,
        # distract_density = dd_mu + delta_h * dd_sd,
        # common_fate = cf_mu + delta_h * cf_sd,
        nearest_object = no_hard,
        # eccentricity = ecc_mu + delta_h * ecc_sd,
        # eccentricity = ecc_mu - ecc_sd,
    )
    M = package_targets(metrics;
        avg_distractor_distance = tdd_mu + (-1.5) * tdd_sd,
        nearest_distractor = nd_mu + (-0.5) * nd_sd,
        # distract_density = dd_mu + delta_m * dd_sd,
        # common_fate = cf_mu + delta_m * cf_sd,
        nearest_object = no_hard,
        # eccentricity = ecc_mu + delta_m * ecc_sd,
        # eccentricity = ecc_mu -  ecc_sd,
    )
    E = package_targets(metrics;
        avg_distractor_distance = tdd_mu + 2.0 * tdd_sd,
        nearest_distractor = 200.0,
        # distract_density = dd_mu + delta_e * dd_sd,
        # common_fate = cf_mu + delta_e * cf_sd,
        nearest_object = no_easy,
        # eccentricity = ecc_mu + ecc_sd,
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

    peak_half_window = round(Int64, fps * 1.50)
    peak_max_window =  round(Int64, fps * 0.50)


    df_schema = Dict(:scene => UInt8[],
                     :frame => UInt32[])
    for m = metrics.names
        df_schema[m] = Float32[]
    end
    df = DataFrame(df_schema)

    complete = 1
    while complete <= nscenes
        ishard  = isodd(complete)
        peak_targets = ishard ? H : M

        islate = complete <= 0.5 * nscenes
        peak_onset = round(Int64, 3.0 * fps)
        if islate
            peak_onset += round(Int64, 2.0 * fps)
        end

        trial = gen_trial(
            wm, peak_targets, E,
            metrics,
            peak_onset, peak_max_window,
            peak_half_window,
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
            peak_end = peak_onset + 2*peak_half_window + peak_max_window
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

    for i = 1:nexamples
        peak_onset = round(Int64, 4 * fps)
        trial = gen_trial(
            wm, H, E,
            metrics,
            peak_onset, peak_max_window,
            peak_half_window,
            trial_frames
        )
        push!(examples, trial)
    end
    write_dataset(examples, "$(base)/examples.json")

    cp(@__FILE__, "$(base)/script.jl"; force = true)
end

main();
