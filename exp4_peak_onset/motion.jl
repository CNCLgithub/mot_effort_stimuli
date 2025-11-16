using Gen
using MOTCore
using FillArrays
using StaticArrays
using LinearAlgebra: dot, norm, normalize


include("../running_stats.jl")

################################################################################
# Motion model
################################################################################

Base.@kwdef struct MagneticWM <: WorldModel
    n_dots::Int64 = 8
    dot_radius::Float64 = 20.0
    area_width::Float64 = 720.0
    area_height::Float64 = 480.0
    mag_power::Float64 = 100.0
    mag_scale::Float64 = 0.05
    rep_power::Float64 = 300.0
    rep_scale::Float64 = 0.2
    max_distance::Float64 = 400.0
    vel::Float64 = 4.0 # base velocity
    vel_min::Float64 = 3.0
    vel_max::Float64 = 5.0
    vel_step::Float64 = 0.90
end

struct MagneticState <: WorldState{MagneticWM}
    objects::Vector{Dot}
end

function tracker_bounds(gm::MagneticWM)
    xs = (-0.5*gm.area_width + gm.dot_radius, 0.5*gm.area_width - gm.dot_radius)
    ys = (-0.5*gm.area_height + gm.dot_radius, 0.5*gm.area_height - gm.dot_radius)
    (xs, ys)
end

function Dot(wm::MagneticWM, pos, vel)
    Dot(wm.dot_radius, pos, vel)
end

function step(gm::MagneticWM,
              state::MagneticState,
              forces::AbstractVector{<:SVector{2, Float64}},
              polarity::Float64)
    # Dynamics (computing forces)
    n_dots = gm.n_dots
    objects = state.objects
    new_dots = Vector{Dot}(undef, n_dots)

    @inbounds for i in eachindex(objects)
        dot = objects[i]
        # force accumalator
        facc = MVector{2, Float64}(forces[i])
        for j in eachindex(objects)
            i === j && continue
            force!(facc, gm, objects[j], dot, polarity)
        end
        new_pos, new_vel = update_kinematics(gm, dot, facc)
        new_dots[i] = Dot(gm, new_pos, new_vel)
    end
    MagneticState(new_dots)
end

function force!(f::MVector{2, Float64}, gm::MagneticWM, x::Dot, d::Dot,
                polarity::Float64)
    v = get_pos(d) - get_pos(x)
    d = norm(v) - d.radius - x.radius
    d = clamp(d, 0.1, gm.max_distance)
    mag = polarity*gm.mag_power * exp(-d * gm.mag_scale)
    rep = gm.rep_power * exp(-d * gm.rep_scale)
    delta_f = (mag + rep) * normalize(v)
    f .+= delta_f
    return nothing
end

function update_kinematics(gm::MagneticWM, d::Dot, f::MVector{2, Float64})
    new_vel = d.vel + f
    vel_mag = max(norm(new_vel), 1E-3)
    if vel_mag < gm.vel_min
        new_vel = (gm.vel_min / vel_mag) .* new_vel
    elseif vel_mag > gm.vel_max
        new_vel = (gm.vel_max / vel_mag) .* new_vel
    end
    # adjust out of bounds positions
    pos = get_pos(d)
    new_pos = pos + new_vel
    if !in_bounds(gm, new_pos)
        new_vel = deflect(gm, pos, new_vel)
        new_pos = pos + new_vel
    end
    new_pos = SVector{2, Float64}(
        clamp(new_pos[1], -gm.area_width * 0.5 + gm.dot_radius,
              gm.area_width * 0.5  - gm.dot_radius),
        clamp(new_pos[2], -gm.area_height * 0.5 + gm.dot_radius,
              gm.area_height * 0.5  - gm.dot_radius)
    )
    return (new_pos, new_vel)
end

function in_bounds(gm::MagneticWM, position::SVector{2, Float64})
    (abs(position[1]) + gm.dot_radius) < 0.5 * gm.area_width &&
        (abs(position[2]) + gm.dot_radius) < 0.5 * gm.area_height
end


const vec_up = SVector{2, Float64}([0, 1])
const vec_down = SVector{2, Float64}([0, -1])
const vec_left = SVector{2, Float64}([-1, 0])
const vec_right = SVector{2, Float64}([1, 0])

function rot2dvec(vec::SVector{2, Float64}, rad::Float64)
    cs = cos(rad)
    sn = sin(rad)
    x, y = vec
    SVector{2, Float64}(x * cs - y * sn,
                        x * sn + y * cs)
end

function deflect(gm::MagneticWM, pos::T, vel::T) where {T<:SVector{2, Float64}}
    tpos = pos + vel
    nv = normalize(vel)
    dir = if tpos[1] < (-0.5 * gm.area_width + gm.dot_radius)
        # left wall
        vec_down
    elseif tpos[1] > (0.5 * gm.area_width - gm.dot_radius)
        # right wall
        vec_up
    elseif tpos[2] < (-0.5 * gm.area_height + gm.dot_radius)
        # bottom wall
        vec_right
    else
        # top wall
        vec_left
    end
    aoi = acos(dot(nv, dir))
    new_vel = rot2dvec(vel, 2.0 * aoi)
end

################################################################################
# Initial State
################################################################################
@gen static function magnetic_dot(wm::MagneticWM)
    xs, ys = tracker_bounds(wm)
    x = @trace(uniform(xs[1], xs[2]), :x)
    y = @trace(uniform(ys[1], ys[2]), :y)

    ang = @trace(uniform(0.0, 2*pi), :ang)
    mag = @trace(normal(wm.vel, 1.0), :std)

    pos = SVector{2, Float64}(x, y)
    vel = SVector{2, Float64}(mag*cos(ang), mag*sin(ang))

    new_dot::Dot = Dot(wm, pos, vel)
    return new_dot
end

################################################################################
# Dynamics
################################################################################

@gen (static) function magnetic_force(wm::MagneticWM)
    fx = @trace(normal(0, wm.vel_step), :fx)
    fy = @trace(normal(0, wm.vel_step), :fy)
    f::SVector{2, Float64} = SVector{2, Float64}(fx, fy)
    return f
end

@gen (static) function obs_metrics(state::MagneticState, ms::Metrics)
    mus = ms((state,))
    obs ~ broadcasted_normal(mus, 1.0)
    return mus
end

@gen (static) function magnetic_kernel(t::Int,
                                       prev_st::MagneticState,
                                       wm::MagneticWM,
                                       ms::Metrics)
    polarity ~ uniform(-1.0, 1.0) # pull or push
    deltas ~ Gen.Map(magnetic_force)(Fill(wm, wm.n_dots))
    next_st::MagneticState = step(wm, prev_st, deltas, polarity)
    metrics ~ obs_metrics(next_st, ms)
    return next_st
end

magnetic_chain = Gen.Unfold(magnetic_kernel)

@gen static function magnetic_init(wm::MagneticWM, ms::Metrics)
    wms = Fill(wm, wm.n_dots)
    dots ~ Gen.Map(magnetic_dot)(wms)
    state::MagneticState = MagneticState(dots)
    metrics ~ obs_metrics(state, ms)
    return state
end


@gen (static) function wm_magnetic(k::Int, wm::MagneticWM, init_state::MagneticState, ms::Metrics)
    states ~ magnetic_chain(k, init_state, wm, ms)
    return states
end
