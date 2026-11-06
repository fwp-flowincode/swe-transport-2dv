# Performance overview of a session (issue #10): every Model records its steps, the wall-clock time and the allocations
# of step_forward!; together with its section timers (TimerOutputs) they are collected in a session list, and
# print_performance_report prints wall-clock, CPU, compilation and GC time, the allocations, the cost per cell and step
# of every model and the merged section timers. The experiment scripts print it at exit
# (experiments/common/performance.jl).

"""
    ModelPerformance

Cost record of one Model: `label` (grid and mode), `cells` (nx nz), the section timers of the model and of its
pressure-correction solver, `steps` (number of step_forward! calls), `step_time_ns` (ns, wall-clock time spent in them)
and `step_bytes` (bytes allocated in them).
"""
mutable struct ModelPerformance
    label::String
    cells::Int
    timers::TimerOutput
    pressure_timers::TimerOutput
    steps::Int
    step_time_ns::UInt64
    step_bytes::Int64
end

# Session state: the records of the models built since the last reset_performance!, and the wall-clock time (ns), CPU
# time (s), GC counters and compilation time (ns) at that reset.
const _PERFORMANCE_MODELS = ModelPerformance[]
const _PERFORMANCE_START = Ref((wall_ns=UInt64(0), cpu_s=0.0, gc=Base.gc_num(), compile_ns=UInt64(0)))
# WHY a root timer created at the reset: the merged section table then shows the session total and the share measured
# in the sections in its header (a timer created at print time would show nonsense there)
const _PERFORMANCE_ROOT = Ref(TimerOutput())
const PERFORMANCE_REPORT_MAX_MODELS = 20   # rows of the per-model table; further models are summed in one row

function _register_performance!(label::AbstractString, cells::Int, timers::TimerOutput, pressure_timers::TimerOutput)
    p = ModelPerformance(String(label), cells, timers, pressure_timers, 0, UInt64(0), Int64(0))
    push!(_PERFORMANCE_MODELS, p)
    return p
end

@inline function _record_step!(p::ModelPerformance, elapsed_ns::UInt64, bytes::Int64)
    p.steps += 1
    p.step_time_ns += elapsed_ns
    p.step_bytes += bytes
    return nothing
end

"""
    process_cpu_time() -> Float64

CPU time (s) of this process so far, user plus system, summed over all its threads (Julia, BLAS, FFTW); NaN if the
operating system does not report it. Windows: GetProcessTimes; Linux and macOS: getrusage(RUSAGE_SELF).
"""
function process_cpu_time()
    if Sys.iswindows()
        handle = ccall((:GetCurrentProcess, "kernel32"), stdcall, Ptr{Cvoid}, ())
        creation, exit, kernel, user = Ref{UInt64}(0), Ref{UInt64}(0), Ref{UInt64}(0), Ref{UInt64}(0)
        ok = ccall((:GetProcessTimes, "kernel32"), stdcall, Cint,
            (Ptr{Cvoid}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}, Ref{UInt64}), handle, creation, exit, kernel, user)
        return ok == 0 ? NaN : (kernel[] + user[]) * 1e-7   # FILETIME: 100 ns units
    end
    # struct rusage begins with ru_utime and ru_stime (struct timeval: seconds, microseconds); 144 bytes on 64-bit
    # Linux and macOS, the buffer is larger
    buf = zeros(Int64, 32)
    ccall(:getrusage, Cint, (Cint, Ptr{Int64}), 0, buf) == 0 || return NaN
    # WHY the mask: tv_usec is a 32-bit integer on macOS, followed by padding
    usec(x) = Sys.isapple() ? (x & 0xffffffff) : x
    return buf[1] + 1e-6 * usec(buf[2]) + buf[3] + 1e-6 * usec(buf[4])
end

_compile_time_ns() = isdefined(Base, :cumulative_compile_time_ns) ? UInt64(first(Base.cumulative_compile_time_ns())) : UInt64(0)

"""
    reset_performance!()

Starts a new performance overview: forgets the models built so far and takes the wall-clock time, CPU time, GC
counters and compilation time as the new origin of `print_performance_report`. Also switches on the measurement of the
compilation time (Base.cumulative_compile_timing).
"""
function reset_performance!()
    empty!(_PERFORMANCE_MODELS)
    isdefined(Base, :cumulative_compile_timing) && Base.cumulative_compile_timing(true)
    _PERFORMANCE_ROOT[] = TimerOutput()
    _PERFORMANCE_START[] = (wall_ns=time_ns(), cpu_s=process_cpu_time(), gc=Base.gc_num(), compile_ns=_compile_time_ns())
    return nothing
end

"""
    performance_models() -> Vector{ModelPerformance}

The cost records of the models built since the last `reset_performance!` (or since the package was loaded).
"""
performance_models() = _PERFORMANCE_MODELS

function _format_bytes(bytes::Real)
    bytes < 1024 && return @sprintf("%d B", round(Int, bytes))
    for (unit, scale) in (("GiB", 2.0^30), ("MiB", 2.0^20), ("KiB", 2.0^10))
        bytes >= scale && return @sprintf("%.2f %s", bytes / scale, unit)
    end
end

"""
    print_performance_report(io=stdout; title="Performance overview", timers=true)

Prints the performance overview since the last `reset_performance!`:
- wall-clock time, CPU time (all threads; the ratio CPU/wall shows the use of the BLAS and FFTW threads), compilation
  time, GC time and allocated memory of the whole session;
- one row per model that took steps (at most PERFORMANCE_REPORT_MAX_MODELS, further models in one row): grid and mode,
  steps, wall-clock time of the steps (s), cost per cell and step (us), allocations per step;
- with `timers`, the section timers of all models merged (TimerOutputs, time and allocations per section).
The cost per cell and step, wall time / (steps nx nz), compares set-ups of different size; the lock exchange of the
non-hydrostatic fixed grid takes about 0.13 us (1500 x 200, spectral pressure solver; docs, section 32).
"""
function print_performance_report(io::IO=stdout; title::AbstractString="Performance overview", timers::Bool=true)
    start = _PERFORMANCE_START[]
    wall = (time_ns() - start.wall_ns) * 1e-9
    cpu = process_cpu_time() - start.cpu_s
    gc = Base.GC_Diff(Base.gc_num(), start.gc)
    compile = (_compile_time_ns() - start.compile_ns) * 1e-9
    stepped = filter(p -> p.steps > 0, _PERFORMANCE_MODELS)
    println(io, title)
    println(io, "  " * "-"^(length(title) + 2))
    @printf(io, "  wall-clock time   %10.2f s\n", wall)
    @printf(io, "  CPU time          %10.2f s   (%.1f x wall-clock; Julia threads %d, BLAS threads %d)\n", cpu,
        cpu / max(wall, eps()), Threads.nthreads(), LinearAlgebra.BLAS.get_num_threads())
    @printf(io, "  compilation       %10.2f s\n", compile)
    @printf(io, "  garbage collection%10.2f s\n", gc.total_time * 1e-9)
    @printf(io, "  allocated         %12s   (%d allocations)\n", _format_bytes(gc.allocd), Base.gc_alloc_count(gc))
    steps = sum(p -> p.steps, stepped; init=0)
    step_time = sum(p -> p.step_time_ns, stepped; init=UInt64(0)) * 1e-9
    cell_steps = sum(p -> p.steps * p.cells, stepped; init=0)
    @printf(io, "  models %d, steps %d, time in the steps %.2f s (%.0f %% of the wall-clock time)\n", length(stepped), steps,
        step_time, 100 * step_time / max(wall, eps()))
    if !isempty(stepped)
        println(io)
        @printf(io, "  %-52s %9s %10s %12s %14s\n", "model", "steps", "time (s)", "us/cell/step", "alloc/step")
        rows = stepped[1:min(end, PERFORMANCE_REPORT_MAX_MODELS)]
        for p in rows
            @printf(io, "  %-52s %9d %10.2f %12.3f %14s\n", first(p.label, 52), p.steps, p.step_time_ns * 1e-9,
                p.step_time_ns * 1e-3 / (p.steps * p.cells), _format_bytes(p.step_bytes / p.steps))
        end
        if length(stepped) > length(rows)
            rest = stepped[(length(rows) + 1):end]
            n, t, b = sum(p -> p.steps, rest), sum(p -> p.step_time_ns, rest), sum(p -> p.step_bytes, rest)
            cells = sum(p -> p.steps * p.cells, rest)
            @printf(io, "  %-52s %9d %10.2f %12.3f %14s\n", "$(length(rest)) further models", n, t * 1e-9, t * 1e-3 / cells,
                _format_bytes(b / n))
        end
        @printf(io, "  %-52s %9d %10.2f %12.3f\n", "all", steps, step_time, step_time * 1e6 / cell_steps)
    end
    if timers && !isempty(stepped)
        merged = deepcopy(_PERFORMANCE_ROOT[])
        for p in stepped
            merge!(merged, p.timers, p.pressure_timers)
        end
        println(io)
        print_timer(io, merged; allocations=true, sortby=:time, linechars=:ascii, title="Sections of all models")
        println(io)
    end
    return nothing
end
