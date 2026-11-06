# Performance overview of an experiment script (issue #10). Included at the top of every script: it starts the
# measurement and prints the overview when Julia exits (print_performance_report, src/performance.jl): wall-clock, CPU,
# compilation and GC time, allocations, the cost per cell and step of every model and the merged section timers.

using SweTransport2DV

# WHY the guard: a script that includes another script (or this file twice) must not reset the measurement or print
# the overview twice
if !@isdefined(EXPERIMENT_PERFORMANCE_REPORT)
    const EXPERIMENT_PERFORMANCE_REPORT = true
    reset_performance!()
    atexit(() -> print_performance_report(; title="Performance overview: " *
        (isempty(PROGRAM_FILE) ? "interactive session" : basename(PROGRAM_FILE))))
end
