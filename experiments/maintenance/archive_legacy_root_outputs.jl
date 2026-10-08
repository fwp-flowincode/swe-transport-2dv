using Dates

function archive_legacy_root_outputs!(; root::AbstractString=joinpath(@__DIR__, "..", "..", "output_barotropic"), dry_run::Bool=false)
    root_abs = abspath(root)
    if !isdir(root_abs)
        println("No output directory found at: $(root_abs)")
        return 0
    end

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    archive_dir = joinpath(root_abs, "_legacy_root_archive", timestamp)

    entries = readdir(root_abs)
    legacy_items = String[]

    for name in entries
        if name in ("_legacy_root_archive",)
            continue
        end

        p = joinpath(root_abs, name)
        if isfile(p)
            push!(legacy_items, name)
        end
    end

    if isempty(legacy_items)
        println("No legacy root-level files to archive in: $(root_abs)")
        return 0
    end

    println("Found $(length(legacy_items)) legacy root-level files.")
    println("Archive destination: $(archive_dir)")

    if dry_run
        println("Dry run enabled; no files moved.")
        for name in legacy_items
            println("  would move: $(name)")
        end
        return length(legacy_items)
    end

    mkpath(archive_dir)
    moved = 0

    for name in sort(legacy_items)
        src = joinpath(root_abs, name)
        dst = joinpath(archive_dir, name)
        mv(src, dst; force=true)
        moved += 1
        println("  moved: $(name)")
    end

    println("Archived $(moved) files to: $(archive_dir)")
    return moved
end

moved = archive_legacy_root_outputs!()
println("ARCHIVE_DONE moved=$(moved)")
