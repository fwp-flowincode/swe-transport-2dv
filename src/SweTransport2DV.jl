module SweTransport2DV

using FFTW
using LinearAlgebra
using NPZ
using Plots
using Printf
using SparseArrays
using Statistics
using TimerOutputs

include("state.jl")
include("geometry.jl")
include("diagnostics.jl")
include("physics.jl")
include("pressure.jl")
include("visualization.jl")
include("turbulence.jl")
include("model.jl")
include("vegetation.jl")

export State
export copy_state, from_shape
export water_volume, constituent_mass, boundary_normal_flux, cfl_numbers
export reduced_gravity, internal_wave_speed, froude_number
export grad_x_c, grad_x_c!, grad_z_c, grad_z_c!, div_x_u, div_x_u!, div_z_w, div_z_w!
export advection_term_u, advection_term_u!, advection_term_w, advection_term_w!
export advection_term_rho, advection_term_rho!, advect_density, advect_density!
export buoyancy, diffusion_term, diffusion_term!
export solve_hydrostatic_pressure, solve_hydrostatic_pressure!
export build_poisson_matrix, solve_nonhydrostatic_pressure, PressureCorrectionSolver
export solve!
export QUIVER_STRIDE, QUIVER_TARGET_COUNT_X, QUIVER_TARGET_COUNT_Z, QUIVER_TARGET_LENGTH_FACTOR
export plot_state, create_animation, plot_pressure_buoyancy_diagnostics
export ModelConfig, Model, BarotropicSolver, solve_variable_thickness!, BedGeometry
export top_layer_face_thickness!, advect_density_top_layer!, use_variable_top_layer
export VegetationStemConfig, VegetationStem, VegetationModel, VegetationPatch, patch_frontal_area
export KEpsilonState, k_epsilon_step!, turbulent_viscosity_u!, implicit_vertical_diffusion!, bed_friction_coefficient, turbulence_closure
export KE_C_MU_VALUE, KE_C1_VALUE, KE_C2_VALUE
export update_stem_posture!, vegetation_drag_rate, stem_tip_deflection, stem_deflected_height, VEGETATION_EXPLICIT_DRAG_LIMIT
export attach_vegetation!, clear_vegetation!, vegetation_source!, copy_vegetation, vegetation_path, set_x_inflow_outflow!, clear_x_inflow_outflow!
export initialize_lock_exchange!, initialize_rayleigh_taylor!
export initialize_rising_plume!, initialize_sinking_plume!, initialize_jet_impulse!
export initialize_internal_seiche!, initialize_surface_dam_break!, initialize_kelvin_helmholtz!
export output_case_tag, copy_config, compute_diagnostics, pressure_buoyancy_components, step_forward!, run!, refresh_density_metadata!

end