const working_directory = string(pwd(), "/")
const random_number_generator_seed = 123

const file_id = string("processed_trajectory_invivo")
const data_range = 1:920000 #Use Colon() to choose full data range

# Availability of Ground Truth
const ground_truth_available = false

# Known Constants
const kb = 1.38e-23 # (Kg*m^2/s^2)/K
const T = 300 # K
const dt = 5.0e-7 # s

# Output Parameters
const output_frequency::Int64 = 1000
const plotting_on::Bool = false
const plotting_frequency::Int64 = output_frequency
const save_plots::Bool = false
const data_plotting_linewidth = 1.0
const data_plotting_opacity = 0.4
const data_plotting_sampling_frequency = 100

const n_samples_to_save::Int64 = 200

##### Inference Parameters #####

# Parameters to bound jump size for numerical stability 
const minimum_allowed_jump_size = 5.0e-9 # m
const maximum_allowed_jump_size = 50.0e-9 # m

# Prior Parameters
const typical_dwell_gamma_shape = 2.0 # s
const typical_dwell_time = 3500*10*dt # s
const dwell_gamma_shape_uniform_prior_min = 1.0
const dwell_gamma_shape_uniform_prior_max = 5.0
const dwell_gamma_mean_prior_shape = 1.0
const dwell_gamma_mean_prior_scale = typical_dwell_time #s

const typical_jump_size = 8.0e-9 # m
const std_jump_size_prior = 8.0e-9 #m

const typical_k_value = 7.5e-5 # N/m
const k_gamma_prior_shape = 1.0

const typical_diffusion_coeff = 1.0e-13 # m^2/s
const std_diffusion_coeff_prior = 100.0 * typical_diffusion_coeff #m^2/s

# Nonparametrics
const modeling_choice = "nonparametric"
const expected_n_dwells::Int64 = 10

if modeling_choice == "nonparametric"
	const dwell_collapse_required = true
elseif modeling_choice == "parametric"
	#Always choose false for parametric case
	const dwell_collapse_required = false
end

# Parameter Search step size control parameters
const conc_parameter_change_points_proposal = 1000.0 # Larger value, smaller step size
const step_size_position_sampler = 0.5e-9 #m
const step_size_diffusion_coeff_sampler = 1.0e-2 
const step_size_dwell_gamma = 5.0e-1 
const stiffness_gamma_proposal_alpha = 5.0e2 # Larger value, smaller step size
const stiffness_xy_proposal_alpha = 5.0e2 # Larger value, smaller step size

# Parameters for simulated annealing to more
# efficiently explore parameter space during
# Monte Carlo sampling. Simulated annealing smoothens
# and widens the probability distribution by
# increasing temperature in order to
# make it easy for the sampler to get out of local
# maxima and approach global maximum. Annealing is
# repeated with a frequency = save_burn_in_period
# parameter below. Temperature decays exponentially.
# Burn in period corresponds to the time during which
# sampler is approaching the maximum of the
# probability distribution and is not fully converged.
# Convegence is typically evident when the posterior
# probabilities fluctuate about a fixed value (ideally maximum).

const starting_temperature::Float64 = 20000.0 # For nonparametric
const annealing_constant::Float64 = 1000.0

if dwell_collapse_required == true
	const burn_in_period::Int64 = 11.0*annealing_constant
	const collapsed_burn_in_period::Int64 = 
			12.0*annealing_constant - burn_in_period
	const save_burn_in_period::Int64 = 
			burn_in_period+collapsed_burn_in_period
	const annealing_constant_after_collapse::Float64 = 
		annealing_constant*((save_burn_in_period-burn_in_period)/burn_in_period)
else
	const burn_in_period::Int64 = 12.0*annealing_constant
	const save_burn_in_period::Int64 = burn_in_period
end
const save_frequency::Int64 = save_burn_in_period 
const total_draws::Int64 = n_samples_to_save * save_burn_in_period

# For computing moving window acceptance ratio
const window_size = 100 
