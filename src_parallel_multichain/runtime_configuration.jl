struct MultichainConfiguration
	input_file::String
	output_directory::String
	base_random_seed::Int64
	data_range::Union{Colon, UnitRange{Int64}}
	ground_truth_available::Bool
	kb::Float64
	temperature_kelvin::Int64
	dt::Float64
	chain_to_plot::Int64
	output_frequency::Int64
	plotting_on::Bool
	save_plots::Bool
	data_plotting_linewidth::Float64
	data_plotting_opacity::Float64
	data_plotting_sampling_frequency::Int64
	n_MCMC_chains::Int64
	n_samples_to_save::Int64
	minimum_allowed_jump_size::Float64
	maximum_allowed_jump_size::Float64
	typical_dwell_gamma_shape::Float64
	typical_dwell_time::Float64
	dwell_gamma_shape_uniform_prior_min::Float64
	dwell_gamma_shape_uniform_prior_max::Float64
	dwell_gamma_mean_prior_shape::Float64
	typical_jump_size::Float64
	std_jump_size_prior::Float64
	typical_k_value::Float64
	k_gamma_prior_shape::Float64
	typical_diffusion_coeff::Float64
	std_diffusion_coeff_prior::Float64
	modeling_choice::String
	expected_n_dwells::Int64
	conc_parameter_change_points_proposal::Float64
	step_size_position_sampler::Float64
	step_size_diffusion_coeff_sampler::Float64
	step_size_dwell_gamma::Float64
	stiffness_gamma_proposal_alpha::Float64
	stiffness_xy_proposal_alpha::Float64
	starting_temperature::Float64
	annealing_constant::Float64
	window_size::Int64
end

function _required_table(configuration::AbstractDict, name::String)
	haskey(configuration, name) || error("Missing TOML table [$name]")
	table = configuration[name]
	table isa AbstractDict || error("TOML entry '$name' must be a table")
	return table
end

function _required_value(table::AbstractDict, key::String, expected_type::Type)
	haskey(table, key) || error("Missing required TOML setting '$key'")
	value = table[key]

	if expected_type === Bool
		value isa Bool || error("TOML setting '$key' must be a Bool")
		return value
	elseif expected_type === String
		value isa AbstractString || error("TOML setting '$key' must be a String")
		return String(value)
	elseif expected_type === Int64
		value isa Integer && !(value isa Bool) || error("TOML setting '$key' must be an integer")
		return Int64(value)
	elseif expected_type === Float64
		value isa Real && !(value isa Bool) || error("TOML setting '$key' must be numeric")
		return Float64(value)
	end

	error("Unsupported configuration type $expected_type for '$key'")
end

function _resolve_path(path::String, configuration_directory::String)
	resolved = isabspath(path) ? path : joinpath(configuration_directory, path)
	return normpath(abspath(resolved))
end

function _parse_data_range(value)
	if value == "all"
		return Colon()
	end

	if value isa AbstractVector && length(value) == 2 &&
			all(item -> item isa Integer && !(item isa Bool), value)
		first_index = Int64(value[1])
		last_index = Int64(value[2])
		first_index >= 1 || error("data_range must start at index 1 or greater")
		last_index >= first_index || error("data_range end must not precede its start")
		return first_index:last_index
	end

	error("TOML setting 'data_range' must be \"all\" or a two-integer array such as [1, 1000]")
end

function load_multichain_configuration(configuration_path::AbstractString)
	absolute_configuration_path = abspath(configuration_path)
	isfile(absolute_configuration_path) || error("Configuration file does not exist: $absolute_configuration_path")
	configuration = TOML.parsefile(absolute_configuration_path)
	configuration_directory = dirname(absolute_configuration_path)

	run = _required_table(configuration, "run")
	parameters = _required_table(configuration, "parameters")

	raw_input_files = get(run, "input_files", nothing)
	raw_input_files isa AbstractVector || error("TOML setting 'input_files' must be an array of paths")
	length(raw_input_files) == 1 ||
		error("The multichain sampler requires exactly one input file per Julia instance")
	raw_input_files[1] isa AbstractString || error("The input file path must be a string")
	input_file = _resolve_path(String(raw_input_files[1]), configuration_directory)
	isfile(input_file) || error("Input file does not exist: $input_file")

	output_directory = _resolve_path(
		_required_value(run, "output_directory", String),
		configuration_directory,
	)
	mkpath(output_directory)

	modeling_choice = _required_value(parameters, "modeling_choice", String)
	modeling_choice in ("nonparametric", "parametric") ||
		error("modeling_choice must be either 'nonparametric' or 'parametric'")

	configuration_object = MultichainConfiguration(
		input_file,
		output_directory,
		_required_value(run, "base_random_seed", Int64),
		_parse_data_range(get(run, "data_range", "all")),
		_required_value(parameters, "ground_truth_available", Bool),
		_required_value(parameters, "kb", Float64),
		_required_value(parameters, "T", Int64),
		_required_value(parameters, "dt", Float64),
		_required_value(parameters, "chain_to_plot", Int64),
		_required_value(parameters, "output_frequency", Int64),
		_required_value(parameters, "plotting_on", Bool),
		_required_value(parameters, "save_plots", Bool),
		_required_value(parameters, "data_plotting_linewidth", Float64),
		_required_value(parameters, "data_plotting_opacity", Float64),
		_required_value(parameters, "data_plotting_sampling_frequency", Int64),
		_required_value(parameters, "n_MCMC_chains", Int64),
		_required_value(parameters, "n_samples_to_save", Int64),
		_required_value(parameters, "minimum_allowed_jump_size", Float64),
		_required_value(parameters, "maximum_allowed_jump_size", Float64),
		_required_value(parameters, "typical_dwell_gamma_shape", Float64),
		_required_value(parameters, "typical_dwell_time", Float64),
		_required_value(parameters, "dwell_gamma_shape_uniform_prior_min", Float64),
		_required_value(parameters, "dwell_gamma_shape_uniform_prior_max", Float64),
		_required_value(parameters, "dwell_gamma_mean_prior_shape", Float64),
		_required_value(parameters, "typical_jump_size", Float64),
		_required_value(parameters, "std_jump_size_prior", Float64),
		_required_value(parameters, "typical_k_value", Float64),
		_required_value(parameters, "k_gamma_prior_shape", Float64),
		_required_value(parameters, "typical_diffusion_coeff", Float64),
		_required_value(parameters, "std_diffusion_coeff_prior", Float64),
		modeling_choice,
		_required_value(parameters, "expected_n_dwells", Int64),
		_required_value(parameters, "conc_parameter_change_points_proposal", Float64),
		_required_value(parameters, "step_size_position_sampler", Float64),
		_required_value(parameters, "step_size_diffusion_coeff_sampler", Float64),
		_required_value(parameters, "step_size_dwell_gamma", Float64),
		_required_value(parameters, "stiffness_gamma_proposal_alpha", Float64),
		_required_value(parameters, "stiffness_xy_proposal_alpha", Float64),
		_required_value(parameters, "starting_temperature", Float64),
		_required_value(parameters, "annealing_constant", Float64),
		_required_value(parameters, "window_size", Int64),
	)

	validate_multichain_configuration(configuration_object)
	return configuration_object
end

function validate_multichain_configuration(configuration::MultichainConfiguration)
	configuration.dt > 0.0 || error("dt must be positive")
	configuration.output_frequency > 0 || error("output_frequency must be positive")
	configuration.data_plotting_sampling_frequency > 0 ||
		error("data_plotting_sampling_frequency must be positive")
	configuration.n_MCMC_chains > 0 || error("n_MCMC_chains must be positive")
	configuration.n_samples_to_save > 0 || error("n_samples_to_save must be positive")
	configuration.n_samples_to_save % configuration.n_MCMC_chains == 0 ||
		error("n_samples_to_save must be divisible by n_MCMC_chains")
	1 <= configuration.chain_to_plot <= configuration.n_MCMC_chains ||
		error("chain_to_plot must be between 1 and n_MCMC_chains")
	configuration.expected_n_dwells > 1 || error("expected_n_dwells must be greater than 1")
	configuration.typical_dwell_time > 0.0 || error("typical_dwell_time must be positive")
	configuration.annealing_constant > 0.0 || error("annealing_constant must be positive")
	configuration.window_size > 0 || error("window_size must be positive")
	configuration.minimum_allowed_jump_size <= configuration.maximum_allowed_jump_size ||
		error("minimum_allowed_jump_size must not exceed maximum_allowed_jump_size")
	configuration.dwell_gamma_shape_uniform_prior_min <=
		configuration.dwell_gamma_shape_uniform_prior_max ||
		error("dwell gamma prior minimum must not exceed its maximum")
	return nothing
end

function output_path(configuration::MultichainConfiguration)
	file_id = splitext(basename(configuration.input_file))[1]
	return joinpath(configuration.output_directory, "mcmc_output_$(file_id).h5")
end

function _define_runtime_constant(name::Symbol, value)
	isdefined(Main, name) && error("Runtime constant '$name' is already defined")
	Core.eval(Main, :(const $name = $value))
	return nothing
end

function install_runtime_parameters!(
		configuration::MultichainConfiguration,
		selected_output_file::String,
		random_seed::Int64,
	)
	file_id_value = splitext(basename(configuration.input_file))[1]
	dwell_collapse_required_value = configuration.modeling_choice == "nonparametric"
	if dwell_collapse_required_value
		burn_in_period_value = Int64(11.0 * configuration.annealing_constant)
		collapsed_burn_in_period_value = Int64(12.0 * configuration.annealing_constant) - burn_in_period_value
		save_burn_in_period_value = burn_in_period_value + collapsed_burn_in_period_value
		annealing_constant_after_collapse_value = configuration.annealing_constant *
			(save_burn_in_period_value - burn_in_period_value) / burn_in_period_value
	else
		burn_in_period_value = Int64(12.0 * configuration.annealing_constant)
		collapsed_burn_in_period_value = Int64(0)
		save_burn_in_period_value = burn_in_period_value
		annealing_constant_after_collapse_value = configuration.annealing_constant
	end

	n_samples_to_save_per_chain_value = div(
		configuration.n_samples_to_save,
		configuration.n_MCMC_chains,
	)

	values = (
		:working_directory => configuration.output_directory,
		:input_file => configuration.input_file,
		:output_file => selected_output_file,
		:random_number_generator_seed => random_seed,
		:file_id => file_id_value,
		:data_range => configuration.data_range,
		:ground_truth_available => configuration.ground_truth_available,
		:kb => configuration.kb,
		:T => configuration.temperature_kelvin,
		:dt => configuration.dt,
		:chain_to_plot => configuration.chain_to_plot,
		:output_frequency => configuration.output_frequency,
		:plotting_on => configuration.plotting_on,
		:plotting_frequency => configuration.output_frequency,
		:save_plots => configuration.save_plots,
		:data_plotting_linewidth => configuration.data_plotting_linewidth,
		:data_plotting_opacity => configuration.data_plotting_opacity,
		:data_plotting_sampling_frequency => configuration.data_plotting_sampling_frequency,
		:n_MCMC_chains => configuration.n_MCMC_chains,
		:n_samples_to_save => configuration.n_samples_to_save,
		:n_samples_to_save_per_chain => n_samples_to_save_per_chain_value,
		:minimum_allowed_jump_size => configuration.minimum_allowed_jump_size,
		:maximum_allowed_jump_size => configuration.maximum_allowed_jump_size,
		:typical_dwell_gamma_shape => configuration.typical_dwell_gamma_shape,
		:typical_dwell_time => configuration.typical_dwell_time,
		:dwell_gamma_shape_uniform_prior_min => configuration.dwell_gamma_shape_uniform_prior_min,
		:dwell_gamma_shape_uniform_prior_max => configuration.dwell_gamma_shape_uniform_prior_max,
		:dwell_gamma_mean_prior_shape => configuration.dwell_gamma_mean_prior_shape,
		:dwell_gamma_mean_prior_scale => configuration.typical_dwell_time,
		:typical_jump_size => configuration.typical_jump_size,
		:std_jump_size_prior => configuration.std_jump_size_prior,
		:typical_k_value => configuration.typical_k_value,
		:k_gamma_prior_shape => configuration.k_gamma_prior_shape,
		:typical_diffusion_coeff => configuration.typical_diffusion_coeff,
		:std_diffusion_coeff_prior => configuration.std_diffusion_coeff_prior,
		:modeling_choice => configuration.modeling_choice,
		:expected_n_dwells => configuration.expected_n_dwells,
		:dwell_collapse_required => dwell_collapse_required_value,
		:conc_parameter_change_points_proposal => configuration.conc_parameter_change_points_proposal,
		:step_size_position_sampler => configuration.step_size_position_sampler,
		:step_size_diffusion_coeff_sampler => configuration.step_size_diffusion_coeff_sampler,
		:step_size_dwell_gamma => configuration.step_size_dwell_gamma,
		:stiffness_gamma_proposal_alpha => configuration.stiffness_gamma_proposal_alpha,
		:stiffness_xy_proposal_alpha => configuration.stiffness_xy_proposal_alpha,
		:starting_temperature => configuration.starting_temperature,
		:annealing_constant => configuration.annealing_constant,
		:burn_in_period => burn_in_period_value,
		:collapsed_burn_in_period => collapsed_burn_in_period_value,
		:save_burn_in_period => save_burn_in_period_value,
		:annealing_constant_after_collapse => annealing_constant_after_collapse_value,
		:save_frequency => save_burn_in_period_value,
		:total_draws => n_samples_to_save_per_chain_value * save_burn_in_period_value,
		:window_size => configuration.window_size,
	)

	for (name, value) in values
		_define_runtime_constant(name, value)
	end

	return nothing
end
