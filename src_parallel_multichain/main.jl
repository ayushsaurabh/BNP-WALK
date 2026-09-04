using Distributed

include("input_parameters.jl")
addprocs(n_MCMC_chains)

@everywhere begin

	using Random, Distributions, SpecialFunctions, Statistics, LinearAlgebra
	using HDF5
	include("input_parameters.jl")
	rng = Xoshiro(random_number_generator_seed);
	include("function_layer_1.jl")
	include("function_layer_2.jl")

	if ground_truth_available
		fname = string(working_directory, file_id, ".h5")
		fid = h5open(fname, "r")
		gt_position_x = read(fid, "gt_x(m)")
		gt_position_y = read(fid, "gt_y(m)")
		gt_change_points_time = read(fid, "gt_change_points_time(s)")
		gt_dwell_time = read(fid, "gt_dwell_time(s)")
		gt_jump_size = read(fid, "gt_jump_size(m)")
		close(fid)
	end

	const detected_position_x, detected_position_y = get_data();
	const n_time_points = size(detected_position_x)[1]
	const min_detected_position_x = minimum(detected_position_x)
	const min_detected_position_y = minimum(detected_position_y)
	const max_detected_position_x = maximum(detected_position_x)
	const max_detected_position_y = maximum(detected_position_y)
	const mean_first_100_positions_x = mean(detected_position_x[1:100])
	const mean_first_100_positions_y = mean(detected_position_y[1:100])
	const mean_last_100_positions_x = mean(detected_position_x[end-99:end])
	const mean_last_100_positions_y = mean(detected_position_y[end-99:end])
	
	
	# Parameter for Beta-Bernoulli Nonparametrics
	if modeling_choice == "nonparametric"
		const max_n_dwells::Int64 = 5 * ceil(n_time_points * dt /typical_dwell_time)
		const expected_n_change_points::Int64 = expected_n_dwells + 1
		const max_n_change_points::Int64 = max_n_dwells + 1
	elseif modeling_choice == "parametric"
		const max_n_dwells::Int64 = expected_n_dwells
		const expected_n_change_points::Int64 = max_n_dwells + 1
		const max_n_change_points::Int64 = expected_n_change_points 
	end
	
	const prior_on::Float64 =
			1.0/(1.0 + ((max_n_change_points-2-1)/
				(expected_n_change_points-2)))
	const log_prior_on::Float64 = log(prior_on)
	const log_prior_off::Float64 = log(1.0 - prior_on)
end
include("function_layer_3.jl")

function sampler()

	draw = 1

	# Print some of the parameters
	println("*********************************************************")
	@show file_id
	@show modeling_choice
	@show n_time_points
	@show max_n_dwells
	@show log_prior_on
	@show log_prior_off
	println("*********************************************************")
	flush(stdout);

	mcmc_loads = zeros(Int64, n_samples_to_save, max_n_change_points)
	mcmc_position_x = zeros(n_samples_to_save, max_n_dwells)
	mcmc_position_y = zeros(n_samples_to_save, max_n_dwells)
	mcmc_change_points_time = zeros(n_samples_to_save, max_n_change_points)
	mcmc_k_xx = zeros(n_samples_to_save, max_n_dwells)
	mcmc_k_yy = zeros(n_samples_to_save, max_n_dwells)
	mcmc_k_xy = zeros(n_samples_to_save, max_n_dwells)
	mcmc_log_posterior = zeros(total_draws)
	mcmc_diffusion_coeff = zeros(n_samples_to_save)
	mcmc_dwell_gamma_shape = zeros(n_samples_to_save)
	mcmc_dwell_gamma_mean = zeros(n_samples_to_save)
	mcmc_dwell_gamma_shape[1] = typical_dwell_gamma_shape  # For plotting purposes only



	@everywhere workers() begin

		diffusion_coeff = typical_diffusion_coeff
		dwell_gamma_shape = typical_dwell_gamma_shape
		dwell_gamma_mean = typical_dwell_time

		# Parameters to be sampled
		position_x = zeros(max_n_dwells)
		position_y = zeros(max_n_dwells)

		jump_sizes = zeros(max_n_dwells-1)

		k_xx = zeros(max_n_dwells)
		k_yy = zeros(max_n_dwells)
		k_xy = zeros(max_n_dwells)


 		change_points_time = zeros(max_n_change_points)
		change_points_index = zeros(Integer, max_n_change_points)

		loads = zeros(Int64, max_n_change_points)
		loads_active = zeros(Int64, max_n_change_points)
		loads_inactive = zeros(Int64, max_n_change_points)

		n_change_points = initialize_parameters!(position_x, 
				position_y, 
				change_points_time,
				change_points_index,
				loads,
				loads_active,
				loads_inactive,
				k_xx, 
				k_yy, 	
				k_xy)

		# Intermediate_variables for calculation purposes only. Not required to be saved or monitored
		intermediate_vec1 = zeros(max_n_dwells)
		intermediate_vec2 = zeros(max_n_dwells)
 		proposed_change_points_time = zeros(max_n_change_points)
		proposed_change_points_index = zeros(Integer, max_n_change_points)

		dynamics_matrix = zeros(max_n_dwells, 2, 2)
		stiffness_matrix = zeros(max_n_dwells, 2, 2)
		inv_stiffness_matrix = zeros(max_n_dwells, 2, 2)
		det_stiffness = zeros(max_n_dwells)
		OU_covariance = zeros(max_n_dwells, 2, 2)
		inv_OU_covariance = zeros(max_n_dwells, 2, 2)
		det_OU_covariance = zeros(max_n_dwells)
		for dwell in 1:max_n_change_points-1

			stiffness_matrix[dwell, 1, 1] = k_xx[dwell] 
			stiffness_matrix[dwell, 1, 2] = k_xy[dwell] 
			stiffness_matrix[dwell, 2, 1] = k_xy[dwell] 
			stiffness_matrix[dwell, 2, 2] = k_yy[dwell] 

			prepare_OU!(dwell, 
			    	diffusion_coeff,
				dynamics_matrix,
				stiffness_matrix,
				inv_stiffness_matrix,
				det_stiffness,
				OU_covariance,
				inv_OU_covariance,
				det_OU_covariance)

		end

	end

	plot_mcmc_log_posterior = zeros(total_draws) # For plotting purposes only
	plot_mcmc_dwell_gamma_shape = zeros(n_samples_to_save) # For plotting purposes only

	plot_gamma_shape = zeros(total_draws) # For plotting purposes only
	plot_diffusion_coeff = zeros(total_draws) # For plotting purposes only

	println(" Starting Sampler...")
	flush(stdout);


	# Parameters to save MCMC acceptance ratios
	acceptance_moving_window_change_points_time = zeros(Float64, window_size)
	acceptance_moving_window_positions = zeros(Float64, window_size)
	acceptance_moving_window_diffusion_coeff = zeros(Float64, window_size)
	acceptance_moving_window_gamma = zeros(Float64, window_size)
	acceptance_moving_window_k = zeros(Float64, window_size)
	acceptance_moving_window_loads = zeros(Float64, window_size)

	accepted_change_points_time = 0 
	accepted_positions = 0 
	accepted_diffusion_coeff = 0
	accepted_gamma = 0 
	accepted_k = 0 
	accepted_loads = 0

	println("Loads on trace ", chain_to_plot, " = ",  @fetchfrom chain_to_plot+1 loads)
	println("Loads active on trace ", chain_to_plot," = ",  @fetchfrom chain_to_plot+1 loads_active)

	plot_position_x = Float64.(@fetchfrom chain_to_plot+1 position_x)
	plot_position_y = Float64.(@fetchfrom chain_to_plot+1 position_y)
	plot_change_points_time = Float64.(@fetchfrom chain_to_plot+1 change_points_time)
	plot_loads_active = Int64.(@fetchfrom chain_to_plot+1 loads_active)
	plot_mcmc_log_posterior = mcmc_log_posterior
	plot_mcmc_dwell_gamma_shape = mcmc_dwell_gamma_shape

	plot_gamma_shape[draw] = @fetchfrom chain_to_plot+1 dwell_gamma_shape
	plot_diffusion_coeff[draw] = @fetchfrom chain_to_plot+1 diffusion_coeff


	if plotting_on == true
  		plot_stuff(1, 
			plot_position_x, 
			plot_position_y, 
			plot_change_points_time, 
			plot_loads_active,
			plot_gamma_shape,
			plot_diffusion_coeff,
			plot_mcmc_dwell_gamma_shape,
			plot_mcmc_log_posterior)
	end

	current_sample_chain::Int64 = 0
	for draw in 1:total_draws

		if draw % output_frequency == 0
			println("")
			println("**************************************************************************************************************************************************")
			@show draw
			println("**************************************************************************************************************************************************")

		end

		annealing_temperature = 1.0 + (starting_temperature - 1.0) * 
			exp(-((draw - 1) % save_burn_in_period)/
			    	annealing_constant)

		@everywhere workers() draw = $draw
		@everywhere workers() annealing_temperature = $annealing_temperature


		@everywhere workers() begin 

			n_change_points =  get_active_inactive_loads!(loads,
								loads_active,
								loads_inactive)

			accepted_positions = sample_positions!(draw, 
							position_x, 
							position_y, 
							change_points_time, 
							change_points_index,
							loads,
							loads_active,
							loads_inactive, 
							n_change_points,
							diffusion_coeff,
							k_xx, 
							k_yy, 	
							k_xy,
							dynamics_matrix,
							stiffness_matrix,
							inv_stiffness_matrix,
							det_stiffness,
							OU_covariance,
							inv_OU_covariance,
							det_OU_covariance,
							annealing_temperature)
	
	 		accepted_k = sample_stiffness!(draw, 
	 						position_x, 
	 						position_y, 
	 						change_points_time, 
	 						change_points_index,
	 						loads,
	 						loads_active,
	 						loads_inactive, 
	 						n_change_points,
	 						diffusion_coeff,
	 						k_xx, 
	 						k_yy, 	
	 						k_xy,
	 						dynamics_matrix,
	 						stiffness_matrix,
	 						inv_stiffness_matrix,
							det_stiffness,
	 						OU_covariance,
	 						inv_OU_covariance,
							det_OU_covariance,
	 						annealing_temperature)

			diffusion_coeff, accepted_diffusion_coeff = sample_diffusion_coeff(draw, 
 							position_x, 
 							position_y, 
 							change_points_time, 
 							change_points_index,
 							loads,
 							loads_active,
 							loads_inactive, 
 							n_change_points,
 							diffusion_coeff,
 							k_xx, 
 							k_yy, 	
 							k_xy,
 							dynamics_matrix,
 							stiffness_matrix,
 							inv_stiffness_matrix,
							det_stiffness,
 							OU_covariance,
 							inv_OU_covariance,
							det_OU_covariance,
 							annealing_temperature)

 			dwell_gamma_shape, dwell_gamma_mean, accepted_gamma = 
					sample_dwell_gamma(draw, 
 							change_points_time, 
 							change_points_index,
							dwell_gamma_shape,
							dwell_gamma_mean,
 							loads,
 							loads_active,
 							loads_inactive, 
 							n_change_points,
 							annealing_temperature)


			accepted_change_points_time, log_posterior =  
						sample_change_points_time!(draw, 
								position_x, 
								position_y, 
								change_points_time, 
								change_points_index,
								dwell_gamma_shape,
								dwell_gamma_mean,
								loads,
								loads_active,
								loads_inactive, 
								n_change_points,
								diffusion_coeff,
								k_xx, 
								k_yy, 	
								k_xy,
								dynamics_matrix,
								stiffness_matrix,
								inv_stiffness_matrix,
								det_stiffness,
								OU_covariance,
								inv_OU_covariance,
								det_OU_covariance,
								intermediate_vec1,
								intermediate_vec2,
								proposed_change_points_time,
								proposed_change_points_index,
								annealing_temperature)

			if (draw-1) % save_burn_in_period <= burn_in_period && modeling_choice == "nonparametric"

				accepted_loads, n_change_points = sample_loads!(draw, 
							position_x, 
							position_y, 
							change_points_time, 
							change_points_index,
							dwell_gamma_shape,
							dwell_gamma_mean,
							loads,
							loads_active,
							loads_inactive, 
							n_change_points,
							diffusion_coeff,
							k_xx, 
							k_yy, 	
							k_xy,
							dynamics_matrix,
							stiffness_matrix,
							inv_stiffness_matrix,
							det_stiffness,
							OU_covariance,
							inv_OU_covariance,
							det_OU_covariance,
							annealing_temperature)
			end
			if dwell_collapse_required == true
				if ((draw-1) % save_burn_in_period == burn_in_period) || 
					((draw-1) % save_burn_in_period == (burn_in_period + 
						Int64(round((save_burn_in_period - burn_in_period)/2))))


					n_change_points = collapse_trajectory!(position_x, 
									position_y, 
									change_points_time, 
									change_points_index,
									loads,
									loads_active,
									loads_inactive, 
									n_change_points)

				end
			end
		end


		if draw <= window_size
			acceptance_moving_window_change_points_time[draw] = accepted_change_points_time
			acceptance_ratio_change_points_time = sum(acceptance_moving_window_change_points_time)/draw

			acceptance_moving_window_positions[draw] = accepted_positions
			acceptance_ratio_positions = sum(acceptance_moving_window_positions)/draw

			acceptance_moving_window_diffusion_coeff[draw] = @fetchfrom chain_to_plot+1 accepted_diffusion_coeff
			acceptance_ratio_diffusion_coeff = sum(acceptance_moving_window_diffusion_coeff)/draw

			acceptance_moving_window_gamma[draw] = @fetchfrom chain_to_plot+1 accepted_gamma
			acceptance_ratio_gamma = sum(acceptance_moving_window_gamma)/draw

			acceptance_moving_window_k[draw] = accepted_k
			acceptance_ratio_k = sum(acceptance_moving_window_k)/draw

			acceptance_moving_window_loads[draw] = accepted_loads
			acceptance_ratio_loads = sum(acceptance_moving_window_loads)/draw

		else
			circshift!(acceptance_moving_window_change_points_time, -1)
			acceptance_moving_window_change_points_time[end] = accepted_change_points_time
			acceptance_ratio_change_points_time = sum(acceptance_moving_window_change_points_time)/window_size

			circshift!(acceptance_moving_window_positions, -1)
			acceptance_moving_window_positions[end] = accepted_positions
			acceptance_ratio_positions = sum(acceptance_moving_window_positions)/window_size

			circshift!(acceptance_moving_window_diffusion_coeff, -1)
			acceptance_moving_window_diffusion_coeff[end] = accepted_diffusion_coeff
			acceptance_ratio_diffusion_coeff = sum(acceptance_moving_window_diffusion_coeff)/window_size

			circshift!(acceptance_moving_window_gamma, -1)
			acceptance_moving_window_gamma[end] = accepted_gamma
			acceptance_ratio_gamma = sum(acceptance_moving_window_gamma)/window_size

			circshift!(acceptance_moving_window_k, -1)
			acceptance_moving_window_k[end] = accepted_k
			acceptance_ratio_k = sum(acceptance_moving_window_k)/window_size

			circshift!(acceptance_moving_window_loads, -1)
			acceptance_moving_window_loads[end] = accepted_loads
			acceptance_ratio_loads = sum(acceptance_moving_window_loads)/window_size
		end


		mcmc_log_posterior[draw] = @fetchfrom chain_to_plot+1 log_posterior
		plot_gamma_shape[draw] = @fetchfrom chain_to_plot+1 dwell_gamma_shape
		plot_diffusion_coeff[draw] = @fetchfrom chain_to_plot+1 diffusion_coeff

		if draw % output_frequency == 0

			@show annealing_temperature
			#@show position_x[loads_active[1:n_change_points-1]]
			#@show position_y[loads_active[1:n_change_points-1]]
			#jump_sizes .= 0.0
			#for jump in 1:n_change_points-2
			#	jump_sizes[jump] = sqrt((position_x[loads_active[jump+1]] - position_x[loads_active[jump]])^2 
			#					+(position_y[loads_active[jump+1]] - position_y[loads_active[jump]])^2)
			#end
			#jump_sizes .= jump_sizes/(1.0e-9)
			#@show jump_sizes

			#@show n_change_points
			#@show loads_active[1:n_change_points]
			#@show change_points_time[loads_active[1:n_change_points]]
			#@show change_points_index[loads_active[1:n_change_points]]

			

			#@show k_xx[loads_active[1:n_change_points-1]]
			#@show k_yy[loads_active[1:n_change_points-1]]
			#@show k_xy[loads_active[1:n_change_points-1]]

			#@show acceptance_ratio_change_points_time 
			@show acceptance_ratio_gamma 
			#@show acceptance_ratio_positions 
			@show acceptance_ratio_diffusion_coeff 
			#@show acceptance_ratio_k 
			#@show acceptance_ratio_loads
			#@show log_posterior
			println("**************************************************************************************************************************************************")
			println("")

			println("Loads on trace ", chain_to_plot, " = ",  @fetchfrom chain_to_plot+1 loads)
			println("Loads active on trace ", chain_to_plot," = ",  @fetchfrom chain_to_plot+1 loads_active)

			plot_position_x .= Float64.(@fetchfrom chain_to_plot+1 position_x)
			plot_position_y .= Float64.(@fetchfrom chain_to_plot+1 position_y)
			plot_change_points_time .= Float64.(@fetchfrom chain_to_plot+1 change_points_time)
			plot_loads_active .= Int64.(@fetchfrom chain_to_plot+1 loads_active)
			plot_mcmc_log_posterior .= mcmc_log_posterior
			plot_mcmc_dwell_gamma_shape .= mcmc_dwell_gamma_shape

			@show plot_diffusion_coeff[draw]
			@show plot_gamma_shape[draw]

			if plotting_on == true
	  			plot_stuff(draw, 
					plot_position_x, 
					plot_position_y, 
					plot_change_points_time, 
					plot_loads_active,
					plot_gamma_shape,
					plot_diffusion_coeff,
					plot_mcmc_dwell_gamma_shape,
					plot_mcmc_log_posterior)
			end

			flush(stdout);
		end


 		# Save Data 
		if draw % save_frequency == 0

			current_sample_chain += 1

			for worker in workers()
 
				current_chain = worker - 1
				current_sample = (current_sample_chain - 1) * n_MCMC_chains + current_chain 

  				mcmc_loads[current_sample, 1:max_n_change_points] .= @fetchfrom worker loads[:]
				mcmc_change_points_time[current_sample, 1:max_n_change_points] .= @fetchfrom worker change_points_time[:]
  				mcmc_position_x[current_sample, 1:max_n_dwells] .= @fetchfrom worker position_x[:]
  				mcmc_position_y[current_sample, 1:max_n_dwells] .= @fetchfrom worker position_y[:]
  				mcmc_k_xx[current_sample, 1:max_n_dwells] .= @fetchfrom worker k_xx[:]
  				mcmc_k_yy[current_sample, 1:max_n_dwells] .= @fetchfrom worker k_yy[:]
  				mcmc_k_xy[current_sample, 1:max_n_dwells] .= @fetchfrom worker k_xy[:]
 				mcmc_diffusion_coeff[current_sample] = @fetchfrom worker diffusion_coeff 
				mcmc_dwell_gamma_mean[current_sample] = @fetchfrom worker dwell_gamma_mean 
				mcmc_dwell_gamma_shape[current_sample] = @fetchfrom worker dwell_gamma_shape 
			end

      			save_mcmc_data(draw, 
        			current_sample_chain * n_MCMC_chains, 
				mcmc_loads,
				mcmc_change_points_time,
				mcmc_dwell_gamma_shape,
				mcmc_dwell_gamma_mean,
				mcmc_position_x,
				mcmc_position_y,
				mcmc_k_xx,
				mcmc_k_yy,
				mcmc_k_xy,
				mcmc_diffusion_coeff,
				mcmc_log_posterior)


			@everywhere workers() begin
				n_change_points = initialize_parameters!(position_x, 
						position_y, 
						change_points_time,
						change_points_index,
						loads,
						loads_active,
						loads_inactive,
						k_xx, 
						k_yy, 	
						k_xy)

				for dwell in 1:max_n_change_points-1

					stiffness_matrix[dwell, 1, 1] = k_xx[dwell] 
					stiffness_matrix[dwell, 1, 2] = k_xy[dwell] 
					stiffness_matrix[dwell, 2, 1] = k_xy[dwell] 
					stiffness_matrix[dwell, 2, 2] = k_yy[dwell] 

					prepare_OU!(dwell, 
					    	diffusion_coeff,
						dynamics_matrix,
						stiffness_matrix,
						inv_stiffness_matrix,
						det_stiffness,
						OU_covariance,
						inv_OU_covariance,
						det_OU_covariance)
		
				end
			end


			println("")
			println("Saved Samples upto No. ", current_sample_chain * n_MCMC_chains)
			println("")
			flush(stdout);

		end

	end

	return nothing
end

sampler()
rmprocs()
