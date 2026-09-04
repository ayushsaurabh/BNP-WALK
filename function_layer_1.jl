function get_data()

	fname = string(working_directory, file_id, ".h5")
	fid = h5open(fname, "r")
	detected_position_x = read(fid, "x(m)")
	detected_position_y = read(fid, "y(m)")
	close(fid)

	detected_position_x .= (detected_position_x .- detected_position_x[1])
	detected_position_y .= (detected_position_y .- detected_position_y[1])

	return detected_position_x[data_range], detected_position_y[data_range]
end


function initialize_parameters!(
		position_x, 
		position_y, 
		change_points_time,
		change_points_index,
		loads,
		loads_active,
		loads_inactive,
		k_xx, 
		k_yy, 	
		k_xy)

	max_n_jumps = max_n_dwells - 1
	
	for i in 1:max_n_change_points
		loads[i] = i
	end
	n_change_points =  get_active_inactive_loads!(loads,
				loads_active,
				loads_inactive)

	change_points_time[1] = 0.0
	dwell_time = ((n_time_points-1) * dt)/max_n_dwells
	for change_point in 2:max_n_change_points-1
		change_points_time[change_point] =  
				change_points_time[change_point-1] + dwell_time 
	end
	change_points_time[end] = (n_time_points-1) * dt 

	change_points_index[1] = 1
	change_points_index[end] = n_time_points
	for change_point in 2:max_n_change_points-1
		change_points_index[change_point] = 
			Integer(ceil(change_points_time[change_point]/dt))+1
	end

	k_xy .= 0.0

	for dwell in 1:max_n_dwells
		position_x[dwell] = mean(detected_position_x[change_points_index[dwell]:change_points_index[dwell+1]-1])
		position_y[dwell] = mean(detected_position_y[change_points_index[dwell]:change_points_index[dwell+1]-1])
		k_xx[dwell] = typical_k_value
		k_yy[dwell] = typical_k_value
	end

	return n_change_points
end

function sample_loads!(draw, 
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

	accepted = 0
	log_posterior = 0.0
	old_log_prior = 0.0
	proposed_log_prior = 0.0
	old_log_likelihood = 0.0
	proposed_log_likelihood = 0.0

	#if draw > save_burn_in_period
	#	@show n_change_points
	#end

	# Changepoints 1 & last one are always on
	for change_point in 2:max_n_change_points-1

		previous_active_change_point_index = 
				findlast(x-> x < change_point,view(loads_active, 1:n_change_points)) 
		next_active_change_point_index = 
				findfirst(x-> x > change_point, view(loads_active, 1:n_change_points)) 

		if loads[change_point] == 0

			time_range = change_points_index[loads_active[previous_active_change_point_index]]+
					1:change_points_index[loads_active[next_active_change_point_index]]

			dwell_index = loads_active[previous_active_change_point_index]

			old_log_likelihood = get_log_likelihood!(dwell_index,
			    			position_x[dwell_index],
			    			position_y[dwell_index],
						dynamics_matrix,
						det_stiffness,
						OU_covariance,
						inv_OU_covariance,
						det_OU_covariance,
			    			time_range)

			old_log_likelihood += logpdf(Gamma(dwell_gamma_shape,
							dwell_gamma_mean/dwell_gamma_shape), 
							change_points_time[loads_active[next_active_change_point_index]] - 
							change_points_time[loads_active[previous_active_change_point_index]])

			old_log_prior = log_prior_off 

			#Proposal
			loads[change_point] = change_point
			time_range = change_points_index[loads_active[previous_active_change_point_index]]+
					1:change_points_index[change_point]

			dwell_index = loads_active[previous_active_change_point_index]
			proposed_log_likelihood = get_log_likelihood!(dwell_index,
			    			position_x[dwell_index],
			    			position_y[dwell_index],
						dynamics_matrix,
						det_stiffness,
						OU_covariance,
						inv_OU_covariance,
						det_OU_covariance,
			    			time_range)
			proposed_log_likelihood += logpdf(Gamma(dwell_gamma_shape,
							dwell_gamma_mean/dwell_gamma_shape), 
							change_points_time[change_point] - 
							change_points_time[loads_active[previous_active_change_point_index]])

			if !(minimum(view(detected_position_x, time_range)) < position_x[dwell_index] < maximum(view(detected_position_x, time_range)))
				proposed_log_likelihood = -Inf
			end
			if !(minimum(view(detected_position_y, time_range)) < position_y[dwell_index] < maximum(view(detected_position_y, time_range)))
				proposed_log_likelihood = -Inf
			end

			time_range = change_points_index[change_point]+1:change_points_index[loads_active[next_active_change_point_index]]
			dwell_index = change_point
			proposed_log_likelihood += get_log_likelihood!(dwell_index,
			    			position_x[dwell_index],
			    			position_y[dwell_index],
						dynamics_matrix,
						det_stiffness,
						OU_covariance,
						inv_OU_covariance,
						det_OU_covariance,
			    			time_range)

			proposed_log_likelihood += logpdf(Gamma(dwell_gamma_shape,
							dwell_gamma_mean/dwell_gamma_shape), 
							change_points_time[loads_active[next_active_change_point_index]] - 
							change_points_time[change_point])

			if !(minimum(view(detected_position_x, time_range)) < position_x[dwell_index] < maximum(view(detected_position_x, time_range)))
				proposed_log_likelihood = -Inf
			end
			if !(minimum(view(detected_position_y, time_range)) < position_y[dwell_index] < maximum(view(detected_position_y, time_range)))
				proposed_log_likelihood = -Inf
			end

			proposed_log_prior = log_prior_on

			old_log_posterior = old_log_likelihood + old_log_prior
			proposed_log_posterior = proposed_log_likelihood + proposed_log_prior

			log_hastings = 1/annealing_temperature * (proposed_log_posterior - old_log_posterior)

			if log_hastings >= log(rand(rng))
				accepted += 1
			 	old_log_likelihood = proposed_log_likelihood
			else
				loads[change_point] = 0
			end


		elseif loads[change_point] != 0

			time_range = change_points_index[loads_active[previous_active_change_point_index]]+
					1:change_points_index[change_point]
			dwell_index = loads_active[previous_active_change_point_index]
			old_log_likelihood = get_log_likelihood!(dwell_index,
			    			position_x[dwell_index],
			    			position_y[dwell_index],
						dynamics_matrix,
						det_stiffness,
						OU_covariance,
						inv_OU_covariance,
						det_OU_covariance,
			    			time_range)
			old_log_likelihood += logpdf(Gamma(dwell_gamma_shape,
							dwell_gamma_mean/dwell_gamma_shape), 
							change_points_time[change_point] - 
							change_points_time[loads_active[previous_active_change_point_index]])

			time_range = change_points_index[change_point]+1:change_points_index[loads_active[next_active_change_point_index]]
			dwell_index = change_point 
			old_log_likelihood += get_log_likelihood!(dwell_index,
			    			position_x[dwell_index],
			    			position_y[dwell_index],
						dynamics_matrix,
						det_stiffness,
						OU_covariance,
						inv_OU_covariance,
						det_OU_covariance,
			    			time_range)
			old_log_likelihood += logpdf(Gamma(dwell_gamma_shape,
							dwell_gamma_mean/dwell_gamma_shape), 
							change_points_time[loads_active[next_active_change_point_index]] - 
							change_points_time[change_point])

			old_log_prior = log_prior_on 

			loads[change_point] = 0

			time_range = change_points_index[loads_active[previous_active_change_point_index]]+
					1:change_points_index[loads_active[next_active_change_point_index]]
			dwell_index = loads_active[previous_active_change_point_index]
			proposed_log_likelihood = get_log_likelihood!(dwell_index,
			    			position_x[dwell_index],
			    			position_y[dwell_index],
						dynamics_matrix,
						det_stiffness,
						OU_covariance,
						inv_OU_covariance,
						det_OU_covariance,
			    			time_range)
			proposed_log_likelihood += logpdf(Gamma(dwell_gamma_shape,
							dwell_gamma_mean/dwell_gamma_shape), 
							change_points_time[loads_active[next_active_change_point_index]] - 
							change_points_time[loads_active[previous_active_change_point_index]])

			if loads_active[next_active_change_point_index] < max_n_change_points

				previous_position_x = position_x[loads_active[previous_active_change_point_index]]
 				previous_position_y = position_y[loads_active[previous_active_change_point_index]]
				next_position_x = position_x[loads_active[next_active_change_point_index]]
 				next_position_y = position_y[loads_active[next_active_change_point_index]]

				jump_size = sqrt((next_position_x - previous_position_x)^2 + 
						      (next_position_y - previous_position_y)^2)

				if jump_size > maximum_allowed_jump_size
					proposed_log_likelihood = -Inf
				end
			end

			proposed_log_prior = log_prior_off

			old_log_posterior = old_log_likelihood + old_log_prior
			proposed_log_posterior = proposed_log_likelihood + proposed_log_prior

			log_hastings = 1/annealing_temperature * (proposed_log_posterior - old_log_posterior)

			if log_hastings >= log(rand(rng))
				accepted += 1
			 	old_log_likelihood = proposed_log_likelihood
			else
				loads[change_point] = change_point
			end
		end
		n_change_points = get_active_inactive_loads!(loads,
							loads_active,
							loads_inactive)
	
	end
	if n_change_points > 2
		return accepted/(max_n_change_points-2), n_change_points
	elseif n_change_points == 2
		return 0.0, n_change_points
	end
end

function sample_change_points_time!(draw, 
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

	accepted = 0
	log_posterior = 0.0

	for dwell in 1:max_n_dwells
		intermediate_vec1[dwell] = (change_points_time[dwell+1] - change_points_time[dwell])
	end
	intermediate_vec2 .= change_points_time[end] .* 
				rand(rng, (Dirichlet(1.0 .+ (conc_parameter_change_points_proposal .* 
						(intermediate_vec1 ./ change_points_time[end])))))

	proposed_change_points_time .= change_points_time
	proposed_change_points_index .= change_points_index

	for change_point in 2:max_n_change_points-1

		proposed_change_points_time[change_point] = 
			proposed_change_points_time[change_point-1] + intermediate_vec2[change_point-1]
		proposed_change_points_index[change_point] = 
			Int64(ceil(proposed_change_points_time[change_point]/dt))+1
	end

	valid_proposal = !any(intermediate_vec2 .<= 2.0 * dt)

	old_log_likelihood = 0.0
	proposed_log_likelihood = 0.0
	old_log_prior = 0.0
	proposed_log_prior = 0.0

	for change_point in 2:max_n_change_points

		previous_active_change_point_index = 
				findlast(x-> x < change_point, view(loads_active, 1:n_change_points)) 
		previous_active_change_point = loads_active[previous_active_change_point_index]

		if loads[change_point] != 0

			time_range = change_points_index[previous_active_change_point]+1:change_points_index[change_point]
			dwell_index = previous_active_change_point 
			old_log_likelihood += get_log_likelihood!(dwell_index,
			    			position_x[dwell_index],
			    			position_y[dwell_index],
						dynamics_matrix,
						det_stiffness,
						OU_covariance,
						inv_OU_covariance,
						det_OU_covariance,
			    			time_range)

			old_log_prior += logpdf(Gamma(dwell_gamma_shape,
							dwell_gamma_mean/dwell_gamma_shape), 
							change_points_time[change_point] - 
							       change_points_time[previous_active_change_point])

			if valid_proposal
				time_range = proposed_change_points_index[previous_active_change_point]+1:proposed_change_points_index[change_point]
				dwell_index = previous_active_change_point 
				proposed_log_likelihood += get_log_likelihood!(dwell_index,
				    			position_x[dwell_index],
				    			position_y[dwell_index],
							dynamics_matrix,
							det_stiffness,
							OU_covariance,
							inv_OU_covariance,
							det_OU_covariance,
				    			time_range)

				proposed_log_prior += logpdf(Gamma(dwell_gamma_shape,
								dwell_gamma_mean/dwell_gamma_shape), 
								proposed_change_points_time[change_point] - 
								       proposed_change_points_time[previous_active_change_point])
			end
		end
	end


	old_log_posterior::Float64 = old_log_likelihood + old_log_prior
	proposed_log_posterior::Float64 = proposed_log_likelihood + proposed_log_prior

	if !valid_proposal
	 	return 0, old_log_posterior	
	end

	log_forward_transition_probability =
	    logpdf_Dirichlet(
	        1.0 .+
	        conc_parameter_change_points_proposal .*
	        (intermediate_vec1 ./ change_points_time[end]),
	        intermediate_vec2 ./ change_points_time[end]
	    )
	
	log_backward_transition_probability =
	    logpdf_Dirichlet(
	        1.0 .+
	        conc_parameter_change_points_proposal .*
	        (intermediate_vec2 ./ change_points_time[end]),
	        intermediate_vec1 ./ change_points_time[end]
	    )

	log_hastings = 1/annealing_temperature * (proposed_log_posterior - old_log_posterior) 
			+ log_backward_transition_probability - log_forward_transition_probability


	if log_hastings >= log(rand(rng))
		accepted += 1
		change_points_time .= proposed_change_points_time
		change_points_index .= proposed_change_points_index
		log_posterior = proposed_log_posterior
	else
		log_posterior = old_log_posterior
	end

	return accepted, log_posterior
end


function sample_positions!(draw, 
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

	accepted::Integer = 0
	old_log_prior::Float64 = 0.0
	proposed_log_prior::Float64 = 0.0
	old_log_likelihood::Float64 = 0.0
	proposed_log_likelihood::Float64 = 0.0

	n_active_dwells = n_change_points - 1
	n_inactive_dwells = max_n_dwells - n_active_dwells

	# Dwell 1 is always on
	for dwell in 1:n_active_dwells

		time_range = change_points_index[loads_active[dwell]]+1:change_points_index[loads_active[dwell+1]]

		dwell_index = loads_active[dwell]
		old_position_x::Float64 = position_x[dwell_index]
		old_position_y::Float64 = position_y[dwell_index]
		old_log_likelihood = get_log_likelihood!(dwell_index,
		    			position_x[dwell_index],
		    			position_y[dwell_index],
					dynamics_matrix,
					det_stiffness,
					OU_covariance,
					inv_OU_covariance,
					det_OU_covariance,
		    			time_range)

		mean_x = mean(view(detected_position_x, time_range))
		mean_y = mean(view(detected_position_y, time_range))

		if 1 < dwell < n_active_dwells


			previous_position_x = position_x[loads_active[dwell-1]]
 			previous_position_y = position_y[loads_active[dwell-1]]
			next_position_x = position_x[loads_active[dwell+1]]
 			next_position_y = position_y[loads_active[dwell+1]]


			previous_jump_size = sqrt((old_position_x - previous_position_x)^2 + 
						      (old_position_y - previous_position_y)^2)
			next_jump_size = sqrt((next_position_x - old_position_x)^2 + 
						      (next_position_y - old_position_y)^2)

			old_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), previous_jump_size) +
						 logpdf(Normal(typical_jump_size, std_jump_size_prior), next_jump_size)

		elseif dwell == 1 && n_active_dwells > 1

			next_position_x = position_x[loads_active[dwell+1]]
 			next_position_y = position_y[loads_active[dwell+1]]

			next_jump_size = sqrt((next_position_x - old_position_x)^2 + 
						      (next_position_y - old_position_y)^2)
			old_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), next_jump_size)

			old_log_prior += logpdf(Normal(mean_first_100_positions_x, typical_jump_size/2.0), old_position_x) +
					logpdf(Normal(mean_first_100_positions_y, typical_jump_size/2.0), old_position_y)

		elseif dwell == n_active_dwells && n_active_dwells > 1

			previous_position_x = position_x[loads_active[dwell-1]]
 			previous_position_y = position_y[loads_active[dwell-1]]

			previous_jump_size = sqrt((old_position_x - previous_position_x)^2 + 
						      (old_position_y - previous_position_y)^2)

			old_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), previous_jump_size) 

			old_log_prior += logpdf(Normal(mean_last_100_positions_x, typical_jump_size/2.0), old_position_x) +
					logpdf(Normal(mean_last_100_positions_y, typical_jump_size/2.0), old_position_y)

		end
		old_log_prior += logpdf(Normal(mean_x, std_jump_size_prior/2.0), old_position_x) +
						 logpdf(Normal(mean_y, std_jump_size_prior/2.0), old_position_y)

 		proposed_position_x::Float64 = rand(rng, Normal(old_position_x, step_size_position_sampler))
 		proposed_position_y::Float64 = rand(rng, Normal(old_position_y, step_size_position_sampler))
 
		position_x[dwell_index] = proposed_position_x
		position_y[dwell_index] = proposed_position_y
		proposed_log_likelihood = get_log_likelihood!(dwell_index,
		    			position_x[dwell_index],
		    			position_y[dwell_index],
					dynamics_matrix,
					det_stiffness,
					OU_covariance,
					inv_OU_covariance,
					det_OU_covariance,
		    			time_range)

		if !(minimum(view(detected_position_x, time_range)) < proposed_position_x < maximum(view(detected_position_x, time_range)))
			proposed_log_likelihood = -Inf
		end
		if !(minimum(view(detected_position_y, time_range)) < proposed_position_y < maximum(view(detected_position_y, time_range)))
			proposed_log_likelihood = -Inf
		end

		if 1 < dwell < n_active_dwells

			previous_position_x = position_x[loads_active[dwell-1]]
 			previous_position_y = position_y[loads_active[dwell-1]]
			next_position_x = position_x[loads_active[dwell+1]]
 			next_position_y = position_y[loads_active[dwell+1]]


			previous_jump_size = sqrt((proposed_position_x - previous_position_x)^2 + 
						      (proposed_position_y - previous_position_y)^2)
			next_jump_size = sqrt((next_position_x - proposed_position_x)^2 + 
						      (next_position_y - proposed_position_y)^2)

			proposed_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), previous_jump_size) +
						 logpdf(Normal(typical_jump_size, std_jump_size_prior), next_jump_size)

		elseif dwell == 1 && n_active_dwells > 1

			next_position_x = position_x[loads_active[dwell+1]]
 			next_position_y = position_y[loads_active[dwell+1]]

			next_jump_size = sqrt((next_position_x - proposed_position_x)^2 + 
						      (next_position_y - proposed_position_y)^2)
			proposed_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), next_jump_size)

			proposed_log_prior += logpdf(Normal(mean_first_100_positions_x, typical_jump_size/2.0), proposed_position_x) +
					logpdf(Normal(mean_first_100_positions_y, typical_jump_size/2.0), proposed_position_y)

		elseif dwell == n_active_dwells && n_active_dwells > 1

			previous_position_x = position_x[loads_active[dwell-1]]
 			previous_position_y = position_y[loads_active[dwell-1]]

			previous_jump_size = sqrt((proposed_position_x - previous_position_x)^2 + 
						      (proposed_position_y - previous_position_y)^2)

			proposed_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), previous_jump_size) 

			proposed_log_prior += logpdf(Normal(mean_last_100_positions_x, typical_jump_size/2.0), proposed_position_x) +
					logpdf(Normal(mean_last_100_positions_y, typical_jump_size/2.0), proposed_position_y)

		end
		proposed_log_prior += logpdf(Normal(mean_x, std_jump_size_prior/2.0), proposed_position_x) +
						 logpdf(Normal(mean_y, std_jump_size_prior/2.0), proposed_position_y)


		old_log_posterior::Float64 = old_log_likelihood + old_log_prior
		proposed_log_posterior::Float64 = proposed_log_likelihood + proposed_log_prior

		log_hastings = (proposed_log_posterior - old_log_posterior)

		if log_hastings >= log(rand(rng))
			accepted += 1
		else
			position_x[dwell_index] = old_position_x
			position_y[dwell_index] = old_position_y
		end
	end

	if n_inactive_dwells > 0
		for dwell in 1:n_inactive_dwells

			dwell_index = loads_inactive[dwell]
			old_position_x::Float64 = position_x[dwell_index]
			old_position_y::Float64 = position_y[dwell_index]


			previous_active_dwell_index = 
				findlast(x-> x < dwell_index, view(loads_active, 1:n_change_points)) 
			next_active_dwell_index = 
				findfirst(x-> x > dwell_index, view(loads_active, 1:n_change_points)) 

			old_log_likelihood = 0.0 

			if 0 < previous_active_dwell_index < n_active_dwells && 1 < next_active_dwell_index < n_active_dwells+1

				previous_position_x = position_x[loads_active[previous_active_dwell_index]]
 				previous_position_y = position_y[loads_active[previous_active_dwell_index]]
				next_position_x = position_x[loads_active[next_active_dwell_index]]
 				next_position_y = position_y[loads_active[next_active_dwell_index]]


				previous_jump_size = sqrt((old_position_x - previous_position_x)^2 + 
							      (old_position_y - previous_position_y)^2)
			 	next_jump_size = sqrt((next_position_x - old_position_x)^2 + 
			 				      (next_position_y - old_position_y)^2)

				old_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), previous_jump_size) +
							 logpdf(Normal(typical_jump_size, std_jump_size_prior), next_jump_size)

			elseif next_active_dwell_index == n_active_dwells+1

				previous_position_x = position_x[loads_active[previous_active_dwell_index]]
				previous_position_y = position_y[loads_active[previous_active_dwell_index]]

				previous_jump_size = sqrt((old_position_x - previous_position_x)^2 + 
							      (old_position_y - previous_position_y)^2)

				old_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), previous_jump_size)

			end

			old_log_posterior::Float64 = old_log_likelihood + old_log_prior

 			proposed_position_x::Float64 = rand(rng, Normal(old_position_x, step_size_position_sampler))
 			proposed_position_y::Float64 = rand(rng, Normal(old_position_y, step_size_position_sampler))
 
			position_x[dwell_index] = proposed_position_x
			position_y[dwell_index] = proposed_position_y

			proposed_log_likelihood = 0.0 

			if 0 < previous_active_dwell_index < n_active_dwells && 1 < next_active_dwell_index < n_active_dwells+1

				previous_position_x = position_x[loads_active[previous_active_dwell_index]]
 				previous_position_y = position_y[loads_active[previous_active_dwell_index]]
				next_position_x = position_x[loads_active[next_active_dwell_index]]
 				next_position_y = position_y[loads_active[next_active_dwell_index]]


				previous_jump_size = sqrt((proposed_position_x - previous_position_x)^2 + 
							      (proposed_position_y - previous_position_y)^2)
				next_jump_size = sqrt((next_position_x - proposed_position_x)^2 + 
							      (next_position_y - proposed_position_y)^2)

				proposed_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), previous_jump_size) +
							logpdf(Normal(typical_jump_size, std_jump_size_prior), next_jump_size)

			elseif next_active_dwell_index == n_active_dwells+1

				previous_position_x = position_x[loads_active[previous_active_dwell_index]]
 				previous_position_y = position_y[loads_active[previous_active_dwell_index]]

				previous_jump_size = sqrt((proposed_position_x - previous_position_x)^2 + 
							      (proposed_position_y - previous_position_y)^2)

				proposed_log_prior = logpdf(Normal(typical_jump_size, std_jump_size_prior), previous_jump_size)

			end

			proposed_log_posterior::Float64 = proposed_log_likelihood + proposed_log_prior

			log_hastings = (proposed_log_posterior - old_log_posterior)

			if log_hastings >= log(rand(rng))
				accepted += 0
			else
				position_x[dwell_index] = old_position_x
				position_y[dwell_index] = old_position_y
			end

		end
	end

	return accepted/(n_change_points-1)
end

function sample_stiffness!(draw, 
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

	accepted::Integer = 0
	old_log_prior::Float64 = 0.0
	proposed_log_prior::Float64 = 0.0
	old_log_likelihood::Float64 = 0.0
	proposed_log_likelihood::Float64 = 0.0

	n_active_dwells = n_change_points - 1
	n_inactive_dwells = max_n_dwells - n_active_dwells

	# Dwell 1 is always on
	for dwell in 1:n_active_dwells

		time_range = change_points_index[loads_active[dwell]]+1:change_points_index[loads_active[dwell+1]]

		dwell_index = loads_active[dwell]
		old_k_xx::Float64 = k_xx[dwell_index]
		old_k_yy::Float64 = k_yy[dwell_index]
		old_k_xy::Float64 = k_xy[dwell_index]
		old_rho_xy = old_k_xy / sqrt(old_k_xx * old_k_yy)
		old_rho_xy = min(1.0 - eps(Float64),
		                  max(-1.0 + eps(Float64), old_rho_xy))

		old_log_likelihood = get_log_likelihood!(dwell_index,
		    			position_x[dwell_index],
		    			position_y[dwell_index],
					dynamics_matrix,
					det_stiffness,
					OU_covariance,
					inv_OU_covariance,
					det_OU_covariance,
		    			time_range)

		old_log_prior = logpdf(Gamma(k_gamma_prior_shape, typical_k_value/k_gamma_prior_shape), old_k_xx)
		old_log_prior += logpdf(Gamma(k_gamma_prior_shape, typical_k_value/k_gamma_prior_shape), old_k_yy)
		old_log_prior += logpdf(Uniform(-1.0, 1.0), old_rho_xy)


		proposed_k_xx::Float64 = rand(rng, Gamma(stiffness_gamma_proposal_alpha, 
							old_k_xx/stiffness_gamma_proposal_alpha))
		proposed_k_yy::Float64 = rand(rng, Gamma(stiffness_gamma_proposal_alpha, 
							old_k_yy/stiffness_gamma_proposal_alpha))

		beta = (1.0 - (old_rho_xy + 1.0)/2.0) /
		       ((old_rho_xy + 1.0)/2.0) * stiffness_xy_proposal_alpha
		proposed_rho_xy = -1.0 + 2.0 * rand(rng, Beta(stiffness_xy_proposal_alpha, beta))
		proposed_k_xy = proposed_rho_xy *sqrt(proposed_k_xx * proposed_k_yy)

		k_xx[dwell_index] = proposed_k_xx
		k_yy[dwell_index] = proposed_k_yy
		k_xy[dwell_index] = proposed_k_xy

		stiffness_matrix[dwell_index, 1, 1] = k_xx[dwell_index] 
		stiffness_matrix[dwell_index, 1, 2] = k_xy[dwell_index] 
		stiffness_matrix[dwell_index, 2, 1] = k_xy[dwell_index] 
		stiffness_matrix[dwell_index, 2, 2] = k_yy[dwell_index] 

		prepare_OU!(dwell_index, 
		    	diffusion_coeff,
			dynamics_matrix,
			stiffness_matrix,
			inv_stiffness_matrix,
			det_stiffness,
			OU_covariance,
			inv_OU_covariance,
			det_OU_covariance)

		proposed_log_likelihood = get_log_likelihood!(dwell_index,
		    			position_x[dwell_index],
		    			position_y[dwell_index],
					dynamics_matrix,
					det_stiffness,
					OU_covariance,
					inv_OU_covariance,
					det_OU_covariance,
		    			time_range)

		proposed_log_prior = logpdf(Gamma(k_gamma_prior_shape, typical_k_value/k_gamma_prior_shape), proposed_k_xx)
		proposed_log_prior += logpdf(Gamma(k_gamma_prior_shape, typical_k_value/k_gamma_prior_shape), proposed_k_yy)
		proposed_log_prior += logpdf(Uniform(-1.0, 1.0), proposed_rho_xy)

		old_log_posterior::Float64 = old_log_likelihood + old_log_prior
		proposed_log_posterior::Float64 = proposed_log_likelihood + proposed_log_prior

		log_forward_transition_probability::Float64 = 
				logpdf_Gamma(stiffness_gamma_proposal_alpha, 
					old_k_xx/stiffness_gamma_proposal_alpha, proposed_k_xx)
				logpdf_Gamma(stiffness_gamma_proposal_alpha, 
					old_k_yy/stiffness_gamma_proposal_alpha, proposed_k_yy)

		log_backward_transition_probability::Float64 = 
				logpdf_Gamma(stiffness_gamma_proposal_alpha, 
					proposed_k_xx/stiffness_gamma_proposal_alpha, old_k_xx)
				logpdf_Gamma(stiffness_gamma_proposal_alpha, 
					proposed_k_yy/stiffness_gamma_proposal_alpha, old_k_yy)

		old_u = (old_rho_xy + 1.0)/2.0
		proposed_u = (proposed_rho_xy + 1.0)/2.0
		
		log_forward_transition_probability +=
		    logpdf_Beta(stiffness_xy_proposal_alpha, beta, proposed_u)
		
		beta = (1.0 - proposed_u)/proposed_u * stiffness_xy_proposal_alpha
		
		log_backward_transition_probability +=
		    		logpdf_Beta(stiffness_xy_proposal_alpha, beta, old_u)



		log_hastings = (proposed_log_posterior - old_log_posterior) +
			log_backward_transition_probability - log_forward_transition_probability

		if log_hastings >= log(rand(rng))
			accepted += 1
		else
			k_xx[dwell_index] = old_k_xx
			k_yy[dwell_index] = old_k_yy
			k_xy[dwell_index] = old_k_xy

			stiffness_matrix[dwell_index, 1, 1] = k_xx[dwell_index] 
			stiffness_matrix[dwell_index, 1, 2] = k_xy[dwell_index] 
			stiffness_matrix[dwell_index, 2, 1] = k_xy[dwell_index] 
			stiffness_matrix[dwell_index, 2, 2] = k_yy[dwell_index] 

			prepare_OU!(dwell_index, 
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

	if n_inactive_dwells > 0
		for dwell in 1:n_inactive_dwells

			dwell_index = loads_inactive[dwell]
			old_k_xx::Float64 = k_xx[dwell_index]
			old_k_yy::Float64 = k_yy[dwell_index]
			old_k_xy::Float64 = k_xy[dwell_index]
			old_rho_xy = old_k_xy / sqrt(old_k_xx * old_k_yy)
			old_rho_xy = min(1.0 - eps(Float64),
		                  max(-1.0 + eps(Float64), old_rho_xy))

			old_log_likelihood = 0.0 
			old_log_prior = logpdf(Gamma(1.0, typical_k_value), old_k_xx)
			old_log_prior += logpdf(Gamma(1.0, typical_k_value), old_k_yy)
			old_log_prior += logpdf(Uniform(-1.0, 1.0), old_rho_xy)

			proposed_k_xx::Float64 = rand(rng, Gamma(stiffness_gamma_proposal_alpha, 
								old_k_xx/stiffness_gamma_proposal_alpha))
			proposed_k_yy::Float64 = rand(rng, Gamma(stiffness_gamma_proposal_alpha, 
								old_k_yy/stiffness_gamma_proposal_alpha))

			beta = (1.0 - (old_rho_xy + 1.0)/2.0) /
			       ((old_rho_xy + 1.0)/2.0) * stiffness_xy_proposal_alpha
			proposed_rho_xy = -1.0 + 2.0 * rand(rng, Beta(stiffness_xy_proposal_alpha, beta))
			proposed_k_xy = proposed_rho_xy *sqrt(proposed_k_xx * proposed_k_yy)

			k_xx[dwell_index] = proposed_k_xx
			k_yy[dwell_index] = proposed_k_yy
			k_xy[dwell_index] = proposed_k_xy

			stiffness_matrix[dwell_index, 1, 1] = k_xx[dwell_index] 
			stiffness_matrix[dwell_index, 1, 2] = k_xy[dwell_index] 
			stiffness_matrix[dwell_index, 2, 1] = k_xy[dwell_index] 
			stiffness_matrix[dwell_index, 2, 2] = k_yy[dwell_index] 

			prepare_OU!(dwell_index, 
			    	diffusion_coeff,
				dynamics_matrix,
				stiffness_matrix,
				inv_stiffness_matrix,
				det_stiffness,
				OU_covariance,
				inv_OU_covariance,
				det_OU_covariance)

			proposed_log_likelihood = 0.0
			proposed_log_prior = logpdf(Gamma(1.0, typical_k_value), proposed_k_xx)
			proposed_log_prior += logpdf(Gamma(1.0, typical_k_value), proposed_k_yy)
			proposed_log_prior += logpdf(Uniform(-1.0, 1.0), proposed_rho_xy)


			old_log_posterior::Float64 = old_log_likelihood + old_log_prior
			proposed_log_posterior::Float64 = proposed_log_likelihood + proposed_log_prior

			log_forward_transition_probability::Float64 = 
					logpdf_Gamma(stiffness_gamma_proposal_alpha, 
						old_k_xx/stiffness_gamma_proposal_alpha, proposed_k_xx)
					logpdf_Gamma(stiffness_gamma_proposal_alpha, 
						old_k_yy/stiffness_gamma_proposal_alpha, proposed_k_yy)

			log_backward_transition_probability::Float64 = 
					logpdf_Gamma(stiffness_gamma_proposal_alpha, 
						proposed_k_xx/stiffness_gamma_proposal_alpha, old_k_xx)
					logpdf_Gamma(stiffness_gamma_proposal_alpha, 
						proposed_k_yy/stiffness_gamma_proposal_alpha, old_k_yy)

			old_u = (old_rho_xy + 1.0)/2.0
			proposed_u = (proposed_rho_xy + 1.0)/2.0
			
			log_forward_transition_probability +=
			    logpdf_Beta(stiffness_xy_proposal_alpha, beta, proposed_u)
			
			beta = (1.0 - proposed_u)/proposed_u * stiffness_xy_proposal_alpha
			
			log_backward_transition_probability +=
			    		logpdf_Beta(stiffness_xy_proposal_alpha, beta, old_u)


			log_hastings = (proposed_log_posterior - old_log_posterior) +
						log_backward_transition_probability - log_forward_transition_probability

			if log_hastings >= log(rand(rng))
				accepted += 0
			else
				k_xx[dwell_index] = old_k_xx
				k_yy[dwell_index] = old_k_yy
				k_xy[dwell_index] = old_k_xy

				stiffness_matrix[dwell_index, 1, 1] = k_xx[dwell_index] 
				stiffness_matrix[dwell_index, 1, 2] = k_xy[dwell_index] 
				stiffness_matrix[dwell_index, 2, 1] = k_xy[dwell_index] 
				stiffness_matrix[dwell_index, 2, 2] = k_yy[dwell_index] 

				prepare_OU!(dwell_index, 
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
	end

	return accepted/(n_change_points-1)
end


function save_mcmc_data(current_draw::Int64, 
	current_sample::Int64,
	mcmc_loads::Matrix{Int64},
	mcmc_change_points_time::Matrix{Float64},
	mcmc_dwell_gamma_shape::Vector{Float64},
	mcmc_dwell_gamma_mean::Vector{Float64},
	mcmc_position_x::Matrix{Float64},
	mcmc_position_y::Matrix{Float64},
	mcmc_k_xx::Matrix{Float64},
	mcmc_k_yy::Matrix{Float64},
	mcmc_k_xy::Matrix{Float64},
	mcmc_diffusion_coeff::Vector{Float64},
	mcmc_log_posterior::Vector{Float64},
	)

	# Save the data in HDF5 format.
	file_name = string(working_directory, "mcmc_output_", file_id, ".h5")

	h5open(file_name,"w") do fid

	write_dataset(fid, "mcmc_loads",
			view(mcmc_loads, 1:current_sample, :))
	write_dataset(fid, "mcmc_change_points",
			view(mcmc_change_points_time, 1:current_sample, :))
	write_dataset(fid, "mcmc_dwell_gamma_shape",
			view(mcmc_dwell_gamma_shape, 1:current_sample))
	write_dataset(fid, "mcmc_dwell_gamma_mean",
			view(mcmc_dwell_gamma_mean, 1:current_sample))
	write_dataset(fid, "mcmc_position_x",
			view(mcmc_position_x, 1:current_sample, :))
	write_dataset(fid, "mcmc_position_y",
			view(mcmc_position_y, 1:current_sample, :))
	write_dataset(fid, "mcmc_k_xx",
			view(mcmc_k_xx, 1:current_sample, :))
	write_dataset(fid, "mcmc_k_yy",
			view(mcmc_k_yy, 1:current_sample, :))
	write_dataset(fid, "mcmc_k_xy",
			view(mcmc_k_xy, 1:current_sample, :))
	write_dataset(fid, "mcmc_diffusion_coeff",
			view(mcmc_diffusion_coeff, 1:current_sample))
	write_dataset(fid, "mcmc_log_posterior",
			view(mcmc_log_posterior, 1:current_draw))

	end	

	return nothing
end
