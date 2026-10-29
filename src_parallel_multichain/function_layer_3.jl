function get_data()
	detected_position_x, detected_position_y = h5open(input_file, "r") do fid
		if haskey(fid, "analysis") && haskey(fid["analysis"], "xy")
			xy = read(fid["analysis/xy"])
			return xy[1, 1, :], xy[1, 2, :]
		elseif haskey(fid, "x(m)") && haskey(fid, "y(m)")
			# Retain compatibility with trajectories written by the original code.
			return read(fid, "x(m)"), read(fid, "y(m)")
		end

		error(
			"Input file '$input_file' must contain either analysis/xy " *
			"or the x(m) and y(m) datasets",
		)
	end

	# Select the requested samples before centering so excluded non-finite values,
	# including a non-finite first sample in the file, cannot contaminate the range.
	detected_position_x = detected_position_x[data_range]
	detected_position_y = detected_position_y[data_range]
	detected_position_x .-= detected_position_x[1]
	detected_position_y .-= detected_position_y[1]

	return detected_position_x, detected_position_y
end


if plotting_on == true

	using Plots, Plots.Measures
	function plot_stuff(current_draw,
				position_x, 
				position_y, 
				change_points_time, 
				loads_active,
				gamma_shape,
				diffusion_coeff,
				mcmc_dwell_gamma_shape,
				mcmc_log_posterior)

		n_change_points = count(x -> x != 0, loads_active) 

		if ground_truth_available
			plot(detected_position_x[1:data_plotting_sampling_frequency:end] .* 1.0e9, 
			     detected_position_y[1:data_plotting_sampling_frequency:end] .* 1.0e9, 
			     linewidth = data_plotting_linewidth, alpha = data_plotting_opacity)

			scatter!(gt_position_x .* 1.0e9, 
					gt_position_y .* 1.0e9, 
            				color=:black,
					markersize=6)

			xy_plot = scatter!(position_x[loads_active[1:n_change_points-1]] .* 1.0e9, 
					position_y[loads_active[1:n_change_points-1]] .* 1.0e9, 
            				xlabel = "x (nm)", 
					ylabel = "y (nm)", 
					legend = false, 
					color=:orange, 
					tickfontsize=16, 
					guidefontsize=20)

			scatter(gt_change_points_time, 
					legend = false, 
					color=:black,
					markersize=6)
			plot_change_points = scatter!(change_points_time[loads_active[1:n_change_points]], 
					ylabel = "Time (s)", 
					xlabel = "Change Point", 
					legend = false, 
					color=:orange, 
					tickfontsize=16, 
					guidefontsize=20)
		else
			plot(detected_position_x[1:data_plotting_sampling_frequency:end] .* 1.0e9, 
			     detected_position_y[1:data_plotting_sampling_frequency:end] .* 1.0e9, 
			     linewidth = data_plotting_linewidth, alpha = data_plotting_opacity)

			xy_plot = scatter!(position_x[loads_active[1:n_change_points-1]] .* 1.0e9, 
					position_y[loads_active[1:n_change_points-1]] .* 1.0e9, 
            				xlabel = "x (nm)", 
					ylabel = "y (nm)", 
					legend = false, 
					color=:orange, 
					tickfontsize=16, 
					guidefontsize=20)

			plot_change_points = scatter(change_points_time[loads_active[1:n_change_points]], 
					ylabel = "Time (s)", 
					xlabel = "Change Point", 
					legend = false, 
					color=:orange, 
					tickfontsize=16, 
					guidefontsize=20)

		end

		plot_gamma = plot(collect(1:current_draw), 
				gamma_shape[1:current_draw],
			      	xlabel = "Draws", 
				ylabel="Gamma Shape Parameter",
				legend=false, 
				tickfontsize=16, 
				guidefontsize=20)

		plot_diffusion = plot(collect(1:current_draw), 
				diffusion_coeff[1:current_draw],
			      	xlabel = "Draws", 
				ylabel="Diffusion Coefficient (m^2/s)",
				legend=false, 
				tickfontsize=16, 
				guidefontsize=20)

		bin_edges = (dwell_gamma_shape_uniform_prior_min-0.5):1:(dwell_gamma_shape_uniform_prior_max+0.5)
		plot_mcmc_dwell_gamma_shape = histogram( 
				mcmc_dwell_gamma_shape[mcmc_dwell_gamma_shape .!= 0],
				bins=bin_edges,
				xlims=(dwell_gamma_shape_uniform_prior_min-0.5, dwell_gamma_shape_uniform_prior_max+0.5),
				xlabel="Gamma Shape Parameter",
			      	ylabel = "MCMC Sample Counts", 
				legend=false, 
				tickfontsize=16, 
				guidefontsize=20)


		plot_posterior = plot(collect(1:current_draw), 
				mcmc_log_posterior[1:current_draw],
			      	xlabel = "Draws", 
				ylabel="log(posterior)",
				legend=false, 
				tickfontsize=16, 
				guidefontsize=20)
			
		final_plot = plot(xy_plot, 
				plot_change_points, 
				plot_mcmc_dwell_gamma_shape,
				plot_gamma,
				plot_diffusion,
				plot_posterior, 
				layout = (3, 2), 
				size=(3000, 1500), 
				leftmargin = 25mm, 
				bottommargin=15mm,
				rightmargin=10mm)
		display(final_plot)
		if save_plots == true && current_draw % save_frequency == 0
			savefig(final_plot, joinpath(working_directory, string(file_id, "_", current_draw, ".png")))
		end

		return nothing
	end
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
	h5open(output_file,"w") do fid

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
