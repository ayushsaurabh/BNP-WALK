using CairoMakie, KernelDensity
using HDF5
using Distributions
using Statistics

include("input_parameters.jl")
current_directory =  "/home/mbgnjasb/codes/chu_lab/inverse_code/no_straight_line_assumption_add_anisotropy_nonparametrics_FBM/experimental_data/dynein/testing_2/"

fname = string(current_directory, file_id, ".h5")
fid = h5open(fname, "r")
det_position_x = read(fid, "x(m)")
det_position_y = read(fid, "y(m)")
close(fid)

detected_position_x = (det_position_x .- det_position_x[1])[data_range]
detected_position_y = (det_position_y .- det_position_y[1])[data_range]
n_time_points = size(detected_position_x)[1]

if ground_truth_available
	fname = string(current_directory, file_id, ".h5")
	fid = h5open(fname, "r")
	gt_position_x = read(fid, "gt_x(m)")
	gt_position_y = read(fid, "gt_y(m)")
	gt_change_points_time = read(fid, "gt_change_points_time(s)")
	gt_dwell_time = read(fid, "gt_dwell_time(s)")
	gt_jump_size = read(fid, "gt_jump_size(m)")
	close(fid)
end

function get_active_inactive_loads!(loads, 
			loads_active,
			loads_inactive)

	n_change_points = count(x -> x != 0, loads) 
	n_inactive_change_points::Int64 = size(loads)[1] - n_change_points

	loads_active .= 0
	loads_inactive .= 0

	loads_active[1:n_change_points] .= findall(x -> x != 0, loads)
	loads_inactive[1:n_inactive_change_points] .= findall(x -> x == 0, loads)

	return n_change_points 
end

fname = string(current_directory, "mcmc_output_", file_id, ".h5")
fid = h5open(fname, "r")
mcmc_loads = read(fid, "mcmc_loads")
mcmc_change_points_time = read(fid, "mcmc_change_points",)
mcmc_dwell_gamma_shape = 	read(fid, "mcmc_dwell_gamma_shape")
mcmc_dwell_gamma_mean = 	read(fid, "mcmc_dwell_gamma_mean")
mcmc_position_x = 	read(fid, "mcmc_position_x")
mcmc_position_y = 	read(fid, "mcmc_position_y")
mcmc_k_xx = 	read(fid, "mcmc_k_xx")
mcmc_k_yy = 	read(fid, "mcmc_k_yy")
mcmc_k_xy = 	read(fid, "mcmc_k_xy")
mcmc_diffusion_coeff = read(fid, "mcmc_diffusion_coeff")
mcmc_log_posterior = 	read(fid, "mcmc_log_posterior")
close(fid)

n_samples = size(mcmc_loads)[1]

change_points_time = []
position_x = []
position_y = []
k_xx = []
k_yy = []
k_xy = []
dwells = []
jump_sizes = []

loads=zeros(Int64, size(mcmc_loads)[2])
loads_active=zeros(Int64, size(mcmc_loads)[2])
loads_inactive=zeros(Int64, size(mcmc_loads)[2])

for sample in 1:n_samples
    loads .= mcmc_loads[sample, :]
    n_change_points = get_active_inactive_loads!(loads, loads_active, loads_inactive)

    for change_point in 1:n_change_points
		load = loads_active[change_point]
        push!(change_points_time, mcmc_change_points_time[sample, load])
		if change_point < n_change_points
        	push!(dwells, mcmc_change_points_time[sample, loads_active[change_point+1]] - 
								mcmc_change_points_time[sample, load])
        	push!(position_x, mcmc_position_x[sample, load])
			push!(position_y, mcmc_position_y[sample, load])
			push!(k_xx, mcmc_k_xx[sample, load])
			push!(k_yy, mcmc_k_yy[sample, load])
			push!(k_xy, mcmc_k_xy[sample, load])

			if change_point < n_change_points -1
				push!(jump_sizes, 
					sqrt((mcmc_position_x[sample, load] - 
						mcmc_position_x[sample, loads_active[change_point+1]])^2 +
						(mcmc_position_y[sample, load] - 
						mcmc_position_y[sample, loads_active[change_point+1]])^2))
			end
		end
    end
end

# Fit a Gamma distribution by maximum likelihood
dist = fit(Gamma, Float64.(dwells))
dwell_fitted_gamma_shape = dist.α
dwell_fitted_gamma_scale = dist.θ #seconds
dwell_grid = collect(0.0:n_time_points * dt/1000:n_time_points * dt)
gamma_pdf = pdf.(Gamma.(dwell_fitted_gamma_shape .+ 0.4, dwell_fitted_gamma_scale .- 0.002), dwell_grid)

#gamma_prior_pdf = pdf.(Gamma.(change_point_gamma_prior_shape, 
#					typical_dwell_time/change_point_gamma_prior_shape), dwell_grid)


begin

	upper_lim_heatmap_color = 0.00045
	position_sampling_frequency = 20
	alpha_data_plot = 1.0
	linewidth_data_plot = 0.1
	n_bins_change_point_hist = 100 

	n_bins_dwell_time_hist = 200
	upper_lim_dwell_time_hist = 0.05#(n_time_points -1) * dt
	text_location_x, text_location_y = 0.025, 120

	n_bins_jump_size_hist =200
	upper_lim_jump_size_hist = 50 #nm 

	n_bins_gamma_shape_hist = 10
	upper_lim_gamma_shape_hist = 5.0

	n_bins_gamma_mean_hist = 40
	upper_lim_gamma_mean_hist = upper_lim_dwell_time_hist


	kde_points = (2048, 2048)
	kde_bw = (2.0, 2.0)

	den = kde((position_x .* 1.0e9, position_y .* 1.0e9), npoints = kde_points, bandwidth = kde_bw)
	normalized_density = den.density ./ sum(den.density)

	min_position_x = den.x[1] 
	max_position_x = den.x[end] 
	min_position_y = den.y[1] 
	max_position_y = den.y[end] 

	plot_aspect_ratio = abs((max_position_x - min_position_x) / (max_position_y - min_position_y))

	fig = Figure(; size=(3000, 1300), font="CMU Serif", fontsize = 24, figure_padding= 20)

	# Plot 1
	axis = Axis(fig[1, 1:5]; xlabel = L"x \, (nm)", ylabel = L"y \, (nm)",
				 xlabelsize = 30, ylabelsize = 30, aspect = DataAspect(),
				 title=rich("Bayesian inference of ", 
				 	rich("Dynein ", color=:red),"trajectory using ",  
					rich("200", color=:red)," MCMC samples"), titlesize=34)#0,
				 #xticks = [0, 50, 100, 150, 200, 250, 300])
				 #yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
	limits!(axis, min_position_x, max_position_x, min_position_y, max_position_y)
	heatmap!(den.x, den.y, den.density, colorrange=(0.0, upper_lim_heatmap_color))
	lines!(detected_position_x[1:position_sampling_frequency:end] .* 1.0e9, 
			detected_position_y[1:position_sampling_frequency:end] .* 1.0e9, 
			linewidth=linewidth_data_plot, alpha = alpha_data_plot, color=:white)
	if ground_truth_available
		scatter!(gt_position_x .* 1.0e9, 
			gt_position_y .* 1.0e9, 
			color=:red, markersize=20)
	end

	legend_line = LineElement(color = :white, linewidth = 0.5)
	if ground_truth_available
		dot_element = [MarkerElement(marker = :circle, color = :red, markersize = 20)]
		axislegend(axis, [legend_line, dot_element], ["Data", "Ground Truth Dwell Positions"],
						position=:lt, framevisible=false, labelcolor=:white)
	else
		axislegend(axis, [legend_line], ["Data"],
						position=:lt, framevisible=false, labelcolor=:white)
	end

	# Plot 2
	axis = Axis(fig[2, 1]; xlabel = L"Change \,\, Point \, (s)", ylabel = L"Counts",
				 xlabelsize = 30, ylabelsize = 30, aspect=1.3)
				 #xticks = [0.00, 0.05, 0.1, 0.15, 0.20])
				 #yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

	hist!(change_points_time, bins=n_bins_change_point_hist)
	if ground_truth_available
		vlines!(gt_change_points_time, color=:red)
	end
	xlims!(axis, 0.00, (n_time_points-1) * dt)

	if ground_truth_available
		legend_line = LineElement(color = :red, linewidth = 1.0)
		axislegend(axis, [legend_line], ["Ground Truth"],
						position=:rt, framevisible=false, labelcolor=:Black)
	end

	# Plot 3
	axis = Axis(fig[2, 2]; xlabel = L"Dwell \, Time \, (s)", ylabel = L"PDF",
				 xlabelsize = 30, ylabelsize = 30, aspect=1.3)
				 #xticks = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
				 #yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

	xlims!(axis, 0.00, upper_lim_dwell_time_hist)
	#band!(dwell_grid, fill(0, length(dwell_grid)), 
	#		gamma_prior_pdf, color = (:orange, 0.25))
	hist!(dwells, bins=n_bins_dwell_time_hist, normalization=:pdf)
	lines!(dwell_grid, gamma_pdf, color=:red, linewidth=4.0)
  	text!(text_location_x, text_location_y, 
			text="visual Gamma fit, α = 1.88, θ = 0.0088/α (s)",
  				color=:red, align=(:center, :top), fontsize=20)

	legend_line = LineElement(color = (:orange, 0.25), linewidth = 20.0)
	axislegend(axis, [legend_line], ["Prior"],
					position=:rc, framevisible=false)


	# Plot 4
	axis = Axis(fig[2, 3]; xlabel = L"Jump \, Size \, (nm)", ylabel = L"PDF",
				 xlabelsize = 30, ylabelsize = 30, aspect=1.3)
				 #xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
				 #yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

	xlims!(axis, 0.00, upper_lim_jump_size_hist)
	hist!(jump_sizes .* 1e9, bins=n_bins_jump_size_hist, normalization=:pdf)


	axis = Axis(fig[2, 4]; xlabel = L"Dwell \, Time \, Gamma \, Shape \, Parameter", ylabel = L"PDF",
				 xlabelsize = 30, ylabelsize = 30, aspect=1.3)
				 #xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
				 #yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

	xlims!(axis, 0.00, upper_lim_gamma_shape_hist)
	hist!(mcmc_dwell_gamma_shape, bins=n_bins_gamma_shape_hist, normalization=:pdf)
	#vlines!([1.86], color=:red)
 # 	text!(1.9, 0.84, 
	#		text="1.86",
 # 				color=:red, align=(:center, :top), fontsize=20)

	axis = Axis(fig[2, 5]; xlabel = L"Dwell \, Time \, Gamma \, Mean \, Parameter \, (s)", ylabel = L"PDF",
				 xlabelsize = 30, ylabelsize = 30, aspect=1.3)
				 #xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
				 #yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

	xlims!(axis, 0.00, upper_lim_gamma_mean_hist)
	hist!(mcmc_dwell_gamma_mean, bins=n_bins_gamma_mean_hist, normalization=:pdf)
	#vlines!([0.007], color=:red)
 # 	text!(0.011, 240, 
	#		text="0.007",
 # 				color=:red, align=(:center, :top), fontsize=20)
	rowsize!(fig.layout, 2, Aspect(5, 1.0))

	fig

end
plot_file = string(current_directory, "analysis_plot_dynein_with_Gamma_parameters_integer_search_only_200_samples.png")
save(plot_file, fig)
