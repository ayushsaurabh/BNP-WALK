using HDF5
using Plots

working_directory = "/home/mbgnjasb/codes/chu_lab/experimental_data/"

#fname = string(working_directory, "trajectory_2026-01-16-T03-24-37Z_picoscope_2025-11-20-T01-10-03Z-002.h5")
fname = string(working_directory, "trajectory_2026-07-20-T23-49-01Z_picoscope_2026-06-02-T00-04-01Z.h5")

fid = h5open(fname, "r")
valid_ids = read(fid, "analysis/idx_valid")
xy = fid["analysis/xy"] #meters
min_valid_index = valid_ids[1]+1
max_valid_index = valid_ids[2]
xy_without_NaNs = xy[:, :, valid_ids[1]+1:valid_ids[1]+1000] #meters

dt = read(fid, "analysis/t/xy/dt") #seconds
t_0 = read(fid, "analysis/t/xy/t_0")  #seconds

#good_data_start_time_index = Integer(round(35.2/dt)) #seconds
#good_data_end_time_index =  Integer(round(36.9/dt)) #seconds
good_data_start_time_index = Integer(round(1.0/dt))+1 #seconds
good_data_end_time_index =  Integer(round(3.0/dt)) #seconds

xy_good = xy[:, :, good_data_start_time_index:good_data_end_time_index]

close(fid)

n_points = size(xy_good)[3]
x = zeros(n_points)
y = zeros(n_points)
for i in 1:n_points
    x[i] = xy_good[1, 1, i]
    y[i] = xy_good[1, 2, i]
end

fname = string(working_directory, "processed_trajectory_invivo.h5")
fid = h5open(fname, "w")
write_dataset(fid, "x(m)", x)
write_dataset(fid, "y(m)", y)
write_dataset(fid, "dt(s)", dt)
close(fid)

n_points = size(x)[1]
data_range = 1:Integer(round(n_points))
data_range = 1:7000
1000*dt
n_points * dt

# PyQT graph

xy_plot = plot((x[data_range] .- x[1]) .* 10^9, (y[data_range] .- y[1]) .* 10^9, 
            xlabel = "x (nm)", ylabel = "y (nm)", legend = false,
            linewidth = 1.0, alpha=0.5)
savefig(xy_plot, string(working_directory, "xy_plot.png"))

tx_plot = plot(collect(1:n_points) .* (dt), x[data_range] .* 10^6, xlabel = "t (μs)", ylabel = "x (μm)", legend = false)
savefig(tx_plot, string(working_directory, "tx_plot.png"))

ty_plot = plot(collect(1:n_points) .* (dt), y[data_range] .* 10^6, xlabel = "t (μs)", ylabel = "y (μm)", legend = false)
savefig(ty_plot, string(working_directory, "ty_plot.png"))



