function get_active_inactive_loads!(loads, 
			loads_active,
			loads_inactive)

	n_change_points = count(x -> x != 0, loads) 
	n_inactive_change_points::Int64 = max_n_change_points - n_change_points

	loads_active .= 0
	loads_inactive .= 0

	loads_active[1:n_change_points] .= findall(x -> x != 0, loads)
	loads_inactive[1:n_inactive_change_points] .= findall(x -> x == 0, loads)

	return n_change_points 
end
function logpdf_Gamma(shape::Float64, scale::Float64, x::Float64)
	return -(loggamma(shape) + shape * log(scale)) + 
			(shape - 1.0) * log(x) - x/scale	
end

function logpdf_Beta(alpha::Float64, beta::Float64, x::Float64)
	return ((alpha - 1.0) * log(x) + (beta - 1.0) * 
			log(1.0 - x)) - logbeta(alpha, beta)
end

function logpdf_Dirichlet(alpha::Vector{Float64}, x::Vector{Float64})
	logpdf = 0.0
	for i in 1:length(alpha)
		logpdf += (alpha[i] - 1.0) * log(x[i]) - loggamma(alpha[i])
	end
	logpdf += loggamma(sum(alpha))
	return logpdf 
end

## -------------------------------------------------------------------
## FBM likelihood with finite correlation length.
## Keeps correlations only up to fbm_lag time points.
## -------------------------------------------------------------------

## Covariance at lag h for fractional Gaussian noise.
#function fbm_corr(h::Int, alpha::Float64)
    #if h == 0
        #return 1.0
    #else
        #return 0.5 * ((h + 1)^alpha - 2.0 * h^alpha + abs(h - 1)^alpha)
    #end
#end

## Store only the lower band of the covariance matrix.
## A[i,1] = lag 0
## A[i,2] = lag 1
## ...
## A[i,fbm_lag+1] = lag fbm_lag
#function fbm_cov_band(n::Int, sigma0::Float64, alpha::Float64, lag_max::Int)
    #A = zeros(Float64, n, lag_max + 1)
    #scale = sigma0^2

    #for i in 1:n
        #maxd = min(lag_max, i - 1)
        #for d in 0:maxd
            #A[i, d + 1] = scale * fbm_corr(d, alpha)
        #end
    #end

    #return A
#end

## Banded Cholesky factorization.
## Lfac has the same band layout as fbm_cov_band.
#function banded_cholesky(A::Matrix{Float64}, lag_max::Int)
    #n = size(A, 1)
    #Lfac = zeros(Float64, n, lag_max + 1)

    #for i in 1:n
        #jmin = max(1, i - lag_max)

        #for j in jmin:i
            #d = i - j
            #s = A[i, d + 1]

            #kmin = max(1, j - lag_max)
            #for k in kmin:j - 1
                #di = i - k
                #dj = j - k
                #if di <= lag_max && dj <= lag_max
                    #s -= Lfac[i, di + 1] * Lfac[j, dj + 1]
                #end
            #end

            #if i == j
                #Lfac[i, 1] = sqrt(max(s, fbm_jitter))
            #else
                #Lfac[i, d + 1] = s / Lfac[j, 1]
            #end
        #end
    #end

    #return Lfac
#end

## Log-density of a zero-mean Gaussian using the banded Cholesky factor.
#function banded_gaussian_logpdf(r::Vector{Float64}, Lfac::Matrix{Float64}, lag_max::Int)
    #n = length(r)
    #z = zeros(Float64, n)

    ## Solve L z = r
    #for i in 1:n
        #s = r[i]
        #jmin = max(1, i - lag_max)

        #for j in jmin:i - 1
            #d = i - j
            #s -= Lfac[i, d + 1] * z[j]
        #end

        #z[i] = s / Lfac[i, 1]
    #end

    #logdet = 0.0
    #for i in 1:n
        #logdet += 2.0 * log(Lfac[i, 1])
    #end

    #qform = 0.0
    #for i in 1:n
        #qform += z[i] * z[i]
    #end

    #return -0.5 * (qform + logdet + n * log(2π))
#end

#function get_log_likelihood_fbm(
    #reference_position_x::Float64,
    #reference_position_y::Float64,
    #reference_sigma_x::Float64,
    #reference_sigma_y::Float64,
    #k::Float64,
    #time_range
#)
    #lambda = k * D / (kb * T)
    #a = exp(-lambda * dt)

    #first_i = first(time_range)
    #last_i = last(time_range)
    #n = last_i - first_i + 1

    #if n <= 0
        #return 0.0
    #end

    ## Need i-1, so the first frame of the whole trace cannot be used here.
    #if first_i < 2
        #return -Inf
    #end

    ## Residuals for the whole dwell
    #rx = zeros(Float64, n)
    #ry = zeros(Float64, n)

    #idx = 1
    #for i in first_i:last_i
        #rx[idx] = detected_position_x[i] -
                  #a * detected_position_x[i - 1] -
                  #(1.0 - a) * reference_position_x

        #ry[idx] = detected_position_y[i] -
                  #a * detected_position_y[i - 1] -
                  #(1.0 - a) * reference_position_y

        #idx += 1
    #end

    ## x/y anisotropy comes from separate covariance scales
    #Ax = fbm_cov_band(n, reference_sigma_x, fbm_alpha, fbm_lag)
    #Ay = fbm_cov_band(n, reference_sigma_y, fbm_alpha, fbm_lag)

    #Lx = banded_cholesky(Ax, fbm_lag)
    #Ly = banded_cholesky(Ay, fbm_lag)

    #return banded_gaussian_logpdf(rx, Lx, fbm_lag) +
           #banded_gaussian_logpdf(ry, Ly, fbm_lag)
#end

#function get_log_likelihood(reference_position_x, 
#		reference_position_y, 
#		reference_sigma_x, 
#		reference_sigma_y, 
#		reference_sigma_xy, 
#		k, 
#		time_range)
#
#	# Based on exact OU process
#	lambda = k * D / (kb * T)
#	a = exp(-lambda * dt)
#	sigma = sqrt((kb*T/k) * (1.0 - a^2))
#
#	log_likelihood = 0.0
#	
#  	for i in time_range 
#
#	    	increment_x::Float64 = 
#			detected_position_x[i] - reference_position_x	
#	    	increment_y::Float64 = 
#			detected_position_y[i] - reference_position_y 
#
#		trap_extension_x::Float64 = 
#			detected_position_x[i-1] - reference_position_x 
#		trap_extension_y::Float64 = 
#			detected_position_y[i-1] - reference_position_y 
#	
#		log_likelihood += logpdf(Normal(0.0, reference_sigma_x), 
#					(increment_x - a * trap_extension_x)) + 
#				logpdf(Normal(0.0, reference_sigma_y), 
#					(increment_y - a * trap_extension_y))
#	
#	end
#
#	return log_likelihood
#end
#function get_log_likelihood(reference_position_x,
#        reference_position_y,
#        reference_sigma_x,
#        reference_sigma_y,
#        reference_sigma_xy,
#        k,
#        time_range)
#
#    # Based on exact OU process
#    lambda = k * D / (kb * T)
#    a = exp(-lambda * dt)
#
#    log_likelihood = 0.0
#
#    # Covariance matrix entries
#    sxx = reference_sigma_x^2
#    syy = reference_sigma_y^2
#    sxy = reference_sigma_xy
#
#    det_Sigma = sxx * syy - sxy^2
#
#    if det_Sigma <= 0.0
#        return -Inf
#    end
#
#    inv11 =  syy / det_Sigma
#    inv22 =  sxx / det_Sigma
#    inv12 = -sxy / det_Sigma
#
#    log_norm_const = -0.5 * log((2.0 * π)^2 * det_Sigma)
#
#    for i in time_range
#
#        increment_x::Float64 =
#            detected_position_x[i] - reference_position_x
#        increment_y::Float64 =
#            detected_position_y[i] - reference_position_y
#
#        trap_extension_x::Float64 =
#            detected_position_x[i-1] - reference_position_x
#        trap_extension_y::Float64 =
#            detected_position_y[i-1] - reference_position_y
#
#        residual_x = increment_x - a * trap_extension_x
#        residual_y = increment_y - a * trap_extension_y
#
#        quad_form = inv11 * residual_x^2 +
#                    2.0 * inv12 * residual_x * residual_y +
#                    inv22 * residual_y^2
#
#        log_likelihood += log_norm_const - 0.5 * quad_form
#    end
#
#    return log_likelihood
#end


function prepare_OU!(dwell, 
    diffusion_coeff,
	dynamics_matrix,
	stiffness_matrix,
	inv_stiffness_matrix,
	det_stiffness,
	OU_covariance,
	inv_OU_covariance,
	det_OU_covariance)

	det_stiffness[dwell] =
    		stiffness_matrix[dwell, 1,1] * stiffness_matrix[dwell, 2,2] -
    			stiffness_matrix[dwell, 1,2]^2

	inv_stiffness_matrix[dwell, 1,1] =
		stiffness_matrix[dwell, 2,2] / det_stiffness[dwell]
	
	inv_stiffness_matrix[dwell, 1,2] =
		-stiffness_matrix[dwell, 1,2] / det_stiffness[dwell]
	
	inv_stiffness_matrix[dwell, 2,1] =
		-stiffness_matrix[dwell, 1,2] / det_stiffness[dwell]
	
	inv_stiffness_matrix[dwell, 2,2] =
		stiffness_matrix[dwell, 1,1] / det_stiffness[dwell]

    	# Mean dynamics matrix
	dynamics_matrix[dwell, :, :] .= exp(-diffusion_coeff * dt / (kb * T) * view(stiffness_matrix, dwell, :, :))

    	# Exact OU covariance
	OU_covariance[dwell, :, :] .= kb * T * view(inv_stiffness_matrix, dwell, :, :) * 
					(I - view(dynamics_matrix, dwell, :, :) * view(dynamics_matrix, dwell, :, :))

	det_OU_covariance[dwell] = det(view(OU_covariance, dwell, :, :))

    	inv_OU_covariance[dwell, 1,1] =
		OU_covariance[dwell, 2,2] / det_OU_covariance[dwell]

    	inv_OU_covariance[dwell, 1,2] =
		-OU_covariance[dwell, 1,2] / det_OU_covariance[dwell]

    	inv_OU_covariance[dwell, 2,1] =
		-OU_covariance[dwell, 1,2] / det_OU_covariance[dwell]

    	inv_OU_covariance[dwell, 2,2] =
		OU_covariance[dwell, 1,1] / det_OU_covariance[dwell]

	return nothing
end

function get_log_likelihood!(dwell,
    	reference_position_x,
    	reference_position_y,
	dynamics_matrix,
	det_stiffness,
	OU_covariance,
	inv_OU_covariance,
	det_OU_covariance,
    	time_range)


	if !isfinite(det_stiffness[dwell]) || det_stiffness[dwell] <= 0.0
    		return -Inf
	end

	if !isfinite(det_OU_covariance[dwell]) || det_OU_covariance[dwell] <= 0.0
    	    return -Inf
    	end

	log_norm_const = -0.5 * log((2.0 * π)^2 * det_OU_covariance[dwell])

    log_likelihood = 0.0

    for i in time_range

   	    increment_x::Float64 =
   	    	    detected_position_x[i] - reference_position_x

   	    increment_y::Float64 =
   	    	    detected_position_y[i] - reference_position_y

   	    trap_extension_x::Float64 =
   	    	    detected_position_x[i-1] - reference_position_x

   	    trap_extension_y::Float64 =
   	    	    detected_position_y[i-1] - reference_position_y

   	    residual_x =
   	    	    increment_x -
   	    	    (dynamics_matrix[dwell, 1,1] * trap_extension_x +
   	    	     dynamics_matrix[dwell, 1,2] * trap_extension_y)

   	    residual_y =
   	    	    increment_y -
   	    	    (dynamics_matrix[dwell, 2,1] * trap_extension_x +
   	    	     dynamics_matrix[dwell, 2,2] * trap_extension_y)

   	    quad_form =
   	    	    inv_OU_covariance[dwell, 1,1] * residual_x^2 +
   	    	    2.0 * inv_OU_covariance[dwell, 1,2] * residual_x * residual_y +
   	    	    inv_OU_covariance[dwell, 2,2] * residual_y^2

   	    log_likelihood += log_norm_const - 0.5 * quad_form
    end

    return log_likelihood
end
