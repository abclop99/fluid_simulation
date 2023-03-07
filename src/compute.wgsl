struct SimParams {
	timestep: f32,
	viscosity: f32,
	smoothing_radius: f32,
	particle_mass: f32,
	bounding_box_min: vec3<f32>,
	bounding_box_ks: f32,
	bounding_box_max: vec3<f32>,
	bounding_box_kd: f32,
	gravity: vec3<f32>,
	rest_density: f32,
	particle_stiffness: f32,
	// 3 word padding here
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particles_src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_dst: array<Particle>;

struct Particle {
	position: vec3<f32>,
	density: f32,
	velocity: vec3<f32>,
	pressure: f32,
}

// From std::f32::consts::PI
const PI: f32 = 3.14159265358979323846264338327950288;

/// Compute density and pressure
@compute
@workgroup_size(256)
fn density_main(
	@builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
	let total_particles = arrayLength(&particles_src);
	let index = global_invocation_id.x;
	if (index >= total_particles) {
		return;
	}

	let particle = &particles_src[index];

	var density: f32 = 0f;

	// Loop over neighbors to compute the density.
	var i: u32 = 0u;
	loop {
		let other = particles_src[i];

		let dist = distance((*particle).position, other.position);

		// Not in support radius -> no effect
		if dist > params.smoothing_radius * 2f {
			continue;
		}

		density = fma(params.particle_mass, kernel(dist), density);
		
		continuing {
			i++;
			break if i >= total_particles;
		}
	}

	let pressure = params.particle_stiffness
		* (pow(density/params.rest_density, 7f) - 1f);

	// Write back
	particles_dst[index].density = density;
	particles_dst[index].pressure = pressure;
}

/// Calculates forces and integrates them
@compute
@workgroup_size(256)
fn integration_main(
	@builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
	let total_particles = arrayLength(&particles_src);
	let index = global_invocation_id.x;
	if (index >= total_particles) {
		return;
	}

	let particle = &particles_src[index];

	var forces: vec3<f32> = vec3(0.0);
	
	// Bounding boxes
	forces = fma(vec3<f32>(
		max(params.bounding_box_min.x - (*particle).position.x, 0.0),
		max(params.bounding_box_min.y - (*particle).position.y, 0.0),
		max(params.bounding_box_min.z - (*particle).position.z, 0.0),
	), vec3(params.bounding_box_ks), forces);
	forces = fma(vec3<f32>(
		min(params.bounding_box_max.x - (*particle).position.x, 0.0),
		min(params.bounding_box_max.y - (*particle).position.y, 0.0),
		min(params.bounding_box_max.z - (*particle).position.z, 0.0),
	), vec3(params.bounding_box_ks), forces);

	// Damping
	forces = fma(vec3<f32>(
			select(-(*particle).velocity.x, 0.0,
				params.bounding_box_min.x <= (*particle).position.x && (*particle).position.x <= params.bounding_box_max.x),
			select(-(*particle).velocity.y, 0.0, 
				params.bounding_box_min.y <= (*particle).position.y && (*particle).position.y <= params.bounding_box_max.y),
			select(-(*particle).velocity.z, 0.0, 
				params.bounding_box_min.z <= (*particle).position.z && (*particle).position.z <= params.bounding_box_max.z)
		),
		vec3(params.bounding_box_kd),
		forces
	);

	var d_pressure = 0f;

	// Loop neighbor particles to calculate derivatives
	var i: u32 = 0u;
	loop {
		let other = particles_src[i];
		let dist = distance((*particle).position, other.position);

		// Ignore particles outside of support radius
		if dist > params.smoothing_radius * 2f {
			continue;
		}

		// Derivative of pressure
		// Using equation (9) from paper

		continuing {
			i++;
			break if i >= total_particles;
		}
	}

	var acceleration = params.gravity;
	acceleration += forces / params.particle_mass;

	// Limit timestep (not adaptive for now)
	let timestep = min(params.timestep, 0.016);

	// Forward Euler
	let new_velocity = fma(acceleration, vec3(timestep), (*particle).velocity);
	let new_position = fma(new_velocity, vec3(timestep), (*particle).position);

	// Write back
	particles_dst[index] = Particle(
		new_position, 0f,
		new_velocity, 0f,
	);
}

/// W_ij = W(|x_i - x_j| / h) = W(q) = 1/(h^d) f(q)
/// h is the smoothing radius.
fn kernel(distance: f32) -> f32 {
	let q: f32 = distance / params.smoothing_radius;

	var f_q: f32 = 0f;
	if 0f <= q && q < 1f {
		f_q = (2f/3f) - q * q + 0.5 * q * q * q;
	} else if 1f <= q && q < 2f {
		f_q = pow(2f-q, 3f)/6f;
	}

	f_q *= 3f / (2f * PI);

	return f_q / pow(params.smoothing_radius, 3f);
}

/// Computes the gradient of the kernel function.
/// Input: other.position - this.position
/// Output: Gradient vector
fn kernel_gradient(difference: vec3<f32>) -> vec3<f32> {
	let q: f32 = length(difference) / params.smoothing_radius;
	let e: vec3<f32> = normalize(difference);

	var df_dq: f32 = 0f;
	if 0f <= q && q < 1f {
		df_dq = (3f/2f) * q * q - 2f * q;
	}
	if 1f <= q && q < 2f {
		df_dq = -0.5 * (2f-q) * (2f-q);
	}

	return e * ( df_dq /pow(params.smoothing_radius, 4f));
}
