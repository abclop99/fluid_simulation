struct SimParams {
	timestep: f32,
	viscosity: f32,
	support_radius: f32,
	smoothing_radius: f32,
	bounding_box_min: vec3<f32>,
	bounding_box_ks: f32,
	bounding_box_max: vec3<f32>,
	bounding_box_kd: f32,
	gravity: vec3<f32>,
	_padding2: f32,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read> particles_src: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_dst: array<Particle>;

struct Particle {
	position: vec3<f32>,
	mass: f32,
	velocity: vec3<f32>,
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
	let total = arrayLength(&particles_src);
	let index = global_invocation_id.x;
	if (index >= total) {
		return;
	}

	let position = particles_src[index].position;
	let mass = particles_src[index].mass;
	let  velocity = particles_src[index].velocity;

	// dv/dt
	var acceleration = vec3<f32>();

	// External forces, not including gravity.
	var forces = vec3<f32>();

	// Bounding boxes
	let bounding_box_forces = vec3<f32>(
		max(params.bounding_box_min.x - position.x, 0.0),
		max(params.bounding_box_min.y - position.y, 0.0),
		max(params.bounding_box_min.z - position.z, 0.0),
	) * params.bounding_box_ks + vec3<f32>(
		min(params.bounding_box_max.x - position.x, 0.0),
		min(params.bounding_box_max.y - position.y, 0.0),
		min(params.bounding_box_max.z - position.z, 0.0),
	) * params.bounding_box_ks;
	// Damping
	let bounding_box_damping_force = vec3<f32>(
		select(-velocity.x, 0.0, bounding_box_forces.x == 0.0),
		select(-velocity.y, 0.0, bounding_box_forces.y == 0.0),
		select(-velocity.z, 0.0, bounding_box_forces.z == 0.0)
	) * params.bounding_box_kd;

	forces += bounding_box_forces + bounding_box_damping_force;
	
	// Loop over all other particles
	var i: u32 = 0u;
	loop {
		if i >= total {
			break;
		}
		if i == index {
			continue;
		}

		// TODO
		
		continuing {
			i++;
		}
	}

	// 
	acceleration += params.gravity;
	acceleration += forces / mass;

	// Forward Euler Integration
	let new_velocity = velocity + acceleration * params.timestep;
	let new_position = position + new_velocity * params.timestep;

	// Write back
	particles_dst[index] = Particle(new_position, mass, new_velocity);
}
