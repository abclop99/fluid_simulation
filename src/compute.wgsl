struct SimParams {
	timestep: f32,
	viscosity: f32,
	support_radius: f32,
	smoothing_radius: f32,
	bounding_box_min: vec3<f32>,
	_padding0: f32,
	bounding_box_max: vec3<f32>,
	_padding1: f32,
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

	var position = particles_src[index].position;
	let mass = particles_src[index].mass;
	var velocity = particles_src[index].velocity;
	
	// Test by simulating gravity to other particles
	var i: u32 = 0u;
	loop {
		if i >= total {
			break;
		}
		if i == index {
			continue;
		}
		
		let other_position = particles_src[i].position;

		let distance = distance(position, other_position);
		let e = normalize(other_position - position);

		if distance != 0.0 {
			velocity += inverseSqrt(distance) * e * 0.0000000001;
		}

		position += velocity * params.timestep;

		continuing {
			i++;
		}
	}

	// Write back
	particles_dst[index] = Particle(position, mass, velocity);
}
