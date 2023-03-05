struct CameraUniform {
	proj: mat4x4<f32>,
	view: mat4x4<f32>,
	normal_transform: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct Material {
	ambient: vec4<f32>,
	diffuse: vec4<f32>,
	specular: vec4<f32>,
	emission: vec4<f32>,
	shininess: f32,
}
@group(1) @binding(0)
var<uniform> material: Material;

struct Light {
	position: vec4<f32>,
	color: vec4<f32>,
}
@group(2) @binding(0)
var<storage, read> lights: array<Light>;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) normal: vec3<f32>,

	@location(2) particle_position: vec3<f32>,
	@location(3) particle_mass: f32,
	@location(4) particle_velocity: vec3<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) particle_position: vec3<f32>,
	@location(1) fragment_position: vec4<f32>,
	@location(2) normal: vec3<f32>,
}

@vertex
fn vertex_main(
	in: VertexInput,
) -> VertexOutput {
	var out: VertexOutput;

	let in_position = in.position + in.particle_position;
	out.position = camera.proj * camera.view * vec4<f32>(in_position, 1.0);
	out.fragment_position = vec4<f32>(in_position, 1.0);

	// Transfom the normal
	out.normal = from_homogeneous( camera.normal_transform * vec4(in.normal, 0.0) );

	return out;
}

@fragment
fn fragment_main(
	in: VertexOutput
) -> @location(0) vec4<f32> {

	let fragment_position = camera.view * in.fragment_position;

	let viewer = normalize( vec3<f32>(0.0, 0.0, 0.0) - from_homogeneous(fragment_position) );
	let normal = normalize( in.normal );

	var frag_color: vec4<f32> = material.emission;

	let total_lights: u32 = arrayLength(&lights);
	var light_num: u32 = u32(0);
	loop {
		if light_num > total_lights {
			break;
		}

		let light_pos = camera.view * lights[light_num].position;
		//let light_pos = lights[light_num].position;
		let light_color = lights[light_num].color;

		// Points toward light; allows light to be a vector.
		let to_light = normalize(
			(fragment_position.w * light_pos.xyz)
			- (light_pos.w * fragment_position.xyz)
		);

		// Ambient
		frag_color += material.ambient * light_color;

		// Diffuse
		frag_color += diffuseShading(
			viewer, normal, to_light, material.diffuse, light_color,
		);

		// Specular
		frag_color += specularShading(
			viewer, normal, to_light, material.specular,
			light_color, material.shininess,
		);

		light_num++;
	}

	return frag_color;
	//return vec4<f32>(0.5 * in.normal + 0.5, 1.0);
}

fn from_homogeneous(v: vec4<f32>) -> vec3<f32> {
	if v.w == 0.0 {
		return v.xyz;
	} else {
		return v.xyz / v.w;
	}
}

fn diffuseShading(
	viewer: vec3<f32>,
	normal: vec3<f32>,
	light: vec3<f32>,
	diffuse_color: vec4<f32>,
	light_color: vec4<f32>,
) -> vec4<f32> {
	let amount = max( dot(normal, light), 0.0 );
	return diffuse_color * light_color * amount;
}

fn specularShading(
	viewer: vec3<f32>,
	normal: vec3<f32>,
	light: vec3<f32>,
	specular_color: vec4<f32>,
	light_color: vec4<f32>,
	shininess: f32,
) -> vec4<f32> {
	let halfway = normalize(viewer + light);
	return specular_color * light_color * pow(max(dot(normal, halfway), 0.0), shininess);
}
