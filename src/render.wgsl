struct CameraUniform {
	proj_view: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) normal: vec3<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) normal: vec3<f32>,
}

@vertex
fn vertex_main(
	in: VertexInput,
) -> VertexOutput {
	var out: VertexOutput;

	out.position = camera.proj_view * vec4<f32>(in.position, 1.0);
	out.normal = in.normal;

	return out;
}

@fragment
fn fragment_main(
	in: VertexOutput
) -> @location(0) vec4<f32> {
	return vec4<f32>(0.5 * in.normal + 0.5, 1.0);
}
