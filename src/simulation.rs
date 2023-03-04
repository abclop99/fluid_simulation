use crate::framework::Application;

mod mesh;
use mesh::DrawMesh;
mod shapes;

const PARTICLE_RENDER_RADIUS: f32 = 0.1;

/// The fluid simulation.
pub struct Simulation {
    render_pipeline: wgpu::RenderPipeline,

    /// Mesh used to render the particles.
    particle_mesh: mesh::Mesh,
}

impl Application for Simulation {
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let render_pipeline = create_render_pipeline(config, device);

        let particle_mesh = shapes::icosahedron(device, PARTICLE_RENDER_RADIUS);

        // TODO
        Self {
            render_pipeline,
            particle_mesh,
        }
    }

    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        // TODO
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        // empty
    }

    fn render(&mut self, view: &wgpu::TextureView, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Create render pass descriptor and its color attachments
        let color_attachment = [Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: true,
            },
        })];
        let render_pass_descriptor = wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &color_attachment,
            depth_stencil_attachment: None, // TODO: Add depth buffer
        };

        // Get the command encoder
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // Render pass
        command_encoder.push_debug_group("Render Pass");
        {
            let mut render_pass = command_encoder.begin_render_pass(&render_pass_descriptor);
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_mesh(&self.particle_mesh);
        }
        command_encoder.pop_debug_group();

        // Submit the command encoder
        queue.submit(Some(command_encoder.finish()));
    }
}

/// Creates the render pipeline.
fn create_render_pipeline(
    config: &wgpu::SurfaceConfiguration,
    device: &wgpu::Device,
) -> wgpu::RenderPipeline {
    let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Render Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("render.wgsl").into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Render Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: "vertex_main",
            buffers: &[mesh::Vertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_shader,
            entry_point: "fragment_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None, // TODO: Enable depth testing.
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    render_pipeline
}
