use crate::{
    camera::{self, BindCamera},
    framework::Application,
    lighting::{self, BindLights},
    mesh,
};
use nanorand::{Rng, WyRand};
use rayon::prelude::*;
use std::time::Duration;
use wgpu::util::DeviceExt;
use winit::event::*;

use mesh::DrawMesh;

const PARTICLE_RENDER_RADIUS: f32 = 0.01;

const NUM_PARTICLES: u32 = 100_000;
const PARTICLE_MASS: f32 = 0.01;

/// The fluid simulation.
pub struct Simulation {
    render_pipeline: wgpu::RenderPipeline,

    particle_buffers: Vec<wgpu::Buffer>,

    camera: camera::Camera,

    /// Mesh used to render the particles.
    particle_mesh: mesh::Mesh,
    particle_material: lighting::Material,

    light_buffer: lighting::LightBuffer,
}

impl Simulation {
    /// Applies CPU-side updates
    fn update(
        &mut self,
        _view: &wgpu::TextureView,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        timestep: Duration,
    ) {
        self.camera.update(queue, timestep);
    }
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
        queue: &wgpu::Queue,
    ) -> Self {
        let render_pipeline = create_render_pipeline(config, device);

        // TODO: Compute pipeline

        // Particle Buffer Data
        let mut initial_particle_data = vec![0.0f32; (NUM_PARTICLES * 8) as usize];
        //let mut rng = WyRand::new_seed(42);
        //let mut unif = || rng.generate::<f32>() * 2f32 - 1f32;
        initial_particle_data.par_chunks_exact_mut(8).for_each_init(
            || WyRand::new(),
            |rng, chunk| {
                // Position
                chunk[0] = rng.generate::<f32>() * 2f32 - 1f32;
                chunk[1] = rng.generate::<f32>() * 2f32 - 1f32;
                chunk[2] = rng.generate::<f32>() * 2f32 - 1f32;
                // Mass
                chunk[3] = PARTICLE_MASS;
                // Velocity and Padding already 0
            },
        );

        // Create particle buffers
        let mut particle_buffers = Vec::<wgpu::Buffer>::new();
        for i in 0..2 {
            particle_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Particle Buffer {i}")),
                    contents: bytemuck::cast_slice(&initial_particle_data),
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                }),
            );
        }

        // TODO: Particle bind groups

        let camera = camera::Camera::new(config, device, queue);

        // Particles
        let particle_mesh = mesh::shapes::icosahedron(device, PARTICLE_RENDER_RADIUS);
        let particle_material = lighting::Material::new(
            device,
            queue,
            lighting::MaterialUniform {
                diffuse: [0.2, 0.2, 1.0, 1.0],
                specular: [0.1, 0.1, 0.1, 1.0],
                ..Default::default()
            },
        );

        // Lights
        let lights = vec![lighting::LightUniform::new(
            [10.0, 10.0, 10.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        )];
        let light_buffer = lighting::LightBuffer::new(device, queue, lights);

        // TODO
        Self {
            render_pipeline,
            particle_buffers,
            particle_mesh,
            particle_material,
            camera,
            light_buffer,
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.camera.resize(config.width, config.height);
    }

    fn handle_event(&mut self, event: winit::event::WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::A => {
                        self.camera.zoom_in(is_pressed);
                    }
                    VirtualKeyCode::Z => {
                        self.camera.zoom_out(is_pressed);
                    }
                    VirtualKeyCode::Up => {
                        self.camera.rotate_up(is_pressed);
                    }
                    VirtualKeyCode::Down => {
                        self.camera.rotate_down(is_pressed);
                    }
                    VirtualKeyCode::Left => {
                        self.camera.rotate_left(is_pressed);
                    }
                    VirtualKeyCode::Right => {
                        self.camera.rotate_right(is_pressed);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        timestep: Duration,
    ) {
        self.update(view, device, queue, timestep);

        // Create render pass descriptor and its color attachments
        let color_attachment = [Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 0.1,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                }),
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
            render_pass.bind_camera(&self.camera);
            render_pass.bind_light_buffer(&self.light_buffer);
            render_pass.draw_mesh(
                &self.particle_mesh,
                &self.particle_material,
                &self.particle_buffers[0],
                0..NUM_PARTICLES,
            );
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
        bind_group_layouts: &[
            &camera::Camera::bind_group_layout(device),
            &lighting::Material::bind_group_layout(device),
            &lighting::LightBuffer::bind_group_layout(device),
        ],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &render_shader,
            entry_point: "vertex_main",
            buffers: &[
                mesh::Vertex::desc(),
                wgpu::VertexBufferLayout {
                    array_stride: 4 * 4 * 2,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32, 4 => Float32x3],
                },
            ],
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
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None, // TODO: Enable depth testing.
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    render_pipeline
}
