use crate::{
    camera::{self, BindCamera},
    framework::Application,
    lighting::{self, BindLights},
    mesh,
    texture::Texture,
};
use nanorand::{Rng, WyRand};
use rayon::prelude::*;
use std::time::Duration;
use wgpu::util::DeviceExt;
use winit::event::*;

use mesh::DrawMesh;

const PARTICLES_PER_WORKGROUP: u32 = 256;

const PARTICLE_RENDER_RADIUS: f32 = 0.025;

const NUM_PARTICLES: u32 = 1_000;

/// The fluid simulation.
pub struct Simulation {
    simulation_params: SimulationParams,
    simulation_params_buffer: wgpu::Buffer,

    density_pipeline: wgpu::ComputePipeline,
    integration_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    depth_texture: Texture,

    particle_buffers: Vec<wgpu::Buffer>,
    particle_bind_groups: Vec<wgpu::BindGroup>,

    camera: camera::Camera,

    /// Mesh used to render the particles.
    particle_mesh: mesh::Mesh,
    particle_material: lighting::Material,

    light_buffer: lighting::LightBuffer,

    work_group_count: u32,
    current_buffer: usize,
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

        // Update simulation parameters
        self.simulation_params.timestep = timestep.as_secs_f32();
        queue.write_buffer(
            &self.simulation_params_buffer,
            0,
            bytemuck::cast_slice(&[self.simulation_params]),
        );
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
        let compute_bind_group_layout = compute_bind_group_layout(device);
        let (density_pipeline, integration_pipeline) =
            create_compute_pipelines(config, device, &compute_bind_group_layout);
        let render_pipeline = create_render_pipeline(config, device);

        // Simulation Parameters
        let simulation_params = SimulationParams {
            timestep: 0.01,
            viscosity: 0.0,
            smoothing_radius: 0.1,
            bounding_box_min: [-1.0, -1.0, -1.0],
            bounding_box_max: [1.0, 1.0, 1.0],
            bounding_box_ks: 100.0,
            bounding_box_kd: 3.0,
            gravity: [0.0, -9.81, 0.0],

            particle_mass: 1.0 / (NUM_PARTICLES as f32),
            rest_density: 2.0,
            particle_stiffness: 1.5E0,

            padding: [0f32; 3],
        };
        let simulation_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Parameters Buffer"),
                contents: bytemuck::cast_slice(&[simulation_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Particle Buffer Data
        let mut initial_particle_data = vec![0.0f32; (NUM_PARTICLES * 8) as usize];
        initial_particle_data
            .par_chunks_exact_mut(8)
            .for_each_init(WyRand::new, |rng, chunk| {
                // Position
                chunk[0] = rng.generate::<f32>() * 2f32 - 1f32;
                chunk[1] = rng.generate::<f32>() * 2f32 - 1f32;
                chunk[2] = rng.generate::<f32>() * 2f32 - 1f32;
                // Velocity and Padding already 0
            });

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

        // Particle bind groups
        let mut particle_bind_groups = Vec::<wgpu::BindGroup>::new();
        for i in 0..2 {
            particle_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Particle Bind Group {i}")),
                layout: &compute_bind_group_layout,
                entries: &[
                    // Simulation Parameters
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: simulation_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[i].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: particle_buffers[(i + 1) % 2].as_entire_binding(),
                    },
                ],
            }));
        }

        let camera = camera::Camera::new(config, device, queue);

        let depth_texture = Texture::create_depth_texture(device, config, "Depth Texture");

        // Particles
        let particle_mesh = mesh::shapes::icosahedron(device, PARTICLE_RENDER_RADIUS);
        let particle_material = lighting::Material::new(
            device,
            queue,
            lighting::MaterialUniform {
                diffuse: [0.5, 0.5, 1.0, 1.0],
                specular: [0.1, 0.1, 0.1, 1.0],
                shininess: 32.0,
                ..Default::default()
            },
        );

        // Lights
        let lights = vec![
            lighting::LightUniform::new([1.5, 5.0, 2.0, 1.0], [0.7, 0.4, 0.4, 1.0]),
            lighting::LightUniform::new([1.5, 5.0, -2.0, 1.0], [0.3, 0.6, 0.3, 1.0]),
            lighting::LightUniform::new([-2.0, 4.0, 2.0, 1.0], [0.5, 0.5, 0.8, 1.0]),
            lighting::LightUniform::new([0.0, -5.0, 0.0, 1.0], [0.3, 0.3, 0.3, 1.0]),
        ];
        let light_buffer = lighting::LightBuffer::new(device, queue, lights);

        let work_group_count =
            (NUM_PARTICLES as f32 / PARTICLES_PER_WORKGROUP as f32).ceil() as u32;

        Self {
            simulation_params,
            simulation_params_buffer,
            density_pipeline,
            integration_pipeline,
            render_pipeline,
            particle_buffers,
            particle_bind_groups,
            particle_mesh,
            particle_material,
            camera,
            depth_texture,
            light_buffer,
            work_group_count,
            current_buffer: 0,
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        self.camera.resize(config.width, config.height);
        self.depth_texture = Texture::create_depth_texture(_device, config, "Depth Texture");
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
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        };

        // Get the command encoder
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        // Compute pass
        command_encoder.push_debug_group("Compute Pass");
        {
            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                });
            compute_pass.set_pipeline(&self.density_pipeline);
            compute_pass.set_bind_group(0, &self.particle_bind_groups[self.current_buffer], &[]);
            compute_pass.dispatch_workgroups(self.work_group_count, 1, 1);

            compute_pass.set_pipeline(&self.integration_pipeline);
            compute_pass.set_bind_group(
                0,
                &self.particle_bind_groups[(self.current_buffer + 1) % 2],
                &[],
            );
            compute_pass.dispatch_workgroups(self.work_group_count, 1, 1);
        }
        command_encoder.pop_debug_group();

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
                &self.particle_buffers[self.current_buffer],
                0..NUM_PARTICLES,
            );
        }
        command_encoder.pop_debug_group();

        // Submit the command encoder
        queue.submit(Some(command_encoder.finish()));

        // Swap the buffers for the next frame
        //self.current_buffer = (self.current_buffer + 1) % 2;
    }
}

/// Parameters for the simulation.
/// Some don't change often, while others are updated every frame.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimulationParams {
    /// The timestep for this frame.
    timestep: f32,
    /// The viscosity of the fluid.
    viscosity: f32,
    /// The smoothing radius of the fluid.
    smoothing_radius: f32,
    /// Mass of each particle
    particle_mass: f32,
    /// Lower bound of the bounding box.
    bounding_box_min: [f32; 3],
    /// The "spring constant" for collisions with the bounding box.
    bounding_box_ks: f32,
    /// Upper bound of the bounding box.
    bounding_box_max: [f32; 3],
    /// The "damping constant" for collisions with the bounding box.
    bounding_box_kd: f32,
    /// Gravity vector.
    gravity: [f32; 3],
    /// Target rest density
    rest_density: f32,
    /// Stiffness used to calculate pressure
    particle_stiffness: f32,
    /// Padding
    padding: [f32; 3],
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

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    attributes: &wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32, 4 => Float32x3, 5 => Float32,],
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
        depth_stencil: Some(wgpu::DepthStencilState {
            format: Texture::DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

/// Returns the compute pipelines for the program
///
fn create_compute_pipelines(
    _config: &wgpu::SurfaceConfiguration,
    device: &wgpu::Device,
    compute_bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
    });

    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[compute_bind_group_layout],
        push_constant_ranges: &[],
    });

    let density_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "density_main",
    });

    let integration_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "integration_main",
    });

    (density_pipeline, integration_pipeline)
}

fn compute_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Compute Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<SimulationParams>() as _,
                    ),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 32) as _),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 32) as _),
                },
                count: None,
            },
        ],
    })
}
