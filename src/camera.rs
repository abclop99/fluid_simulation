use cgmath::{prelude::*, Matrix4, Point3, Vector3};
use std::time::Duration;

/// Converts a matrix from OpenGL to WGPU format.
#[rustfmt::skip]
const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

pub struct Camera {
    /// The point the camera is looking at.
    target: Point3<f32>,
    /// The distance from the camera to the target.
    distance: f32,
    /// The angle of the camera around the target.
    azimuth: f32,
    /// The angle of the camera above the target.
    inclination: f32,

    fovy: f32,
    aspect: f32,
    znear: f32,
    zfar: f32,

    proj_view_buffer: wgpu::Buffer,
    proj_view_bind_group: wgpu::BindGroup,

    rotate_right: bool,
    rotate_left: bool,
    rotate_up: bool,
    rotate_down: bool,
    zoom_in: bool,
    zoom_out: bool,

    uniform_needs_update: bool,
}

impl Camera {
    /// Creates a new instance of `Camera`.
    pub fn new(device: &wgpu::Device) -> Self {
        let (proj_view_buffer, proj_view_bind_group) = Self::create_buffer_and_bind_group(device);

        Self {
            target: Point3::new(0.0, 0.0, 0.0),
            distance: 5.0,
            azimuth: 0.0,
            inclination: 0.0,

            fovy: 45.0,
            aspect: 1.0,
            znear: 0.1,
            zfar: 200.0,

            proj_view_buffer,
            proj_view_bind_group,

            rotate_right: false,
            rotate_left: false,
            rotate_up: false,
            rotate_down: false,
            zoom_in: false,
            zoom_out: false,

            uniform_needs_update: true,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, timestep: Duration) {
        if self.uniform_needs_update {
            // Perform camera movement
            let timestep = timestep.as_secs_f32();
            if self.rotate_right {
                self.azimuth += 1.0 * timestep;
            }
            if self.rotate_left {
                self.azimuth -= 1.0 * timestep;
            }
            if self.rotate_up {
                self.inclination += 1.0 * timestep;
            }
            if self.rotate_down {
                self.inclination -= 1.0 * timestep;
            }
            if self.zoom_in {
                self.distance -= 5.0 * timestep;
                if self.distance < 0.1 {
                    self.distance = 0.1;
                }
            }
            if self.zoom_out {
                self.distance += 5.0 * timestep;
            }

            let proj_view = self.view_proj();

            // Test point (1, 1, 1) in world space transformed by the view-projection matrix.
            println!(
                "Test point: {:?}",
                proj_view * Point3::new(1.0, 1.0, 1.0).to_homogeneous()
            );

            let proj_view: [[f32; 4]; 4] = proj_view.into();

            queue.write_buffer(
                &self.proj_view_buffer,
                0,
                bytemuck::cast_slice(&[proj_view]),
            );
            self.uniform_needs_update = false;
        }
    }

    pub fn view_proj(&self) -> Matrix4<f32> {
        let view = Matrix4::from_translation(Vector3::new(0.0, 0.0, self.distance))
            * Matrix4::from_angle_x(cgmath::Deg(self.inclination))
            * Matrix4::from_angle_y(cgmath::Deg(self.azimuth))
            * Matrix4::from_translation(Point3::origin() - self.target);

        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        OPENGL_TO_WGPU_MATRIX * proj * view
    }

    /// Gets the bind group layout for Camera.
    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_buffer_and_bind_group(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::BindGroup) {
        let proj_view_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform Buffer"),
            size: std::mem::size_of::<[[f32; 4]; 4]>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let proj_view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &Self::bind_group_layout(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: proj_view_buffer.as_entire_binding(),
            }],
        });

        (proj_view_buffer, proj_view_bind_group)
    }
}

impl Camera {
    pub fn rotate_right(&mut self) {
        self.rotate_right = true;
        self.uniform_needs_update = true;
    }
    pub fn rotate_left(&mut self) {
        self.rotate_left = true;
        self.uniform_needs_update = true;
    }
    pub fn rotate_up(&mut self) {
        self.rotate_up = true;
        self.uniform_needs_update = true;
    }
    pub fn rotate_down(&mut self) {
        self.rotate_down = true;
        self.uniform_needs_update = true;
    }
    pub fn zoom_in(&mut self) {
        self.zoom_in = true;
        self.uniform_needs_update = true;
    }
    pub fn zoom_out(&mut self) {
        self.zoom_out = true;
        self.uniform_needs_update = true;
    }
}

pub trait BindCamera<'a> {
    fn bind_camera(&mut self, camera: &'a Camera);
}

impl<'a> BindCamera<'a> for wgpu::RenderPass<'a> {
    fn bind_camera(&mut self, camera: &'a Camera) {
        self.set_bind_group(0, &camera.proj_view_bind_group, &[]);
    }
}
