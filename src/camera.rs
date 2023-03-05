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

const CAMERA_SPEED: f32 = 50.0;

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
}

impl Camera {
    /// Creates a new instance of `Camera`.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (proj_view_buffer, proj_view_bind_group) = Self::create_buffer_and_bind_group(device);

        let mut camera = Self {
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
        };

        camera.set_proj_view(queue);

        camera
    }

    pub fn update(&mut self, queue: &wgpu::Queue, timestep: Duration) {
        // Perform camera movement
        let timestep = timestep.as_secs_f32();
        if self.rotate_right {
            self.azimuth += CAMERA_SPEED * timestep;
        }
        if self.rotate_left {
            self.azimuth -= CAMERA_SPEED * timestep;
        }
        if self.rotate_up {
            self.inclination += CAMERA_SPEED * timestep;
        }
        if self.rotate_down {
            self.inclination -= CAMERA_SPEED * timestep;
        }
        if self.zoom_in {
            self.distance -= 1.0 * timestep;
            if self.distance < 0.1 {
                self.distance = 0.1;
            }
        }
        if self.zoom_out {
            self.distance += 1.0 * timestep;
        }

        if self.rotate_right
            || self.rotate_left
            || self.rotate_up
            || self.rotate_down
            || self.zoom_in
            || self.zoom_out
        {
            self.set_proj_view(queue);
        }
    }

    fn set_proj_view(&mut self, queue: &wgpu::Queue) {
        let view = (Matrix4::identity()
            * Matrix4::from_translation(Point3::origin() - self.target)
            * Matrix4::from_angle_y(cgmath::Deg(self.azimuth))
            * Matrix4::from_angle_x(cgmath::Deg(self.inclination))
            * Matrix4::from_translation(Vector3::new(0.0, 0.0, self.distance)))
        .invert()
        .unwrap();

        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        let proj_view = OPENGL_TO_WGPU_MATRIX * proj * view;

        let proj_view: [[f32; 4]; 4] = proj_view.into();

        queue.write_buffer(
            &self.proj_view_buffer,
            0,
            bytemuck::cast_slice(&[proj_view]),
        );
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
    pub fn rotate_right(&mut self, is_pressed: bool) {
        self.rotate_right = is_pressed;
    }
    pub fn rotate_left(&mut self, is_pressed: bool) {
        self.rotate_left = is_pressed;
    }
    pub fn rotate_up(&mut self, is_pressed: bool) {
        self.rotate_up = is_pressed;
    }
    pub fn rotate_down(&mut self, is_pressed: bool) {
        self.rotate_down = is_pressed;
    }
    pub fn zoom_in(&mut self, is_pressed: bool) {
        self.zoom_in = is_pressed;
    }
    pub fn zoom_out(&mut self, is_pressed: bool) {
        self.zoom_out = is_pressed;
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
