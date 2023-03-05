pub struct Material {
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    ambient: [f32; 4],
    diffuse: [f32; 4],
    specular: [f32; 4],
    emission: [f32; 4],
    shininess: f32,

    /// For alignment
    _padding: [f32; 3],
}

impl Default for MaterialUniform {
    fn default() -> Self {
        Self {
            ambient: [0.02, 0.05, 0.1, 1.0],
            diffuse: [0.6, 0.65, 0.7, 1.0],
            specular: [0.9, 0.9, 0.9, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            shininess: 100.0,
            _padding: [0.0, 0.0, 0.0],
        }
    }
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material_uniform: MaterialUniform,
    ) -> Self {
        let buffer = Self::create_buffer(device);
        let bind_group = Self::create_bind_group(device, &Self::bind_group_layout(device), &buffer);

        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[material_uniform]));

        Self { bind_group }
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_buffer(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Material Buffer"),
            size: std::mem::size_of::<MaterialUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }

    pub fn get_bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    position: [f32; 4],
    color: [f32; 4],
}

impl LightUniform {
    pub fn new(position: [f32; 4], color: [f32; 4]) -> Self {
        Self { position, color }
    }
}

pub struct LightBuffer {
    bind_group: wgpu::BindGroup,
}

impl LightBuffer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        light_uniforms: Vec<LightUniform>,
    ) -> Self {
        let buffer = Self::create_buffer(device, light_uniforms.len());
        let bind_group = Self::create_bind_group(device, &Self::bind_group_layout(device), &buffer);

        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&light_uniforms));

        Self { bind_group }
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_buffer(device: &wgpu::Device, light_count: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Buffer"),
            size: std::mem::size_of::<LightUniform>() as wgpu::BufferAddress
                * light_count as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }
}

pub trait BindLights<'a> {
    fn bind_light_buffer(&mut self, light_buffer: &'a LightBuffer);
}

impl<'a> BindLights<'a> for wgpu::RenderPass<'a> {
    fn bind_light_buffer(&mut self, light_buffer: &'a LightBuffer) {
        self.set_bind_group(2, &light_buffer.bind_group, &[]);
    }
}
