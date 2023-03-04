use cgmath::{Point3, Vector3};
use wgpu::util::DeviceExt;

pub struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl Mesh {
    pub fn new(device: &wgpu::Device, vertices: Vec<Vertex>, triangles: Vec<Triangle>) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(triangles.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let index_count = 3 * triangles.len() as u32;

        Self {
            vertex_buffer,
            index_buffer,
            index_count,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Position of the vertex. It is an array so that Bytemuck::Pod works.
    position: [f32; 3],
    /// Normal of the vertex. It is an array so that Bytemuck::Pod works.
    normal: [f32; 3],
}

impl Vertex {
    pub fn new(position: Point3<f32>, normal: Vector3<f32>) -> Self {
        let position: [f32; 3] = position.into();
        let normal: [f32; 3] = normal.into();
        Self { position, normal }
    }

    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Triangle(pub u16, pub u16, pub u16);

pub trait DrawMesh<'a> {
    fn draw_mesh(&mut self, mesh: &'a Mesh);
}

impl<'a, 'b> DrawMesh<'a> for wgpu::RenderPass<'b>
where
    'a: 'b,
{
    fn draw_mesh(&mut self, mesh: &'b Mesh) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        self.draw_indexed(0..mesh.index_count, 0, 0..1);
    }
}
