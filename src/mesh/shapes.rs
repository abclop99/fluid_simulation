/**
 * This module contains functions to generate various shapes.
 */
use cgmath::prelude::*;
use rayon::prelude::*;

use crate::mesh::{Mesh, Triangle, Vertex};

pub fn icosahedron(device: &wgpu::Device, radius: f32) -> Mesh {
    // Create the 12 vertices of the icosahedron.
    let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let vertices = vec![
        cgmath::vec3(-1.0, phi, 0.0).normalize() * radius,
        cgmath::vec3(1.0, phi, 0.0).normalize() * radius,
        cgmath::vec3(-1.0, -phi, 0.0).normalize() * radius,
        cgmath::vec3(1.0, -phi, 0.0).normalize() * radius,
        cgmath::vec3(0.0, -1.0, phi).normalize() * radius,
        cgmath::vec3(0.0, 1.0, phi).normalize() * radius,
        cgmath::vec3(0.0, -1.0, -phi).normalize() * radius,
        cgmath::vec3(0.0, 1.0, -phi).normalize() * radius,
        cgmath::vec3(phi, 0.0, -1.0).normalize() * radius,
        cgmath::vec3(phi, 0.0, 1.0).normalize() * radius,
        cgmath::vec3(-phi, 0.0, -1.0).normalize() * radius,
        cgmath::vec3(-phi, 0.0, 1.0).normalize() * radius,
    ];
    let vertices: Vec<Vertex> = vertices
        .par_iter()
        .map(|v| Vertex::new(cgmath::Point3::origin() + v, v.normalize()))
        .collect();

    // Create the 20 triangles of the icosahedron.
    let triangles = vec![
        Triangle(0, 11, 5),
        Triangle(0, 5, 1),
        Triangle(0, 1, 7),
        Triangle(0, 7, 10),
        Triangle(0, 10, 11),
        Triangle(1, 5, 9),
        Triangle(5, 11, 4),
        Triangle(11, 10, 2),
        Triangle(10, 7, 6),
        Triangle(7, 1, 8),
        Triangle(3, 9, 4),
        Triangle(3, 4, 2),
        Triangle(3, 2, 6),
        Triangle(3, 6, 8),
        Triangle(3, 8, 9),
        Triangle(4, 9, 5),
        Triangle(2, 4, 11),
        Triangle(6, 2, 10),
        Triangle(8, 6, 7),
        Triangle(9, 8, 1),
    ];

    Mesh::new(device, vertices, triangles)
}
