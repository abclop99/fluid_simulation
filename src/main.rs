mod camera;
mod framework;
mod lighting;
mod mesh;
mod simulation;
mod texture;

fn main() {
    framework::run::<simulation::Simulation>("Fluid Simulation");
}
