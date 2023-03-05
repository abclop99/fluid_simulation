mod camera;
mod framework;
mod lighting;
mod mesh;
mod simulation;

fn main() {
    framework::run::<simulation::Simulation>("Fluid Simulation");
}
