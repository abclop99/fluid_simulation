mod camera;
mod framework;
mod mesh;
mod simulation;

fn main() {
    framework::run::<simulation::Simulation>("Fluid Simulation");
}
