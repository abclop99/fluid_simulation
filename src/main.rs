mod framework;
mod simulation;

fn main() {
    framework::run::<simulation::Simulation>("Fluid Simulation");
}
