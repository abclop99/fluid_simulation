use crate::framework::Application;

/// The fluid simulation.
pub struct Simulation {
    // TODO
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
        _config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        // TODO
        Self {}
    }

    fn resize(
        &mut self,
        _config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        // TODO
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        // TODO
    }

    fn render(&mut self, _view: &wgpu::TextureView, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        // TODO
    }
}
