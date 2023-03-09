use crate::texture::Texture;
use std::time::{Duration, Instant};
use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

/**
 * Adapted from WGPU examples, but with the wasm32 target and other
 * non-essential code removed.
 * */

#[allow(dead_code)]
pub fn cast_slice<T>(data: &[T]) -> &[u8] {
    use std::{mem::size_of, slice::from_raw_parts};

    unsafe { from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>()) }
}

#[allow(dead_code)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// A trait that defines the interface for an application.
/// The framework will call the methods on this trait to run the application.
pub trait Application: 'static + Sized {
    /// Defines the optional features that the application requires. They will be enabled if the
    /// adapter supports them.
    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    /// Defines the required features for the application. They will be enabled even if the adapter
    /// does not support them.
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    /// Returns the required downlevel capabilities that the application requires.
    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::empty(),
            shader_model: wgpu::ShaderModel::Sm5,
            ..wgpu::DownlevelCapabilities::default()
        }
    }

    /// Returns the limits that the application requires.
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_webgl2_defaults() // These downlevel limits will allow the code to run on all possible hardware
    }

    /// Constructs an initial instance of the application.
    fn init(
        config: &wgpu::SurfaceConfiguration,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self;

    /// Called on WindowEvent::Resized events.
    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        scale_factor: f64,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_state: &mut EguiState,
    );

    /// Called for any WindowEvent that is not handled by the framework.
    fn handle_event(&mut self, event: WindowEvent);

    /// Called every frame.
    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_state: &mut EguiState,
        window: &winit::window::Window, // egui_winit needs this
        timestep: Duration,
    );
}

pub struct EguiState {
    pub ctx: egui::Context,
    pub renderer: egui_wgpu::renderer::Renderer,
    pub screen_descriptor: egui_wgpu::renderer::ScreenDescriptor,
    pub winit_state: egui_winit::State,
}

struct Setup {
    window: winit::window::Window,
    event_loop: EventLoop<()>,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn setup<E: Application>(title: &str) -> Setup {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    };

    let event_loop = EventLoop::new();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title(title);
    let window = builder.build(&event_loop).unwrap();

    log::info!("Initializing the surface...");

    let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        dx12_shader_compiler,
    });
    let (size, surface) = unsafe {
        let size = window.inner_size();

        let surface = instance.create_surface(&window).unwrap();

        (size, surface)
    };
    let adapter =
        wgpu::util::initialize_adapter_from_env_or_default(&instance, backends, Some(&surface))
            .await
            .expect("No suitable GPU adapters found on the system!");

    let adapter_info = adapter.get_info();
    println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

    let optional_features = E::optional_features();
    let required_features = E::required_features();
    let adapter_features = adapter.features();
    assert!(
        adapter_features.contains(required_features),
        "Adapter does not support required features for this program: {:?}",
        required_features - adapter_features
    );

    let required_downlevel_capabilities = E::required_downlevel_capabilities();
    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    assert!(
        downlevel_capabilities.shader_model >= required_downlevel_capabilities.shader_model,
        "Adapter does not support the minimum shader model required to run this program: {:?}",
        required_downlevel_capabilities.shader_model
    );
    assert!(
        downlevel_capabilities
            .flags
            .contains(required_downlevel_capabilities.flags),
        "Adapter does not support the downlevel capabilities required to run this program: {:?}",
        required_downlevel_capabilities.flags - downlevel_capabilities.flags
    );

    // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
    let needed_limits = E::required_limits().using_resolution(adapter.limits());

    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: (optional_features & adapter_features) | required_features,
                limits: needed_limits,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }
}

fn start<E: Application>(
    #[cfg(not(target_arch = "wasm32"))] Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }: Setup,
    #[cfg(target_arch = "wasm32")] Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
        offscreen_canvas_setup,
    }: Setup,
) {
    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .expect("Surface isn't supported by the adapter.");

    let surface_caps = surface.get_capabilities(&adapter);

    // Change the format to BgraUnorm if it is supported.
    // This format is the only one supported by Wayland on Nvidia.
    if surface_caps
        .formats
        .contains(&wgpu::TextureFormat::Bgra8Unorm)
    {
        config.format = wgpu::TextureFormat::Bgra8Unorm;
    }

    // Change the present mode to AutoVsync. It is supported on all platforms.
    config.present_mode = wgpu::PresentMode::AutoVsync;

    surface.configure(&device, &config);

    log::info!("Initializing the program...");
    let mut program = E::init(&config, &adapter, &device, &queue);

    let mut egui_state = EguiState {
        ctx: egui::Context::default(),
        renderer: egui_wgpu::renderer::Renderer::new(
            &device,
            config.format,
            Some(Texture::DEPTH_FORMAT),
            1,
        ),
        screen_descriptor: egui_wgpu::renderer::ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: window.scale_factor() as f32,
        },
        winit_state: egui_winit::State::new(&event_loop),
    };

    let mut last_frame_inst = Instant::now();
    let (mut frame_count, mut accum_time) = (0, 0.0);

    log::info!("Entering render loop...");
    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter); // force ownership by the closure
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::RedrawEventsCleared => {
                window.request_redraw();
            }
            event::Event::WindowEvent {
                event:
                    WindowEvent::Resized(size)
                    | WindowEvent::ScaleFactorChanged {
                        new_inner_size: &mut size,
                        ..
                    },
                ..
            } => {
                log::info!("Resizing to {:?}", size);
                config.width = size.width.max(1);
                config.height = size.height.max(1);
                program.resize(
                    &config,
                    window.scale_factor(),
                    &device,
                    &queue,
                    &mut egui_state,
                );
                surface.configure(&device, &config);
            }
            event::Event::WindowEvent { event, .. } => {
                let response = egui_state.winit_state.on_event(&egui_state.ctx, &event);

                // Don't pass event on if it was consumed by egui
                if let egui_winit::EventResponse {
                    consumed: false, ..
                } = response
                {
                    match event {
                        WindowEvent::KeyboardInput {
                            input:
                                event::KeyboardInput {
                                    virtual_keycode: Some(event::VirtualKeyCode::Escape),
                                    state: event::ElementState::Pressed,
                                    ..
                                },
                            ..
                        }
                        | WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit;
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                event::KeyboardInput {
                                    virtual_keycode: Some(event::VirtualKeyCode::R),
                                    state: event::ElementState::Pressed,
                                    ..
                                },
                            ..
                        } => {
                            println!("{:#?}", instance.generate_report());
                        }
                        _ => {
                            let response = egui_state.winit_state.on_event(&egui_state.ctx, &event);

                            if let egui_winit::EventResponse { consumed: true, .. } = response {
                                // Egui consumed the event, so we don't need to pass it to the program.
                            } else {
                                program.handle_event(event);
                            }
                        }
                    }
                }
            }
            event::Event::RedrawRequested(_) => {
                let timestep = last_frame_inst.elapsed();

                {
                    accum_time += timestep.as_secs_f32();
                    last_frame_inst = Instant::now();
                    frame_count += 1;
                    if frame_count >= 1000 && accum_time > 5.0 {
                        println!(
                            "Avg frame time {}ms, timestep {}ms, {} frames / {}ms = {} FPS",
                            accum_time * 1000.0 / frame_count as f32,
                            timestep.as_secs_f32() * 1000.0,
                            frame_count,
                            accum_time * 1000.0,
                            frame_count as f32 / accum_time,
                        );
                        accum_time = 0.0;
                        frame_count = 0;
                    }
                }

                let frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => {
                        surface.configure(&device, &config);
                        surface
                            .get_current_texture()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                program.render(&view, &device, &queue, &mut egui_state, &window, timestep);

                frame.present();
            }
            _ => {}
        }
    });
}

/// Runs the application.
/// Usage:
/// ```
/// framework::run::<MyApplication>("My Title");
/// ```
pub fn run<E: Application>(title: &str) {
    let setup = pollster::block_on(setup::<E>(title));
    start::<E>(setup);
}
