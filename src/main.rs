pub mod shaders;

extern crate nalgebra_glm as glm;

use std::{env, future};
use std::sync::Arc;
use std::time::SystemTime;

use image::{ImageBuffer, Rgba};
use rand::Rng;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::shader::{EntryPoint, ShaderModule};
use vulkano::{VulkanLibrary, Version, shader, Validated, VulkanError, library, command_buffer, DeviceSize};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo, BlitImageInfo, CopyBufferInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet, DescriptorBufferInfo};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceCreateFlags};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator, DeviceLayout};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, PresentMode, SwapchainPresentInfo, acquire_next_image};
use vulkano::sync::{self, GpuFuture};
use winit::dpi::PhysicalPosition;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};

fn main() {
    println!("Hello, world!");

    env::set_var("RUST_BACKTRACE", "1");

    let event_loop = EventLoop::new();

    let vk_library = VulkanLibrary::new().unwrap();

    /* Get extensions required by display */
    let required_extensions = Surface::required_extensions(&event_loop);

    let instance = Instance::new(
        vk_library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        }
    ).unwrap();

    /* Set up window (winit) & surface (vulkan) */
    let window: Arc<Window> = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface: Arc<Surface> = Surface::from_window(instance.clone(), window.clone()).unwrap();


    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };


    let (device, queue) = select_device(instance, device_extensions, &surface);

    /* Get our allocators */
    let memory_allocator = 
        Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let (mut swapchain, mut swapchain_images) = create_swapchain(device.clone(), &surface, &window);

    
    let compute_pipeline = create_compute_pipeline(device.clone());

    let align = device
        .physical_device()
        .properties()
        .min_storage_buffer_offset_alignment
        .as_devicesize();

    println!("needed alignment {align}");

    let mut cells_x: u32 = 32;
    let mut cells_y: u32 = 32;

    let size = cells_x * cells_y * 2;
    /* Get our double buffers for our CA. */

    /* 10 by 10 */
    let buffer = Buffer::new_slice::<u32>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: 
                BufferUsage::TRANSFER_SRC |
                BufferUsage::TRANSFER_DST | 
                BufferUsage::STORAGE_BUFFER,
            
            ..Default::default()
        },
        AllocationCreateInfo { 
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default() 
        },
        (size) as DeviceSize // TODO: Why don't we need alignment here? 
        //DeviceLayout::from_size_alignment(size.into(), align).unwrap(),
    ).unwrap();

    let (a, b) = buffer.split_at(((size) / 2).into());
    let mut compute_buffers = [a, b];

    /* This seems correct, I wonder  */
    println!("{}, {}", compute_buffers[0].size(), compute_buffers[1].size());

    /* Get our staging buffer for setting up the initial state.  */
    let initial = rand_grid(memory_allocator.clone(), [cells_x, cells_y]);

    /* Copy over initial state. TODO:  How do we make this interactive?   */

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator, 
        queue.queue_family_index(), 
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .copy_buffer(
            CopyBufferInfo::buffers(initial, compute_buffers[0].clone())
        )
        .unwrap();  

    let command_buffer = builder.build().unwrap();

    let init_future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    init_future.wait(None).unwrap();

    println!("CA Initialized");
    
    let mut mouse_pos: PhysicalPosition<f64> = PhysicalPosition::default();

    // Swapchain Stuff:
    let mut recreate_swapchain = false;
    
    // Create a (maybe need one per rame in flight? ) temporary image to use in our shaders, 
    let out_image = Image::new(
        memory_allocator.clone(), 
        ImageCreateInfo { 
            image_type: ImageType::Dim2d, 
            format: Format::R8G8B8A8_UNORM, 
            extent: [cells_x, cells_y, 1], /* TODO: Screen dimensions */
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();
    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    // For now, tie the compute and display into the same loop / time step. Only run update every 4 cycles or so. 

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { device_id, position, modifiers },
                ..
            } => {
                mouse_pos = position;
            }
            Event::RedrawEventsCleared => {
                // Don't draw when the screen size is zero.
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();



                if recreate_swapchain {
                    let (new_swapchain, new_images) = swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    swapchain = new_swapchain;

                    swapchain_images = new_images;

                    recreate_swapchain = false;
                }

                /* Get an image from the swapchain */
                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };
                

                

                // `acquire_next_image` can be successful, but suboptimal. This means that the
                // swapchain image will still work, but it may not display correctly. With some
                // drivers this can be when the window resizes, but it may not cause the swapchain
                // to become out of date.
                if suboptimal {
                    recreate_swapchain = true;
                }

                /* Draw onto  */

                // Do one step of GOL. 
                let view = ImageView::new_default(out_image.clone()).unwrap();

                let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();


                let set: Arc<PersistentDescriptorSet> = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    layout.clone(),
                    [
                        WriteDescriptorSet::image_view(0, view),  // 0 is the binding
                        WriteDescriptorSet::buffer(1, compute_buffers[0].clone()),
                        WriteDescriptorSet::buffer(2, compute_buffers[1].clone()),
                    ], 
                    [],
                )
                .expect("Invalid descriptor set");

                /* Start Executing + Draw */

                let mut builder = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator, 
                    queue.queue_family_index(), 
                    CommandBufferUsage::OneTimeSubmit,
                ).unwrap();

                builder
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute, 
                        compute_pipeline.layout().clone(), 
                        0, 
                        set
                    )
                    .unwrap()
                    .dispatch([cells_x / 16, cells_y / 16, 1]) /* 16 * 16 * 1 */
                    .unwrap()
                    /* This seems like it should NOT be part of the same pipeline, 
                    once we decouple computation and display we can think about it more. */
                    .blit_image( 
                        BlitImageInfo::images(out_image.clone(), swapchain_images[image_index as usize].clone())
                    )
                    .unwrap();
                

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(), 
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();
                

                

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }

                compute_buffers.swap(0, 1);

                
            }
            _ => (),
        }
    })

}

fn rand_grid(memory_allocator: Arc<StandardMemoryAllocator>, size: [u32; 2]) -> Subbuffer<[u32]> {
    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (0..(size[0] * size[1])).map(|_| rand::thread_rng().gen_range(0u32..=1)),
    )
    .unwrap()
}


fn create_swapchain(device: Arc<Device>, surface: &Arc<Surface>, window: &Arc<Window>) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let (swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
    
        let image_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()
            [0].0;
        
        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count.max(2),
                present_mode: PresentMode::Fifo,

                image_format: image_format,
                image_extent: window.inner_size().into(),

                image_usage:
                    ImageUsage::COLOR_ATTACHMENT |
                    ImageUsage::TRANSFER_DST, // So we can blit onto it
                
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),

                ..Default::default()
            }
        ).expect("failed to create swapchain")
    };
    (swapchain, images)
}

fn select_device(
    instance: Arc<Instance>, 
    mut device_extensions: DeviceExtensions, 
    surface: &Arc<Surface>,
) -> (Arc<Device>, Arc<Queue>) {

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
        })
        .filter(|p| {
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {

            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::COMPUTE)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                // The code here searches for the first queue family that is suitable. If none is
                // found, `None` is returned to `filter_map`, which disqualifies this physical
                // device.
                .map(|i| (p, i as u32))
        })

        .min_by_key(|(p, _)| {
            // We assign a lower score to device types that are likely to be faster/better.
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
    .expect("no suitable physical device found");

    if physical_device.api_version() < Version::V1_3 {
        device_extensions.khr_dynamic_rendering = true;
    }

    // Some little debug infos.
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    if physical_device.api_version() < Version::V1_3 {
        device_extensions.khr_dynamic_rendering = true;
    }

    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],

            enabled_extensions: device_extensions,

            enabled_features: Features {
                shader_float64: true,
                dynamic_rendering: true,
                ..Features::empty()
            },

            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();
    (device, queue)
}

fn create_compute_pipeline(device: Arc<Device>) -> Arc<ComputePipeline> {
    let shader: Arc<ShaderModule> = shaders::cs::load(device.clone())
        .expect("failed to create shader module");

    let entry_point: EntryPoint = shader.entry_point("main").unwrap();

    let stage = PipelineShaderStageCreateInfo::new(entry_point);

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .expect("failed to create compute pipeline");

    compute_pipeline
}

