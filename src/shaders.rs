pub(crate) mod cs {
    use const_format::concatcp;

    const CELLS_X: u32 = 1024;
    const CELLS_Y: u32 = 512; 
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/ca.glsl",
        define: [("CELLS_X", "2048"), ("CELLS_Y", "1024")]
        
    }
    
}

