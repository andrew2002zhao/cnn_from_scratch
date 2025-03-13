// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        
        rustacuda::init(CudaFlags::empty())?;
    
    
        // Get the first device
        let device = Device::get_device(0)?;
        // CODE SNIPPETS GRABBED FROM RUSTACUDA GIT
        // Load the module containing the function we want to call
        // The module is the object / compiled chunk of code that can call kernel functions
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        
        let module_data = CString::new(include_str!("../kernel/kernel.ptx"))?;
        
        let module = Module::load_from_string(&module_data)?;
    
        

        //The stream is a sequence of work actions
        // Create a stream to submit work to
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        // the context is the execution environment (memory segments) allocated for the kernel
            // Create a context associated to this device
            // TODO: add error handling for initialization
        let conv_layer = DeviceBox::new(&cnn.conv_layer);
        let output_layer = DeviceBox::new(&cnn.output_layer);
        
        
            
        let cudaContext = CudaContext {
            conv_layer: conv_layer.unwrap(),
            output_layer: output_layer.unwrap(),
            module,
            stream,
            _context: context,
        };
    

        return Ok(cudaContext)
    }
    

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        
        let module = &self.module;
        let stream = &self.stream;
        //create 10 cuda threads to run in parallel
        //each thread takes in a different filter 
        //filters are taken from self.conv_layer
        //preforms convolution, relu and output 
        //each neuron writes its output to a specific memory chunk
         
        let mut input_gpu = DeviceBox::new(input)?;
        
        // 
        let mut convolute_output_cpu = ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]);
        let mut convolute_output_gpu = DeviceBox::new(&convolute_output_cpu)?;
        // call convolution layer
// 
        // let block_size = BlockSize::xyz(20, 20, 1);
        // let grid_size = (1, 1, 10);
        
        let grid_size = (1, 1, 10); // (16, 16, 10)
        let block_size = BlockSize::xyz(20, 20, 1); // Match output_dim
        
        unsafe{
            launch!(module.convolute<<<grid_size, &block_size, 0, stream>>> (
                    input_gpu.as_device_ptr(),
                    self.conv_layer.as_device_ptr(),
                    convolute_output_gpu.as_device_ptr(),
                    100,
                    5,
                    20,
                    4000
                )
            )?
        }
        
        stream.synchronize()?;
    
        

        let block_size = 4;
        let grid_size = 40000 / block_size;

        let mut output_mul_cpu = OutputLayer([[0.0; OUT_NEURON_DIM]; OUT_LAYER_SIZE]);
        let mut output_mul_gpu = DeviceBox::new(&output_mul_cpu)?;
        
        //call output layer
        unsafe{
            launch!(
                module.output_mul<<<grid_size, block_size, 0, stream>>> (
                    convolute_output_gpu.as_device_ptr(),
                    self.output_layer.as_device_ptr(),
                    output_mul_gpu.as_device_ptr(),
                    40000,
                    4000
                )
            )?
        }
        stream.synchronize()?;


        let mut output_step_1_cpu = OutputInt1Vec([0.0; OUT_INT_1_SIZE]);
        let mut output_step_1_gpu = DeviceBox::new(&output_step_1_cpu)?;

        let block_size = 256;
        let grid_size = (1000 + 256 - 1) / 256;
        
    
        unsafe{
            launch!(
                module.output_add<<<grid_size, block_size, 0, stream>>> (
                    output_mul_gpu.as_device_ptr(),
                    output_step_1_gpu.as_device_ptr(),
                    1000,
                    40
                )
            )?
        }
        stream.synchronize()?;
        let block_size = 256;
        let grid_size = 1;

        let mut output_step_2_cpu = OutputInt2Vec([0.0; OUT_INT_2_SIZE]);
        let mut output_step_2_gpu = DeviceBox::new(&output_step_2_cpu)?;

        unsafe{
            launch!(
                module.output_add<<<grid_size, block_size, 0, stream>>> (
                    output_step_1_gpu.as_device_ptr(),
                    output_step_2_gpu.as_device_ptr(),
                    100,
                    10
                )
            )?
        }
        stream.synchronize()?;


        let mut output_layer_cpu = OutputVec([0.0; OUT_LAYER_SIZE]);
        let mut output_layer_gpu = DeviceBox::new(&output_layer_cpu)?;

        
        unsafe{
            launch!(
                module.output_add<<<grid_size, block_size, 0, stream>>> (
                    output_step_2_gpu.as_device_ptr(),
                    output_layer_gpu.as_device_ptr(),
                    10,
                    10
                )
            )?
        }
        stream.synchronize()?;

        let mut output_layer = OutputVec([0.0; OUT_LAYER_SIZE]);
        output_layer_gpu.copy_to(&mut output_layer)?;
        return Ok(output_layer);
    }
}
