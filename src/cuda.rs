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
        // TODO: add error handling for initialization
        self.conv_layer = &cnn.conv_layer;
        self.output_layer = &cnn.output_layer;
        
        // CODE SNIPPETS GRABBED FROM RUSTACUDA GIT
        // Load the module containing the function we want to call
        // The module is the object / compiled chunk of code that can call kernel functions
        let module_data = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&module_data)?;
        self.module = module;
        // Get the first device
        let device = Device::get_device(0)?;

        //The stream is a sequence of work actions
        // Create a stream to submit work to
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        self.stream = stream;
        // the context is the execution environment (memory segments) allocated for the kernel
            // Create a context associated to this device
        let context = Context::create_and_push(
                ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        self._context = context;

        return Ok(self)
    }
    

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        
        //create 10 cuda threads to run in parallel
        //each thread takes in a different filter 
        //filters are taken from self.conv_layer
        //preforms convolution, relu and output 
        //each neuron writes its output to a specific memory chunk
         
        let mut input_gpu = DeviceBox::from_slice(&input);


        let mut filter_gpu = &self.conv_layer;
        let mut convolution_output_gpu = DeviceBox::new(ConvOutput);
        let mut output_gpu = &self.output_layer;
        //call convolution layer
        let block_size = BlockSize::xyz(20, 20, 1);

        let grid_size = (1, 1, 10);
        unsafe{
            launch!(self.module.convolute<<<grid_size, block_size>>> (
                    input_gpu.as_device_ptr(),
                    filter_gpu.as_device_ptr(),
                    convolute_output_gpu.as_device_ptr();
                    100,
                    5,
                    20
                )

            )
        }
        stream.synchronize()?;
        //call relu layer
        let mut relu_output = DeviceBox::new(ConvOutput);

        let block_size = BlockSize::xyz(20, 20, 1);
        let grid_size = (1, 1, 10);
        unsafe(
            launch!(
                self.module.relu<<<grid_size, block_size>>> (
                    convolute_output_gpu.as_device_ptr(),
                    relu_output.as_device_ptr(),
                    20
                )
            )
        )
        stream.synchronize()?;
        let output_weights_gpu = DeviceBox::from_slice(&self.output_layer);
        let output_layer_gpu = DeviceBox::new(OutputVec);
        //call output layer
        unsafe(
            launch!(
                self.module.output<<<grid_size, block_size>>> (
                    relu_output.as_device_ptr(),
                    output_weights_gpu.as_device_ptr(),
                    output_layer_gpu.as_device_ptr(),
                    4000
                )
            )
        )
        stream.synchronize()?;

        let mut output_layer = vec![];
        output_layer_gpu.copy_to(&mut output_layer)?;
        return output_layer;
    }
}
