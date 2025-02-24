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
        
        // Load the module containing the function we want to call
        let module_data = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&module_data)?;
        self.module = module;
        // Get the first device
        let device = Device::get_device(0)?;

          // Create a stream to submit work to
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        self.stream = stream;

            // Create a context associated to this device
        let context = Context::create_and_push(
                ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        self._context = context;

        return Ok(self)
    }
    pub fn convolution_layer(input: &InputMatrix, conv_filters: &ConvLayer, outputs: &mut ConvOutput){

    }
    pub fn relu_layer(conv_out: &mut ConvOutput){

    }
    pub fn output_layer(input: &ConvOutput, weights: &OutputLayer, output: &mut OutputVec){

    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        
        let mut conv_output = ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]);
        let mut output = OutputVec([0.0; OUT_LAYER_SIZE]);
    }
}
