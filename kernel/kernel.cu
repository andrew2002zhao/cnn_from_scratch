// Very minimal skeleton for the kernel

#include <stdio.h>

extern "C" __global__ void convolute (
  const double * input,
  const double * filter,
  double * convolute_output,
  int input_width,
  int filter_width,
  int convolute_width
) {

  //dot product between filter and input

  //take output sum and put into an intermediate layer

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  

  //need to determine which thread im on
  // there should be 20 x 20 threads per each output node
  // the input is 100 x 100 x 1
  // this turns into a flat 10000 size vector
  // there should be 20 x 20, 5x5 filters and 20x20 5x5 areas
  // so x, y should give an integer from 0 - 399 together
  //


  double sum = 0;
  for(int fy = 0; fy < filter_width; fy++) {
    for(int fx = 0; fx < filter_width; fx++) {
      // input dimensions are 
      // 100 x 100 x 1
      // from offset - offset + 24 where offset is the number of threads and there are offsets from 0 - 399, 
      // 

      // filter dimensions are
      // 5 x 5 x 1
      // from 0 - 24 for a square matrix
      // fx * filter_width + fy


      int offset = fx + fy * filter_width;
      int convolute_offset = fx + fy * input_width;
      //  500  * y + 5 * x
      int thread_position = y * input_width * filter_width + x * filter_width;
    
      int input_index = thread_position + convolute_offset;


      //input for thread 1 should be 
      //  0 -   4 
      //100 - 104
      //200 - 204
      //300 - 304
      //400 - 404

      //input for thread 2 should be
      //  5 -   9
      //105 - 109
      //205 - 209
      //305 - 309
      //405 - 409
      
      //input for thread 21 sohuld be 
      // 500 - 504
      // 600 - 604
      // 700 - 704
      // 800 - 804
      // 900 - 904
      
      int filter_index = z * (filter_width * filter_width) + offset;
      // printf("input_index %d input_value %f \n" , input_index , input[input_index]);
    
      sum += input[input_index] * filter[filter_index];
      // printf("x: %d y: %d z: %d, fx: %d, fy: %d input_array_position: %d filter_position: %d \n", x, y, z, fx, fy, input_index, filter_index);
    }
  }

  int convolute_index = (convolute_width * convolute_width) * z + (convolute_width * y) + x;
  convolute_output[convolute_index] = sum;
  // printf("x: %d y: %d z: %d convolute_output_position %d \n", x, y, z, convolute_index);
  

}

extern "C" __global__ void relu (
  const double * convolute_output,
  double * relu_output,
  int convolute_width
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;


  int index = z * (convolute_width * convolute_width) + y * (convolute_width) + x;
  if(convolute_output[index] < 0) {
    relu_output[index] = 0;
  }
  else{
    relu_output[index] = convolute_output[index];
  }
      
    
  
}


extern "C" __global__ void output (
  const double * relu_output,
  const double * weights,
  double * output,
  int flatten_width
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  double sum = 0;
  for(int i = 0; i < flatten_width; i++) {
    int weight_index = (flatten_width) * z + i; 
    sum += weights[weight_index] * relu_output[i];
  }

  output[z] = sum;

}