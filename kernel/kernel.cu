// Very minimal skeleton for the kernel

#include <stdio.h>

extern "C" __global__ void convolve (
  const float * input,
  const float * filter,
  float * convolute_output,
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


  float sum = 0;
  for(int fy = 0; fy < filter_width; fy++) {
    for(int fx = 0; fx < filter.width; fx++) {
      // input dimensions are 
      // 100 x 100 x 1
      // from offset - offset + 24 where offset is the number of threads and there are offsets from 0 - 399, 
      // 

      // filter dimensions are
      // 5 x 5 x 1
      // from 0 - 24 for a square matrix
      // fx * filter_width + fy


      int offset = fx + fy * filter_width;

      int thread_number = y * convolulte_width + x;
      int input_index = (thread_number * filter_width * filter_width) + offset;

      int filter_index = z * (filter_width * filter_width) + offset;
      sum += input[input_index] * filter[filter_index];
    }
  }

  int convolute_index = (convolulte_width * convolute_width) * z + (convolute_width * y) + x
  convolute_output[convolute_index] = sum;

}

extern "C" __global__ void relu (
  const float * convolute_output,
  float * relu_output,
  int convolute_width,
  int output_width
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  for(int i = 0; i < convolute_width; i++) {
    for(int j = 0; j < convolute_width; j++) {
      int index = z * (convolulte_width * convolulte_width) + y * (convolulte_width) + x;
      if(convolute_output[index] < 0) {
        relu_output[index] = 0;
      }
      else{
        relu_output[index] = convolute_output[index];
      }
      
    }
  }
}


extern "C" __global__ void output (
  const float * relu_output,
  const float * weights,
  float * output,
  int flatten_width
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  float sum = 0;
  for(int i = 0; i < flatten_width; i++) {
    int weights_index = (flatten_width) * z + i; 
    sum += weights[weight_index] * relu_output[i];
  }

  output[z] = sum;

}