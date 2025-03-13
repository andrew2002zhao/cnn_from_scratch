Added cuda parallel CNN implementation


Created a cuda program in rust using rustacuda to make an existing CNN implementation in parallel. Created 3 kernels to further parallelize the code. Created new structs to further handle additional data processing required for output matrix multiplication. Wrote output data to out_cuda.csv.

First kernel created took in a 100x100 input image and a 5x5x10 filters to convolute the input image. Multiplication and addition were all done in the kernel. Threads in blocks of 20x20x1 and grids of 1x1x10 were created to emulate the convolution output. Each thread would handle the output data from a single convolution. Relu from the original cpu implementaion was lumped into the kernel to speed up code output. 

The second kernel preformed matrix multiplicaiton between a weight matrices and the output from the previous convolution kernel. The convolution kernel outputs a 20x20x10 matrix and the weights matrix is a 4000x10 matrix. The flattened output of the convolution kernel is 4000x1 and the flattened weight matrix is 40000x1. The convolution kernel is matrix multipled 10 times with the weight matrix and the output is stored into an output layer type which is of size 4000x10.

The third kernel preformed summation between the outputs of the second multiplcation kernel. The final output needs to be of size 1 x 10 where the input to the addition kernel is 4000 x 10. To further use parallelism, addition is broken into 3 levels where the data is first scaled from 4000x10 to 100x10 and then to 10x10 before finally to 1x10. 

Tested the code by comparing with compare.py and saw no version differences. Initially used a dummy csv file to generate correct ouputs called in_temp.csv which was just the first line of in.csv. Used the command cargo run --release -- cuda input/cnn.csv input/input_temp.csv output/out_cuda.csv to test the code.

Tested latency by comparing timing differences between correct cuda implementaion and correct cpu implementation and observed equal to better preformance. CPU implementation is called by cargo run --release -- cpu input/cnn.csv input/input.csv output/out.csv. GPU implementation is called by cargo run --release -- cuda input/cnn.csv input/input.csv output/out_cuda.csv  Further speedup can be achieved by parallelizing convolution kernel more.

