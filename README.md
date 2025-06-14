# pcx
`pcx` is a library for performing fft, written in C++23.  
Currently only one-dimensional complex to complex fft is supported.  
Transforms over contiguous or parallel data are supportted.  
Higher dimensional transforms can be perfromed by combining multiple one-dimensional transforms.  
Currently only 32-bit and 64-bit floating point numbers are supported.  
Currently only modern x86 platforms are supported (AVX2 or AVX512).  

## FFT
There are two classes that perform fft: `pcx::fft_plan` and `pcx::par_fft_plan`.
`pcx::fft_plan` is for perfroming fft over contiguous complex data.  
`pcx::par_fft_plan` is for perfroming fft over parallel complex data.  

An example of fast convolution:
```c++
    auto plan = pcx::fft_plan<float>(1024);
    auto data = std::span<const std::complex<float>>{};
    auto result = std::vector<std::complex<flaot>>(1024);
    // acquire data
    plan.fft(result, data);
    // multiply `result` by convulution kernel
    plan.ifft(result);
```



## Design
`pcx` intially stands for `packed complex`, a represntation of complex data where real and imaginary 
values are stored in interleaved packs. `std::complex` can be viewed as packed complex data with pack size of 1.
This representation is better suited for simd processing.
Currently `pcx` APIs use `std::complex`, but the goal in the future is to provide a collection of containers/wrappers/views 
with user-provided pack size, as well as instruments to perfrom common mathematical operations over the data.

`pcx` fft implementation is designed to be very flexible through heavy use of templates.
The goal is to adapt the library to all common platforms with simd. 
`std::experimental::simd` from `libstdc++` and C++26 `std::simd` are planned to be added as backends for simd implementation.

The end goal is to provide a number of utilities for processing of complex data with main application being processing of 
radio data.

##  
- implement radix-4 based nodes
- implement better size-alignment policy
