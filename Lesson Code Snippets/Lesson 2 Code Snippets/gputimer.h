#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

/*
A problem with using host-device synchronization points, such as cudaDeviceSynchronize(), is that they stall the GPU pipeline.
For this reason, CUDA offers a relatively light-weight alternative to CPU timers via the CUDA event API.
The CUDA event API includes calls to create and destroy events, record events, and compute the elapsed time in milliseconds between two recorded events.

CUDA events make use of the concept of CUDA streams.

CUDA events are of type cudaEvent_t and are created and destroyed with cudaEventCreate() and cudaEventDestroy().
cudaEventRecord() places the start and stop events into the default stream, stream 0 unless specified.
The device will record a time stamp for the event when it reaches that event in the stream. 
The function cudaEventSynchronize() blocks CPU execution until the specified event is recorded.
The cudaEventElapsedTime() function returns in the first argument the number of milliseconds time elapsed between the recording of start and stop.
This value has a resolution of approximately one half microsecond
*/

struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;
 
      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }
 
      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }
 
      void Start()
      {
            cudaEventRecord(start, 0);
      }
 
      void Stop()
      {
            cudaEventRecord(stop, 0);
      }
 
      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};

#endif  /* __GPU_TIMER_H__ */
