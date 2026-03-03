This todo list is for humans, not Claude.

- h2d slowness
  - when system is fully loaded it is really slow
  - h2d reported latency is suspiciously close to our "expected wait time" for a the execute_v2() call on the other thread to finish
  - we suspect we are waiting on a global lock while execute_v2 happens
  - proposed fix:
    - Create Per-Worker Streams
      - err, self._stream = cu.cuStreamCreate(0)
      - _check(err, "cuStreamCreate")
    - Use Async Memory Copies
      - _check(
          cu.cuMemcpyHtoDAsync(
              self._inputs[0].device,
              self._inputs[0]._host_ptr,
              self._inputs[0]._nbytes,
              self._stream  # Use the worker's private stream
          ),
          "cuMemcpyHtoDAsync"
      )
    - Use execute_async_v3 (or v2)
      - ok = self._context.execute_async_v3(stream_handle=int(self._stream))
    - Explicitly Synchronize
      - Since the calls are now async, you must wait for the stream to finish before reading the output on the host.
      - cu.cuStreamSynchronize(self._stream)
    - Follow up: Multi-Process Service (MPS)
      - We should explore how efficiently we can use the GPU on the dev machine, which is a fairly powerful 2080ti
      - We should create a load test script to send a ton of requests to a high concurrency condor config to max it out
      - if we believe we are not fully utilizing the gpu, or if CPU usage seems excessive, we can explore nvidia MPS



- finish apply fixes to TUI
  - logos and flying birds
  - make graphs fill up remaining space after logo bars and metrics panels drawn
  
  - top banner: concurrent/rps are redundant
  - metrics:
    - make global consistent with worker metrics
    - for trt that means:
      - e2e, h2d, semwait, infer, d2h, postp
    - always 5 second rolling avg
    - metrics go to 0 if not present, do not hide
    - remove some of those horizontal dividers if we need extra lines (we do)
    - worker: don't need to show sparate req/inf request count. just inf req is fine
    - concurrent: better to show avg over the window? might be hard to sample that frequently
      - i.e. if max is 1, then at 50% duty cycle conc would be at 0.5
      - but don't do this if the sampling is very expensive.

  - other elements
    - put in CONDOR bar:
      - condor version
      - hostname
      - memory host, cpu (process) usage host
    - put in TRT bar:
      - cuda device
      - memory gpu, proc usage gpu (by process)





- allow semaphore around H2D transfer
  - does it improve throughput?
