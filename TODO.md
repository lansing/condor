
- more tui improvements
  - other elements
    - put in CONDOR bar:
      - condor version
      - hostname
      - memory host, cpu (process) usage host
    - put in TRT bar:
      - cuda device
      - memory gpu, proc usage gpu (by process)


- export metrics to hyperdx
  - get hyperdx on HA machine? 
  
  
- load tester / optimizer
  - b2b inf requests per thread
  - search for max throughput
    - +1 num workers, benchmark, till max throughput is reached
  
- work on onnx backend
  - cuda, openvino, what else?
  - ensure working
  - metrics / tui
  
- work on openvino backend
