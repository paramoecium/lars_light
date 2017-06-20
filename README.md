# lars_light
a light-weight implementation of LARS (least angle regression)

Reference:
[Efron, Bradley, et al. "Least angle regression." The Annals of statistics 32.2 (2004): 407-499.](http://statweb.stanford.edu/~tibs/ftp/lars.pdf)

Branch
------------
master: baseline, no optimization

opt_all: optimized with all strategies mentioned in the report with regard to the hardware architecture of Intel i7-6700 CPU, achieving 7.2x speed-up

Usage
------------
```bash
mkdir build
cd build 
cmake .. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc
make
./test_lars
```
Before/After Optimization
------------
<div style="text-align:center">
<img src="https://github.com/paramoecium/lars_light/blob/master/report/pic/roofline.png" width="400">
<img src="https://github.com/paramoecium/lars_light/blob/master/report/pic/performance.png" width="440">
</div>
