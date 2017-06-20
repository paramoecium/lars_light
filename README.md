# lars_light
a light-weight implementation of Lars(least angle regression)

[Efron, Bradley, et al. "Least angle regression." The Annals of statistics 32.2 (2004): 407-499.](http://statweb.stanford.edu/~tibs/ftp/lars.pdf)

Usage
------------
```bash
mkdir build
cd build 
cmake .. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc
make
./test_lars
```

```python
INIT_CORRELATION()
while (COMPUTE_LAMBDA() > LAMBDA)
  GET_ACTIVE_IDX()
  FUSED_CHOLESKY()
  GET_A()
  GET_GAMMA()
  UPDATE_BETA()
```

Branches:

master: baseline, no optimization

opt_all: optimized with all strategies mentioned in the report
<div style="text-align:center">
<img src="https://github.com/paramoecium/lars_light/blob/master/report/pic/roofline.png" width="400">
<img src="https://github.com/paramoecium/lars_light/blob/master/report/pic/performance.png" width="440">
</div>
