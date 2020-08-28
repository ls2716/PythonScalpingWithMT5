# Conclusions of the Python Scalping with MT5 project.

1. The algorithm could achieve profits when trained on 1 minute tick data and evaluated on 1 minute tick data (obviously with train/test separation). This included the 1m tick spread.

2. When evaluated on real-time data, the algorithm was generating loss. After inspeciton, the spread in real-time tick data is different that the recorder spread in 1 minute data. The ratio of spread was about 1.5 i.e. the real time spread to 1 minute spread. This means using a MT4 platform has its serious drawbacks and the mismatch is discouraging.

3. The spread of real time data prevents any profits gained from the accuracy of the algorithm.

4. Scalping is very difficult.

### Scalping is not a good idea.
