# CustomKNN Project TODO

## 1. Performance Optimization
- [ ] Implement batch prediction functionality (`analyze_batch`) in `CustomKNN`
- [ ] Vectorize pairwise distance matrix calculations using NumPy broadcasting
- [ ] Replace full distance sorting with `np.argpartition` for top K elements
- [ ] Benchmark bulk test execution time to ensure 200 samples run under 2 seconds

## 2. Model Accuracy & Class Balance
- [ ] Implement feature scaling (MinMax or Standard Normalization) before distance calculations
- [ ] Rebalance dataset or adjust distance thresholds to reduce False Positive rate (currently 51/200)
- [ ] Experiment with optimal $K$ values using cross-validation
- [ ] Implement distance-weighted voting mechanism

## 3. Code Refactoring & Testing
- [ ] Update `run_bulk_test()` to call `analyze_batch()` instead of looping individual rows
- [ ] Add execution timer metrics for dataset loading, initialization, and evaluation
- [ ] Add explicit validation for missing features in incoming input data