# Embedded Random Forest

This repositotry contains a reference implementation for "A memory Representation of Random Forests Optimized for Resource-Limited Embedded Devices".

## How to use this repo

This repository contains several Rust libraries and directories:

* The Random Forest classification/regression reference implementation is in `embedded-rforest/`.
* `forest-optimizer` is used to transform raw RF text files into an optimized RF memory representation
* `bench-data` contains the pre-trained Random Forests used in the benchmarks, both in raw text format (`*.csv`) and optimized, binary format (`*.rforest`).
* `datasets` contains the datasets used to train the random forests. Note that the `iris` dataset is available by default with an `R` distribution. The `skydive` dataset is separated by individual skydiving jump.

## How to generate an optimized memory representation of Random Forest

Run

```sh
cargo run --bin optimize_forest --input [input_file] --output [output_file] --problem-type {classification|regression}
```

## Different optimizations for different needs

The memory model used to represent a random forest as described in the paper can be fined-tuned to optimize for different needs. This repo has different branches showcasing some optimization tradeoffs which can be made to either speed up predictions, reduce RAM usage or reduce total forest size.

|Branch|Optimization|Outcome|Tradeoff|
|-|-|-|
|`unsafe-max-speed`|Disable array bounds checking|Faster predictions|Undefined behavior if forest is malformed|
|`small-classification`|Use 16-bit node pointers and  8-bit split index|Smaller forests and RAM usage|Reduced max number of nodes in forest and max number of features|
|-|Reduce size of `votes` data structure in classification `predict` method|Significantly reduced RAM usage|Reduced max number of classes in forest|