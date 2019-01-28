# Co-Training Images

Implementation of the paper [Combining Labeled and Unlabeled Data with Co-Training](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf) for images.

## Introduction

Co-Training is a machine-learning algorithm that is proposed by Blum and Mitchell [1]. It can be used when a small portion of a dataset is labeled. The original work used the Co-Training algorithm for classifying web-pages. This project considers the problem of image classification on CIFAR-10 dataset using Co-Training.

## Usage

Clone the repository and run the main python file.

```
$ python main.py
```

## Co-Training Algorithm

![Alt text](/figs/algo.png?raw=true "Algorithm")  

Above figure is taken from the original paper [1]. This project uses the same algorithm.

## Experiment Results

```
Parameters used for Experiment 1
Initial labeled dataset size: 4000
Pool size: 1000
Positive/Negative Examples: 100
```
![Alt text](/figs/exp1.png?raw=true "Experiment 1")  


```
Parameters used for Experiment 2
Initial labeled dataset size: 12000
Pool size: 1000
Positive/Negative Examples: 100
```
![Alt text](/figs/exp2.png?raw=true "Experiment 2")  

```
Parameters used for Experiment 3
Initial labeled dataset size: 40000
Pool size: 1000
Positive/Negative Examples: 100
```
![Alt text](/figs/exp3.png?raw=true "Experiment 3")  


### References

[1] Avrim Blum and Tom Mitchell. 1998. Combining labeled and unlabeled data with co-training. In Proceedings of the Eleventh Annual Conference on Computational Learning Theory, COLT’ 98, pages 92–100, New York, NY, USA. ACM. https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf
