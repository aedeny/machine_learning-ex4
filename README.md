# Machine Learning - Exercise 4

A PyTorch implementation of a Neural Network which classifies an image to one of 10 clothing classes (Fashion MNIST).

## Explanation

### Parameters

* **Hidden layer(s):** One layer of 100 neurons.
* **Number of epochs:** 10.
* **Learning rate:** 0.01.
* **Activation function:** ReLU.
* **Optimizer:** AdaGrad
* **Dropout:** 0.1, 0.2, 0.25
* **Convolution:** Conv2d (1 * 10, 10 * 20) with kernel of size 5.

### Results

* **Training set accuracy:** 68.594%
* **Validation set accuracy:** 89.254%
* **Testing set accuracy:** 89.090%
* **Average training set loss:** 0.472
* **Average validation set loss:** 0.466
* **Average loss sum:** 0.918

### Graph

Training Loss vs. Validation Loss


![graph](https://github.com/aedeny/machine_learning-ex4/blob/master/Training_Loss_vs._Validation_Loss.png?raw=true)

## Example Output  
**Note:** Output can vary from each training.
```
Training Epoch: 0	Accuracy 29176/48000 (60.783%)	Average Loss: 1.264
Validation Epoch: 0	Accuracy: 10103/12032 (83.968%)	Average Loss: 0.786

Training Epoch: 1	Accuracy 31546/48000 (65.721%)	Average Loss: 1.068
Validation Epoch: 1	Accuracy: 10469/12032 (87.010%)	Average Loss: 0.646

Training Epoch: 2	Accuracy 31947/48000 (66.556%)	Average Loss: 1.015
Validation Epoch: 2	Accuracy: 10531/12032 (87.525%)	Average Loss: 0.589

Training Epoch: 3	Accuracy 32294/48000 (67.279%)	Average Loss: 0.983
Validation Epoch: 3	Accuracy: 10589/12032 (88.007%)	Average Loss: 0.552

Training Epoch: 4	Accuracy 32554/48000 (67.821%)	Average Loss: 0.960
Validation Epoch: 4	Accuracy: 10638/12032 (88.414%)	Average Loss: 0.519

Training Epoch: 5	Accuracy 32525/48000 (67.760%)	Average Loss: 0.949
Validation Epoch: 5	Accuracy: 10684/12032 (88.797%)	Average Loss: 0.507

Training Epoch: 6	Accuracy 32691/48000 (68.106%)	Average Loss: 0.937
Validation Epoch: 6	Accuracy: 10665/12032 (88.639%)	Average Loss: 0.490

Training Epoch: 7	Accuracy 32920/48000 (68.583%)	Average Loss: 0.922
Validation Epoch: 7	Accuracy: 10715/12032 (89.054%)	Average Loss: 0.476

Training Epoch: 8	Accuracy 32863/48000 (68.465%)	Average Loss: 0.924
Validation Epoch: 8	Accuracy: 10719/12032 (89.087%)	Average Loss: 0.468

Training Epoch: 9	Accuracy 32925/48000 (68.594%)	Average Loss: 0.918
Validation Epoch: 9	Accuracy: 10739/12032 (89.254%)	Average Loss: 0.466

Testing Set: Average Loss: 0.4722, Accuracy: 8909/10000 (89%)
```