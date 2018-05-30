# Machine Learning - Exercise 4

A PyTorch implementation of a Neural Network which classifies an image to one of 10 clothing classes (Fashion MNIST).

Following are five neural network models.


## 1. Basic NN
***
### Parameters

* **Hidden layer(s):** Two hidden layers in sizes of 100 and 50.
* **Number of epochs:** 10.
* **Learning rate:** 0.01.
* **Activation function:** ReLU.
* **Optimizer:** AdaGrad

### Results

* **Training set accuracy:** 90.252%
* **Validation set accuracy:** 88.098%
* **Testing set accuracy:** 87.960%
* **Average training set loss:** 0.265
* **Average validation set loss:** 0.311
* **Average testing loss sum:** 0.339

![graph](https://github.com/aedeny/machine_learning-ex4/blob/master/Training_Loss_vs._Validation_Loss_Basic.png?raw=true)

## 2. NN With Dropout

### Parameters

* **Hidden layer(s):** Two hidden layers in sizes of 100 and 50.
* **Number of epochs:** 10.
* **Learning rate:** 0.01.
* **Activation function:** ReLU.
* **Optimizer:** AdaGrad
* **Dropout:** 0.1, 0.2, 0.25

### Results

* **Training set accuracy:** 68.452%
* **Validation set accuracy:** 88.040%
* **Testing set accuracy:** 87.010%
* **Average training set loss:** 0.913
* **Average validation set loss:** 0.445
* **Average testing loss sum:** 0.486
![graph](https://github.com/aedeny/machine_learning-ex4/blob/master/Training_Loss_vs._Validation_Loss_Dropout.png?raw=true)


## 3. NN With Batch Normalization

### Parameters

* **Hidden layer(s):** Two hidden layers in sizes of 100 and 50.
* **Number of epochs:** 10.
* **Learning rate:** 0.01.
* **Activation function:** ReLU.
* **Optimizer:** AdaGrad
* **Batch Normalization:** 
### Results

* **Training set accuracy:** 91.071%
* **Validation set accuracy:** 89.021%
* **Testing set accuracy:** 88.150%
* **Average training set loss:** 0.370
* **Average validation set loss:** 0.390
* **Average testing loss sum:** 0.420

![graph](https://github.com/aedeny/machine_learning-ex4/blob/master/Training_Loss_vs._Validation_Loss_Batch_Normalization.png?raw=true)

## 4. NN With Convolution

### Parameters

* **Hidden layer(s):** Two hidden layers in sizes of 100 and 50.
* **Number of epochs:** 10.
* **Learning rate:** 0.01.
* **Activation function:** ReLU.
* **Optimizer:** AdaGrad
* **Convolution:** Conv2d (1 * 10, 10 * 20) with kernel of size 5.

### Results

* **Training set accuracy:** 89.577%
* **Validation set accuracy:** 88.215%
* **Testing set accuracy:** 88.300%
* **Average training set loss:** 0.287
* **Average validation set loss:** 0.312
* **Average testing loss sum:** 0.325
![graph](https://github.com/aedeny/machine_learning-ex4/blob/master/Training_Loss_vs._Validation_Loss_Convolution.png?raw=true)


## 5. Combined NN

### Parameters

* **Hidden layer(s):** Two hidden layers in sizes of 100 and 50.
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

![graph](https://github.com/aedeny/machine_learning-ex4/blob/master/Training_Loss_vs._Validation_Loss_Combined.png?raw=true)
