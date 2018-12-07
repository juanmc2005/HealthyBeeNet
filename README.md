# Beehive Health Classification

![Bee calling in sick](https://media.treehugger.com/assets/images/2011/10/bee_calling_in_sick.jpg)

This is my final project for the Masters 2 subject Fouille de Données et Aide à la Décision at Université Paris Diderot.

I use a Kaggle dataset with annotated bee images to train different models in order to classify a beehive 
as `healthy` or `unhealthy`.

## Dataset
[Honey Bee Annotated Images](https://www.kaggle.com/jenny18/honey-bee-annotated-images)

Data about beehive deseases was compressed into 2 labels: either `healthy` (as it was before), indicating 
there are no infections in the beehive, and `unhealthy`, where the desease was mentioned in the original dataset.

5 images were removed because the quality was too low for SIFT to extract any descriptors.

## Algorithms
- Bag of Visual Words
  - Mini Batch K-Means
  - K Nearest Neighbors
  - Bernoulli Naive Bayes
  - SVM
- Convolutional Neural Network

## Train/Test Splits
- 4133 training images
- 1034 test images
- Stratification with label ratio: 65% healthy and 35% unhealthy in each split

**All hyper parameter choices were made using a validation set** 
(10-fold cross validation for BOVW and 10% of the training set for the CNN)

## CNN Architecture
![BeeNet Architecture](https://github.com/juanmc2005/bh-health-classifier/blob/master/BeeNet%20Architecture.jpg)

### Kernels
#### Conv1 RGB
<img src="https://github.com/juanmc2005/bh-health-classifier/blob/master/bee_dataset/cnn_kernels/conv1_layer_kernels_color.png" width="480">

#### Conv2 Gray Channel 0
<img src="https://github.com/juanmc2005/bh-health-classifier/blob/master/bee_dataset/cnn_kernels/conv2_layer_kernels.png" width="480">

## Bag of Visual Words
### SIFT Descriptor Dictionary
<img src="https://github.com/juanmc2005/bh-health-classifier/blob/master/bee_dataset/descriptor_dict/all.png" width="480">

## Results
### Bag of Visual Words with SVM
---------- | Healthy | Unhealthy
---------- | ------- | ---------
Precision  |   84%   |    72%
Recall     |   86%   |    69%

### Deep Convolutional Neural Network
---------- | Healthy | Unhealthy
---------- | ------- | ---------
Precision  |   93%   |    87%
Recall     |   93%   |    88%

## Libraries
- [Sci-kit Learn](https://scikit-learn.org/stable/index.html)
- [NumPy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)
