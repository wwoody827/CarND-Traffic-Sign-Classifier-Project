#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[//]: # "my References"
[img1]: ./writeup/Class23.png  "Class23"
[img2]: ./writeup/Class41.png  "Class41"
[img3]: ./writeup/distribution.png  "distribution"
[img4]: ./writeup/transfered.png  "transfered"
[img5]: ./writeup/online_test.png  "online"
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/wwoody827/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy methods rather than hardcoding results manually.

I used numpy lib to help me calculating

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.
First I displayed some images samples from dataset to have a feeling of what data I am dealing with.
Here are two images from training dataset (class No.23 and class No.41)

![alt text][img1]

![alt text][img2]


Here is an exploratory visualization of the data set. It is a plots showing how the data distribute among classes.

![alt text][img3]

The original training dataset is very unevenly distributed, ranging from 200 to 2000.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First I tried to modify LeNet. I used 32x32x3 RGB color and change output shape to 5x5. Training started smoothly and I managed to get validation accuracy around 87%. Then, I tried convert images to grayscale and found accuracy improves a lot.

I normalize all grayscale image to [-1, 1]

Soon, I found I run out of data and begin to overfit my net. Thus I also use data
augmentation. I define two helper function, with both opencv lib and skimage lib.

The first one is to rotate image by a certain degree. The second on is to perform
affine transformation to original images.

Here is an example of an original image(left) and an augmented image(right):

![alt text][img1]
![alt text][img4]

Finally, my training dataset is 7 times larger, including 2 rotation and transformation.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |            32x32x1 Grayscale image             |
| Convolution 1 5x5 | 1x1 stride, valid padding, outputs 32x32x64 |
|      ELU       |                                          |
|   Max pooling   |      2x2 stride,  outputs 14x14x16       |
| Convolution 2-1 1x1 |    1x1 stride, Same padding. 14x14x32       |
| Convolution 2-2 3x3 |    1x1 stride, Same padding. 14x14x32       |
| Convolution 2-3 5x5 |    1x1 stride, Same padding. 14x14x32       |
|   Inception    |      Concanate conv 2-1, 2-2 and 2-3 , 14x14x96           |
|      ELU       |                                          |
| Convolution 3 5x5 |    1x1 stride, valid padding. 10x10x128       |
|   Max pooling   |      2x2 stride,  outputs 5x5x128     |
| Fully connected |                   etc.                   |
|      ELU       |                                          |
| Fully connected |                   etc.                   |
|      ELU       |                                          |
| Fully connected |                   etc.                   |
|     Softmax     |                   etc.                   |

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
I added L2 regularization to loss and add dropout layers with keep prob around 0.7 to
prevent overfitting.

To train the model, I used an Adam optimizer, with a learning rate fine tuned for
best performance. Train was done on my desktop PC with GTX 1070. With augmented data
, each epochs takes around 1 min. For the results provided here, I spend around 30 mins for training due to time limitation, and model is not fully converged. But overall preformance is

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 99.0%
* test set accuracy of 97.4%

My best model results were: (Did not save model file.......)
* training set accuracy of ???? (I did not record)
* validation set accuracy of 99.4%
* test set accuracy of 98.6%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

I tried LeNet first but not get performance better than 90%. So I tried to add more
feature maps. My final model is Inception net, which I named as trafNet, with a additional convolution layer.

* What were some problems with the initial architecture?

LeNet is used for MNIST dataset, which is too simple for this problem.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.


* Which parameters were tuned? How were they adjusted and why?

Learning rate, number of feature maps in each conv layer, L2 regularization (beta)

I started with a high learning rate and check how loss reduce epochs. I increase
learning rate if it's too low and reduce learning rate if loss started to oscillate.

I tried to increase the number of feature maps while keep model size not to ridiculous and training not too slow.

I set beta to be 0.001 to avoid overfitting. I found below 0.0001, model starts to
overfit.


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The most important design choice is to use Inception module. In addition, add a l2 regularization is also important. Without l2 I cannot achieve accuracy higher than 96%.

If a well known architecture was chosen:
* What architecture was chosen?

I tried LeNet but the performance is not very well. I think this is because LeNet
is too simple, with only limited conv layers and small number of parameters.

* Why did you believe it would be relevant to the traffic sign application?

LeNet is working well on MNIST dataset and which is similar to this problem.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 17 German traffic signs that I found on the web:

![alt text][img5]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|     Image class    |  Prediction   |
| :-----------: | :-----------: |
| 3    | 3    |
| 1|1|
| 5|5|
| 1| 1|
| 11  |11  |
| 2|2|
| 14 |14 |
| 14|14|
| 12| 12|
| 12| 12|
| 11|11|
| 17|17|
| 17|33|
| 22|22|
| 22| 22|
| 40|40|
| 40| 40|


The model was able to correctly guess 16 of the 17 traffic signs, which gives an accuracy of 94%. This compares favorably to the accuracy on the test set of 97%.

The only mistake is No.13 (No entry vs. Turn right). The reason is No.13 in the image
is rotated by a large angle, and classifier cannot tell the difference between red
and blue with grayscale input.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model can be found in the Ipython notebook.

Top five softmax prob for predictions:

    No. 1 :
       Groudtruth:  3
       Predicted:  [3 1 5 0 6]
       Preb:  [ 0.56514978  0.33061674  0.0571524   0.04491277  0.001037  ]
    No. 2 :
       Groudtruth:  1
       Predicted:  [ 1  2  0 14  8]
       Preb:  [  9.98219430e-01   1.68171176e-03   8.85186673e-05   3.33303160e-06
       2.92436471e-06]
    No. 3 :
       Groudtruth:  5
       Predicted:  [5 7 1 8 0]
       Preb:  [  9.95430708e-01   3.51865147e-03   8.48291034e-04   1.79650771e-04
       1.09603243e-05]
    No. 4 :
       Groudtruth:  1
       Predicted:  [ 1  2  0  5 40]
       Preb:  [  9.97832119e-01   1.75231136e-03   1.42719800e-04   1.06285544e-04
       1.03125371e-04]
    No. 5 :
       Groudtruth:  11
       Predicted:  [11 30 23 25 40]
       Preb:  [  6.36251986e-01   3.42024505e-01   1.87294874e-02   2.43790867e-03
       3.46746092e-04]
    No. 6 :
       Groudtruth:  2
       Predicted:  [ 2  1 21  5  6]
       Preb:  [  9.99470294e-01   5.22376096e-04   3.53283735e-06   1.67089161e-06
       7.79103232e-07]
    No. 7 :
       Groudtruth:  14
       Predicted:  [14 33 40  7  6]
       Preb:  [  9.99999166e-01   5.39463031e-07   1.23893003e-07   8.03690767e-08
       3.09934904e-08]
    No. 8 :
       Groudtruth:  14
       Predicted:  [14 33  2 40  7]
       Preb:  [  9.99999881e-01   5.53785107e-08   3.20553930e-08   1.22163391e-08
       7.32830330e-09]
    No. 9 :
       Groudtruth:  12
       Predicted:  [12 14  8 24 40]
       Preb:  [ 0.67848665  0.20060389  0.07593745  0.01679298  0.01593079]
    No. 10 :
       Groudtruth:  12
       Predicted:  [12 30 38 29 24]
       Preb:  [  9.96896148e-01   1.36081688e-03   9.81835183e-04   6.34461758e-04
       5.06015240e-05]
    No. 11 :
       Groudtruth:  11
       Predicted:  [11 30 28 27 23]
       Preb:  [  9.92154181e-01   7.76467379e-03   5.30335019e-05   2.23486968e-05
       5.33222419e-06]
    No. 12 :
       Groudtruth:  17
       Predicted:  [17  7 34 16 38]
       Preb:  [  9.99983549e-01   1.64175981e-05   3.68867674e-08   2.90567055e-08
       1.23172130e-08]
    No. 13 :
       Groudtruth:  17
       Predicted:  [33 17 14 39 40]
       Preb:  [  8.36685240e-01   1.60699338e-01   9.52908536e-04   7.10331660e-04
       5.20954083e-04]
    No. 14 :
       Groudtruth:  22
       Predicted:  [22 23 26 19  9]
       Preb:  [  9.78440642e-01   1.95896029e-02   1.68543006e-03   1.34829199e-04
       1.06063060e-04]
    No. 15 :
       Groudtruth:  22
       Predicted:  [22 23 19 26 31]
       Preb:  [ 0.53501815  0.41259336  0.03430141  0.01517209  0.00150515]
    No. 16 :
       Groudtruth:  40
       Predicted:  [40 14 33  7 23]
       Preb:  [  9.99934793e-01   4.32646630e-05   9.37538061e-06   7.08246444e-06
       2.70049281e-06]
    No. 17 :
       Groudtruth:  40
       Predicted:  [40  7 14 33 36]
       Preb:  [  9.98589218e-01   6.22631109e-04   3.36213707e-04   1.68192171e-04
       6.07678448e-05]


The model is quite sure about most test images, except No. 1, 13, 15.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
