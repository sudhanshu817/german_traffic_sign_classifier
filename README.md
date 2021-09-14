# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./bins.jpg "Visualization"
[image2]: ./grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./web_images/1.png "Traffic Sign 1"
[image5]: ./web_images/3.png "Traffic Sign 2"
[image6]: ./web_images/4.jpg "Traffic Sign 3"
[image7]: ./web_images/7.png "Traffic Sign 4"
[image8]: ./web_images/8.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing the distribution of various classes in the training data

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the grayscale version of traffic signals contain the same information as an RGB image. (i.e. the same meaning can be conveyed but using less space as greyscale images have only 1 channel)

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

As a last step, I normalized the image data by subtracting the mean and dividing by the standard deviation as it becomes easy for the neural net to find the global minima if the data is normalized.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                                    | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x32 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 				|
|Flatten                |                                               |
|Dropout                |Keep_prob 0.75                                 |
|Dense                  |outputs 120                                    |
|RELU                   |                                               |
|Dropout                |Keep_prob 0.7                                  |
|Dense                  |outputs 128                                    |
|RELU                   |                                               |
|Dropout                |Keep_prob 0.6                                  |
|Dense                  |outputs 43                                     |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batchsize of 128 and trained it for 20 epochs.
The learning rate was set to 0.01 and I used the Adam optimizer to train the network.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.932 
* test set accuracy of 0.922

The first architecture consisted of 2 convolution blocks, no dropout and the activation was set to sigmoid. All the rest part of the model was same.

This model was underfitting the data with very low accuracy of 5-6% on training set, so I added one more convolution block.
However, this didn't solev the problem so I changed the activation to RELU.

This resulted in a better model, but this new model was over-fitting on the training set with a accuracy of 97% and validation accuracy was 85%.

So to avoid overfitting I added dropout. The initial dropout keep probabilities were higher but I wasn't able to reach accuracy of 93% on validation set, so I reduced the keep probabilities of the dropout layer to prevent over-fitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image might be difficult to classify because it also contains a background which seems like a house.
The fifth image also has a wood stand on which the road sign was put-up and hence seems like difficult to classify.
The rest of the images clearly only include the traffic sign and hance should be easier to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road narrows on right | Road narrows on right     					|
| Speed limit 20km/h    | No passing     								|
| Road work     		| Road work 					 				|
| Stop Sign      		| Stop sign   									| 
| Children crossing  	| Speed limit 60km/h   							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in cell 21-33 of the Ipython notebook.

For the first image, the model is relatively sure that this is a Road narrows to the right sign (probability > 0.95), and the image does contain the same sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.58965480e-01		| Road narrows to the right						| 
| 4.10342775e-02		| Dangerous curve to the left					|
| 2.53116866e-07		| Go straight or left           				|
| 5.07119235e-10		| Double curve					 				|
| 3.08033503e-12        | Turn right ahead  							|


For the second image the model gives wrong prediction that it's a no passing sign with a probability of 0.657, the top 5 predictions do not contain the right sign,

| Probability           |      Prediction                               |
|:---------------------:|:---------------------------------------------:| 
| 6.57833457e-01        | No passing                                    |
| 3.40238929e-01        | Vehicles over 3.5 metric tons prohibited      |
| 1.45470921e-03        | No entry                                      |
| 3.26585810e-04        | Priority road                                 |
| 1.46332648e-04        | Turn left ahead                               |

For the third image model accurately predicts the road work sign with a probability almost equal to 1.
The top 5 softmax probabilities are:
         
| Probability           |      Prediction                               |
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | Road work                                     |
| 2.02205548e-12        | Bumpy road                                    |
| 7.03417156e-15        | Yield                                         |
| 1.13748309e-15        | Keep right                                    |
| 8.65764551e-17        | Speed limit (70km/h)                          |
 

Again in the fourth image, the model predicts the stop sign which is indeed right with a probability of 0.9991.
The top 5 probabilities are:

| Probability           |      Prediction                               |
|:---------------------:|:---------------------------------------------:| 
| 9.99180317e-01        | Stop                                          |
| 3.84017592e-04        | Turn right ahead                              |
| 1.60878946e-04        | Right-of-way at the next intersection         |
| 1.50524414e-04        | Turn left ahead                               |
| 1.24268059e-04        | Beware of ice/snow                            |
         
Model mis-classifies the 5th image as speed limit sign with a probability of 0.8245. The correct label isn't present in the top-5 predictions. The next top 5 probabilities are:

| Probability           |      Prediction                               |
|:---------------------:|:---------------------------------------------:| 
| 8.24519932e-01        | Speed limit (60km/h)                          |
| 9.76502970e-02        | Ahead only                                    |
| 7.54698142e-02        | Right-of-way at the next intersection         |
| 1.87816680e-03        | Beware of ice/snow                            |
| 4.81722178e-04        | Priority road                                 |
         
We can see for the 5 test images, when the model is right, the probability of the right label is very high, whereas where its wrong, even the top-5 probabiltities do not contain the right label.
















