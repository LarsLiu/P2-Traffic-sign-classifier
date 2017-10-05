**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---

[//]: # (Image References)

[image1]: ./image-report/visualization "Visualization"
[image2]: ./image-report/hist.png "histogram"
[image3]: ./image-report/preprocess_image "preprocessing"
[image4]: ./image-report/accuracy_display "accuracy"
[image5]: ./image-report/new_image1 "Traffic Sign"
[image6]: ./image-report/new_image2 "Traffic Sign"
[image7]: ./image-report/new_image3 "Traffic Sign"
[image8]: ./image-report/new_image4 "Traffic Sign"
[image9]: ./image-report/new_image5 "Traffic Sign"
[image10]: ./image-report/new_image_test1 "Traffic Sign"
[image11]: ./image-report/new_image_test2 "Traffic Sign"
[image12]: ./image-report/new_image_test3 "Traffic Sign"
[image13]: ./image-report/new_image_test4 "Traffic Sign"
[image14]: ./image-report/new_image_test5 "Traffic Sign"
[image15]: ./image-report/1.png "result"
[image16]: ./image-report/2.png "result"
[image17]: ./image-report/3.png "result"
[image18]: ./image-report/4.png "result"
[image19]: ./image-report/5.png "result"

###Data Set Summary & Exploration

I used the python method **len()**  and **X.train[0].shape** function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an visualization of random 10 traffic-sign samples of the data set. The title is description of that image. 

![alt text][image1]

Here is an histogram of the numbers of each sign

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

I decided not to convert the images to grayscale because the disemsion of input is not  that high that will largely improve training time, and this way we will keep color features.

First step: shuffle all training set to eliminate the effect caused by the training set's order.

Second step, I normalized the image data because it accelerates the convergence of the model.

Here is an example of an original image and an augmented image:

![alt text][image3]
 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32
| Dropout	      	| keep prob: 90%				|
| Convolution  5x5 	    | 1x1 stride, valid padding, outputs 10x10x64
   | RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32
| Dropout	      	| keep prob: 90%				|    									|
| Fully connected		|512 outputs 									|
| RELU				|      
| Dropout	      	| keep prob: 90%				|
| Fully connected		|128 outputs   									|
| RELU				|      
| Dropout	      	| keep prob: 90%				|
| Fully connected(lOGITS)		|43 outputs       

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model: 

* Lost calculation: cross entropy
* Optimizer: AdamOptimzer
* Patch size: 512
* Number of epochs: 40
* Learning rate: 0.001

Network parameter:

* Dropout: 0.9
* Padding: VALID

![alt text][image4]

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.00000
* validation set accuracy of 0.94331
* test set accuracy of 0.96033

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
    LeNet-5, this could provide valid train accuracy and the depth of each 
    layer need to adjust

* What were some problems with the initial architecture?
     Train accuracy is good but the validation accuracy is pool, it is caused by overfitting.
     
 * How was the architecture adjusted and why was it adjusted? 
 	add dropout layer after activate function of each layer
 
* Which parameters were tuned? How were they adjusted and why?
set initial learning rate is 0.0001, which is too low beucause the accuracy improved too slow, change that to 0.0005, which is much better.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9] 

#####Something difficulty: 

Because none of theses images have a standard square dimensions, which means after resize the image to 32*32, which may cause shape change of current traffic sign, which may cause difficulty to classify. Also, the fourth image includes two signs in same window, which may cause classifier confused. Lasty, all these images have watermark, which may add some noisy for analyzing.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 30km/h 		| Speed limit 80km/h  									| 
|     keep right			| No passing 										|
| Pedestrians			|Road work Yield											|
|No entry	      		|  Speed limit  70 km/h				 				|
| Stop			| Pedestrians  							|

![alt text][image10] 
![alt text][image11] 
![alt text][image12] 
![alt text][image13] 
![alt text][image14] 

The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0%. This compares to the accuracy on the test set is terrible. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 72th cell of the Ipython notebook.

![alt text][image15] 
![alt text][image16] 
![alt text][image17] 
![alt text][image18] 
![alt text][image19] 

As result we can see, for image 1, 4, 5. although the prediction is wrong but still the classifier is very sure about the result it predict. 
