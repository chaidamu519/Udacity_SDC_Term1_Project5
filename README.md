# Udacity_SDC_Term1_Project5

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: https://github.com/chaidamu519/Udacity_SDC_Term1_Project5/blob/master/car_nocar.png
[image2]: https://github.com/chaidamu519/Udacity_SDC_Term1_Project5/blob/master/multi_scale.png
[image3]: https://github.com/chaidamu519/Udacity_SDC_Term1_Project5/blob/master/heat_map.png


## Training Support Vector Machine (SVM) model

The code for training is provided [here](https://github.com/chaidamu519/Udacity_SDC_Term1_Project5/blob/master/SVM_Training.ipynb). Both color classification and Histogram of oriented gradients are used in the model. I used the dataset provided for this project.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image1]


All the images in this project are read using `cv2.imread` and the resulted format is BGR with the intensity value between 0 and 255.

### Color features

I extracted the spatially binned color and color histograms. For spatially binned color, the resolution is used as hyperparameters for tuning. For color histograms, the number of bins are tuned to improve the svm model. The final `resolution` for spatially binned color is `16` and the number of bins is `32`.

### Histogram of Oriented Gradients (HOG) features

To extract the HOG features, I used the `hog` method from skimage library. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I tried various combinations of parameters and compare the resulted accuracy of test dataset. The `YCrCb` `colorspace`is found to provide the best accuracy. `9` `orientations` are used for HOG, with `8` `pixels_per_cell` and `4` `cells_per_block`.

### Save the model 
The svm model can be found [here](https://github.com/chaidamu519/Udacity_SDC_Term1_Project5/blob/master/svc.p)
The X_scaler can be found [here](https://github.com/chaidamu519/Udacity_SDC_Term1_Project5/blob/master/X_scaler.p), which is used to normalize the feature in the test.

## The main pipeline of vehicle tracking
The code of the main pipeline is provided [here](https://github.com/chaidamu519/Udacity_SDC_Term1_Project5/blob/master/main_pipeline.ipynb).


### Image pipeline

#### Hog Sub-sampling Window Search
The search for the cars, I used the Hog Sub-sampling Window Search method that is discussed in the course. The HOG feastures are extract once from the entire image. Then, for each of a small set of predetermined window sizes (defined by a scale argument), the HOG features can be sub-sampled to get all of its overlaying windows. 

#### Multi-scale Windows
Then, I used the the approach of multi-scale windows to deal with cars at positions in the images. For vehicles far from the camera (approximately in the middle of the image), I used small windows with a `scale` of `1`, a `cells_per_step` of `1`. On the other hand, for the image part close to the camera, larger windows with larger steps are used. 

#### Draw boxes
Then, I used the `cv2.renctangle` function to draw boxes with different scales on the image. And the results are shwon here:

![alt text][image2]

#### Heat map
Since there are certain false positives on the images. I used the approach of heat map from these detections in order to combine overlapping detections and remove false positives. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The results are shwon here:
![alt text][image3]

### Video pipeline
The video pipeline is similar to the image pipeline. The treshold on single image is used to remove false positives that have shown fewer overlapping detections. In addition, a class is defined to record the previous `heat_map`, which is used to remove false positives that occur in one frame but not in the following. The `threshold` is lineraly proportional to the lenth of recent `heat_map`.
Here's a [link to my video result](https://github.com/chaidamu519/Udacity_SDC_Term1_Project5/blob/master/project_video_out.mp4


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The results are very sensitive to the light conditions and the position as well as the size of the search windows. The overlapping windows on distant cars are less compared with close cars. Therefore, I change the `cells_per_step` for the middle part of the image to be `1` in order to increase the sensitivity. However, this increases the time for the processing of each frame.
To improve the performance, I think we can use the method of convolutional neural network. In this way, the relatively slow sliding window search will be not needed and the speed and accuracy can then be improved significantly.

