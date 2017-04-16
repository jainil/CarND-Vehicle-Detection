## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/image_3.png
[image4]: ./output_images/image_4.png
[image4a]: ./output_images/image_4a.png
[image4b]: ./output_images/image_4b.png
[image5]: ./output_images/image_5.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters with and tried fitting a classifier to
predict the class of the image based on the parameter combinations. Of the
combinations I tried, I was getting the best results with using the 'YCrCb' color
space, 9 orientation directions, 8x8 cells and a block size of (2,2).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I generated features for the images with the above HOG transform and using color  histograms with a bin size of 32 (code cell 5). Then I scaled the features and trained a linear SVC using these(code cell 6). The classifier has an accuracy of 0.947 on the test set.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search based on the `find_car` function developed in one of the lessons(code cell 9). It implements a hog sub-sampling optimization that prevents re computation of the hog transform features for a given image for different sliding windows. I decided to search at scales of 1.5 and 2 and instead of overlap I decided to go with the step size method because it is more intuitive. I used a step size of 16px ( which corresponds to a 75% overlap).

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb Y-channel HOG features and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image4a]
![alt text][image4b]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps, labels and resulting bounding boxes
![alt text][image5]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I used HOG transform and color histograms from a labeled dataset of cars and noncars to train a Support Vector Classifier to detect cars. I then used this SVC and applied it on sliding windows from larger images(or video frames) of the entire road. To optimize this process, I calculated the HOG transform of the input image only once and then sub-sampled it for matching the particular sliding window under consideration. Also in order to avoid duplicate detections and false positives, I used the heatmap technique (and applied a threshold over the last six frames when used in a video).

The most critical piece of the pipeline is the SVC classifier and the pipeline is likely to fail in the cases where the classifier may fail:
1. Night time!
2. Cars in drastically different orientations than those in the training set (front or sideways)
3. Weather conditions that affect the car image like fog, rain or sleet.

Hence, the most important thing to make the pipeline more robust would be to make the SVC more robust and capable of accurately classifying cars in more varied conditions. Additionaly, more work can be done to make the sliding windows more intelligent ( for e.g sliding window sizes and search locations could be based on estimates of distance on the image and expected size of vehicles at that distance).
