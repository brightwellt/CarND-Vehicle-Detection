##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/01_car_not_car.PNG
[image2]: ./output_images/02_HOG.PNG
[image3]: ./output_images/03_sliding.PNG
[image4]: ./output_images/04_boxes.PNG
[image5]: ./output_images/bboxes_heat01.PNG
[image5b]: ./output_images/bboxes_heat02.PNG
[image6]: ./output_images/06_labelHeatmap.PNG
[image7]: ./output_images/07_challenge.PNG
[video1]: ./output_images/project_video_output.mp4
[video2]: ./output_images/project_video_output_lanes.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

###Project Files
- Code for this solution is contained within the Project5-VehicleDetection.ipynb Jupyter notebook.
- the output_images folder contains the images generated for this writeup and the output video.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first couple of code cells of the IPython notebook. The functions used to build lists of images, extract information and generate HOG features are here. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pix_per_cell=16` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, running images through the classifier to assess accuracy. Of the color spaces I explored, YUV and YCrCb seemed particularly effective. I settled on the former. I also paid attention to the speed of operation. Increasing pix_per_cell to 16 helped with this without compromising accuracy. This can also be seen in the HOG features, where the edges are clearly being identified.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The workbook has a number of section headers. The classifier was trained in the cell marked "Classifier Training", using the extract_features functions from Cell 1.
I trained a linear SVM using the following steps:
- load car and not-car images
- extract hog features from images
- extract spatial features from images
- extract color histogram features from images
- stack and normalise features
- split the data into training and test sets(*)
- create a linear svc
- fit the data to it.
- check the accuracy of the fitted data.

(*)The images were randomly sorted into training and test sets. One way I could improve the trainer would be to manually sort the data. The data provided had some near-identical images of cars; ideally I should ensure that identical images end up in the same set (training or test). 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for the  sliding window search is in the "Strip Searching" section. There are also a couple of functions in the "Lesson Functions section too.
I search in a number of strips going down the image. Overlapping each window by 50% with its neighbours seemed effective.
I used 3 search windows. A small search window strip (roughly 42x42) near the horizon, and then two more strips (55x55, 96x96) down towards the bonnet of the car. The area above the horizon was not searched.
I also explored clipping the left and right sides of the image, but ultimately decided to search the whole width of the image. Masking the left side would mean a cleaner final video. However leaving it in - and therefore detecting cars coming the other way seemed like a legitimate thing to do.

Here's the windows I used:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The find_cars function in the "Lesson functions" section of the notebook contains the code for searching images. I adapted this function from the lesson code to take in a collection of searches to perform. I also looked at making use of the multiprocessing library, and HOGDescriptors as per a couple of forum tips.
With more time I would look at tuning the SVC's C parameter, or doing some negative mining.

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

I used deque from collections to build a rolling buffer of heat maps. These were summed together before applying a threshold. For the video, I set-up a MyVideoProcessor class to house the heatmap buffer and update it from frame to frame.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]
![alt text][image5b]

### Here is the output of the integrated heatmap from all six frames and the labels bounding box drawn onto the last image in the series:
![alt text][image6]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Using of more training data is probably the simplest thing I could do. The search area can miss a small segment on the right hand side, which is where new cars are likely to appear. This should be corrected. Testing on more videoes with differing lighting conditions, road types and vehicles would also be helpful. For example, the search area is unlikely to pick up lorries given the size of windows I am using. 

Similarly, it would be use to test different conditions; especially with denser traffic.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

- manually sort the test / training data to avoid having near-identical images in both test and training sets.
- augment the test / training data with additional images.
- look at tuning the SVC's C parameter, or doing some negative mining.
- The window search does not cover the entire image; notably the right and bottom sides can sometimes be missing a window. I could tweak the search algorithm to fit one last set of windows against the those edges hand edge.
- Make use of the multiprocessing library, and HOGDescriptors to speed things up. Using an approach other than SVM may also prove faster.
- some of the bounding boxes are larger than the car, meaning that when the box surrounding the car is larger than it needs to be. It would be nice to spend more time exploring this space so as to better capture the car shape.

### Bonus challenge
I then took the code generated from my previous exercise - Advance Lane Finding - and used this to add lanes to the video.
[link to my video result](./output_videos/project_video_output_lanes.mp4)

Here's a sample frame:
![alt text][image7]
