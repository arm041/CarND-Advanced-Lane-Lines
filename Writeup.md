## Writeup 
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chess_undistort.png "Undistorted"
[image2]: ./output_images/test_undistort.png "Road Transformed"
[image3]: ./output_images/test_filtered.png "Binary Example"
[image4]: ./output_images/test_birdeye.png "Warp Example"
[image5]: ./output_images/filtered_birdeye.png "Warp Example1"
[image6]: ./output_images/original_lanes_detected.png "Fit Visual"
[image7]:  ./output_images/complete_pipeline.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

 The [link to my code](https://github.com/arm041/CarND-Advanced-Lane-Lines/blob/master/examples/example.ipynb) 

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

In this section the complete pipeline that takes an image as in input and outputs the original image with the found lane lines as output, is described. First step is to undistort the images taken by the camera. 

To demonstrate this step, an image of the test images provided in the project is taken and the `cal_undistort` function is called in this picture. Because the camera was calibrated is the same that all the images are taken with it this function can be used to undistort any images taken by this camera. The result is shown below on one of the test images: 

![alt text][image2]

Now that the image is undistorted, this image should be changed such that the lane lines can be identified as easily as possible. For this I used a combination of color and gradient threshold and also the gradient direction threshold. This can be seen in the 4th code snippet of the `example.ipynb` file. Here the functions `abs_sobel_thresh`, `mage_thresh`, `dir_thresh`, and `hls_select` are defined that calculate the sobel in x/y direction the magnitude and direction of the gradient and also the HLS version of the image and apply some thresholds to obtain the best result possible. 
To demonstrate this step again an example image is taken and all the filters all applied to it and the result is as bellow: 

![alt text][image3]

So now the transformed image has all kept all the best features that were required to find the lane lines and detect them. What has to be done now is to change the perspective of the image such that we can see it from above, called the bird-eye-view. This has the benefit that the lane lines will look parallel and can be fitted to polynomials and this makes the lane finding more robust specially in case of turns that one of the lines looks more curving than the other line. 
The code for my perspective transform includes a function called `TransformToBirdeyeView()`, which appears in the 5th code snippet in the `example.ipynp` file. for this I used fixed source and destination points called `src` and `dst` in the images and get the conversion matrix by calling the function `cv2.getPerspectiveTransformation()` which receives the source and destination points as input and returns the matrix. After that I change the perspective of the image with the function `cv2.warpPerspective()`. This function gets the image and the conversion matrix and changes the image to the other perspective, which in this case is the bird-eye-view perspective. For the transformation i chosed `src` and `dst` in following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 70), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4 - 100), 0],
    [(img_size[0] / 4 - 100), img_size[1]],
    [(img_size[0] * 3 / 4 + 100), img_size[1]],
    [(img_size[0] * 3 / 4 + 100), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 220, 0        | 
| 203, 720      | 220, 720      |
| 1127, 720     | 1060, 720      |
| 710, 460      | 1060, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image as follows:

![alt text][image4]


Also an example on a filtered image can be seen below: 

![alt text][image5]

After this stage we obtain an image that is seen from above and the lane lines are as clear as possible. So now there is the need to detect and find the lane lines. 

For detecting the lane lines a function called `laneDetection()` is defined which takes the image and fits two second order polynomials to the two lines on the left and right side lines of the lane. This function uses the histogram approach for finding the lane lines. What is done is that the only the lower half of the image is considered and then the histogram will be taken along the x-axis. Where these histograms have their peaks, means that there were more white signs in the image that correspond to the lane lines that we detected through good filtering and perspective transform, and hence there is where the lines start. This starting point at the end of the image is used as a starting point and with a sliding window approach we detect all the other parts of the line that have candidate points of the lane lines inside them. The code for this part is in the 7th code snippet of the `example.ipynb`. This function returns the points that constitute the lane line. These points are then used in a function called `laneCurvature()`. This function calls the `laneDetection()` function and then uses the point to fit a polynomial to these points that build the lane lines. the polynomials fitted to the image will look like the following: 

![alt text][image6]

Now that we have the polynomials representing the lane lines, the curvature of the lane and also the distance of the car from the middle of the lane lines can be calculated. The curvature has a formula which we can use to calculate the curvature. This formula is given in the lectures of the project and below is the code that I have written to calculate it: 
```python
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5)/ np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
```
This is the curvature in pixels. But we want here the curvature measure in meters such that it can be used in the real physical world. For this a transformation is required. This transformation has to scale the pixels in the image back to meter values. For this we assume that every 720 pixels in y direction correspond to 30 meters and every 700 pixels in x direction are equal to 3.7 meters in the same direction. This conversion is represented in python as following: 
```python
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
after this we change all the points we found to meter and fit the polynomial again as follows: 
``` python
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
```

Also to calculate the distance of the car to the center of the lane line the following code is used: 
```python
offset_right = (right_fit[0] * img.shape[0] ** 2) + (right_fit[1] * img.shape[0]) + right_fit[2]
offset_left = (left_fit[0] * img.shape[0] ** 2) + (left_fit[1] * img.shape[0]) + left_fit[2]
distance_to_center = (offset_right + offset_left)/2.0 - 640
distance_to_center = distance_to_center * 3.7 / 700 #changing the distance to m
```
in this code the `offset_right` and `offset_left` values represent the x values of the fitted polynomials in the bottom end of the image. this tells us exactly where the lane lines are where the car camera is also mounted. With the assumption that the camera is mounted at the middle of the car if take the average of the offsets and reduce the center of the image from it the relative position of the car will be calculated. If the value is positive the car is a little left of the center, otherwise the car is a right óf the center. The last line of the code changes this values back to meter with the conversion that was already explained. 

Now it's time to use all the found information and show them in the original image that was fed into the pipeline. For this all the transformation that we had should be reversed to obtain the lane positions on the original image. For this the function `backTransformation()` is defined in the second last snippet of `example.ipynb` code. 
An example output of this function which takes as input one of the test images is as follows: 

![alt text][image7]

---

### Pipeline (video)

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

In my implementation I am finding the lane lines and all the curvatures in the same frame, so I believe that in frames that one or both of the lane lines are not detected, for example in lighting conditions or sharp turns, then I will have problems detecting the lanes and calculating all the things. This can be solved by using a line class as discussed in the stand out project submission. 
Also another problem that the pipeline can encounter is when the place of the camera changes on the car, due to small changes in the location of the camera on the car. 

