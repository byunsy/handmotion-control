# handmotion-control

## Description

Handmotion-control allows users to actively interact with a given camera to control the brightness and/or the contrast values of each video frame in real time. By waving their hands up and down in front of a camera, users can increase (wave up) or decrease (wave down) the contrast value, and by waving their hands left and right, users can also increase (wave right) or decrease (wave left) the brightness value. Users can reset these values anytime using the space bar.

The objective of handmotion-control project was to explore motion estimation with optical flow. Simply put, optical flow can be described as the relative motion of objects between two consecutive frames, most generally in a video. It has two main variants--sparse and dense optical flow, both of which has its own advantages and disadvantages.

Sparse optical flow selects a few significant features, like edges and corners, and track its velocity vectors (magnitude and direction). The method is relatively quick and efficient as it identifies several interesting features within each frame. This is useful for scenarios where one wishes to track a specific object of interest. Dense optical flow, on the other hand, computes velocity vectors for every single pixel within each frame. While this method is obviously more computationally expensive compared to sparse optical flow, it offers a more accurate and detailed flow result. Since handmotion-project involves changing image values/properties based on the direction of a motion, I wanted to maximize the accuracy of optical flow at the cost of making the program a little slower. Hence, I decided to use the Farneback method, one of the most popular implementations of dense optical flow.

## Installation

I used the OpenCV package for python (version 4.1.0.25 or above) with Python 3.7.2

```bash
pip install opencv-python==4.1.0.25
```

## Usage

Clone the handmotion-control repository in your directory.

```bash
git clone https://github.com/byunsy/handmotion-control.git
```

Move to your specific directory and execute the program.

```bash
python handmotion.py
```

## Demonstrations

![](images/handmotion-demo.gif)

## References

I read [Introduction to Motion Estimation with Optical Flow](https://nanonets.com/blog/optical-flow/) by Chuan-en Lin to understand more in-depth about sparse and dense optical flows.
