Deep neural network for pixel-wise classification of satellite image
====================================================================
Overview
--------------------------------------------------------------------
This is my work during research assistant position at Geospatial Laboratory of Ohio State University. Construct a deep neural network for satellite image segmentation task:
1.input: satellite image and DSM height map
2.output: predicted category for each pixel
Implement with two deep learning frameworks: Caffe and PyTorch

Demo
---------------------------------------------------------------------
![given satellite image](https://raw.githubusercontent.com/cyanBuckeye/SatelliteImageProcessing/master/demo/demo1/test-img.jpg "given satellite image")
Given satellite image
![output labelmap](https://raw.githubusercontent.com/cyanBuckeye/SatelliteImageProcessing/master/demo/demo1/predict-label.jpg "output labelmap")
Output labelmap, in which: blue->buildings, cyan->low vegetation, blue->tree, white->road, yellow->car 
