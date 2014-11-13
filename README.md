Caltech Lane Detection Software
===============================

This package contains C/C++ and Matlab source code that implements the work in [1]. It implements a real time lane detection system for single images by fitting robust Bezier splines. It can detect all lanes in the street, or the two lane markers of the current lane. To quickly see it in action, download the software below and also the Caltech Lanes Dataset .

The detection runs in real time, about 40-50 Hz and detects all lanes in the street. It was compiled and tested on Ubuntu Lucid Lynx 32-bit machine and Red Hat Enterprise Linux 5.5 64-bit machine.

It also includes several functionalities that were missing from OpenCV at the time, including:

Routines for obtaining the Inverse Perspective Mapping (IPM) of an image i.e. getting a bird's eye view of the road.
Routines for conversion to/from image pixel coordinates and coordinates on the road plane (using the ground plane assumption).
Robust & RANSAC line fitting.
Robust & RANSAC Bezier spline fitting.
Bezier spline rasterization and plotting.
Bresenham's line raterization.

Various utility functions for checking intersections of lines with lines and bounding boxes, checking points inside a rectangle, ... etc.
An implementation of a general Hough transform routine for lines.

[1] Mohamed Aly, Real time Detection of Lane Markers in Urban Streets, IEEE Intelligent Vehicles Symposium, Eindhoven, The Netherlands, June 2008. [pdf]
