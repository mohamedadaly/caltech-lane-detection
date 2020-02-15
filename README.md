Caltech Lane Detection Software
===============================

This package contains C/C++ and Matlab source code that implements the work in
[1]. It implements a real time lane detection system for single images by
fitting robust Bezier splines. It can detect all lanes in the street, or the
two lane markers of the current lane. To quickly see it in action, download the
software below and also the Caltech Lanes Dataset .

The detection runs in real time, about 40-50 Hz and detects all lanes in the
street. It was compiled and tested on Ubuntu Lucid Lynx 32-bit machine and Red
Hat Enterprise Linux 5.5 64-bit machine.

It also includes several functionalities that were missing from OpenCV at the
time, including:

* Routines for obtaining the Inverse Perspective Mapping (IPM) of an image i.e.
  getting a bird's eye view of the road.
* Routines for conversion to/from image pixel coordinates and coordinates on
  the road plane (using the ground plane assumption).
* Robust & RANSAC line fitting.
* Robust & RANSAC Bezier spline fitting.
* Bezier spline rasterization and plotting.
* Bresenham's line raterization.

Various utility functions for checking intersections of lines with lines and
bounding boxes, checking points inside a rectangle, ... etc. An implementation
of a general Hough transform routine for lines.

[1] Mohamed Aly, Real time Detection of Lane Markers in Urban Streets, IEEE
Intelligent Vehicles Symposium, Eindhoven, The Netherlands, June 2008. [pdf]


## Contents

src/: contains the C/C++ source files  
|- `CameraInfo.conf`: contains the camera calibration ifo  
|- `CameraInfoOpt.*`: contain gengetopt files for parsing the camera info files  
|- `cmdline.*`: contains gengetopt files for parsing command lines  
|- `InversePerspectiveMapping.*`: code for obtainig the IPM of an image  
|- `LaneDetector.*`: code for the bulk of the algorithm, including Hough
   Transforms, Spline fitting, Post processing, ...  
|- `Lanes.conf`: the typical configuration file for lane detection  
|- `main.*`: code for the main binary  
|- `Makefile`: the Make file  
|- `mcv.*`: contain utility functions  
|- `ranker.h`: code for obtaining the median of a vector  
|- `run.sh`: Shell script for running the detector on the four clips in
    Caltech Lanes Dataset

matlab/: contains the Matlab source files  
|- `ccvCheckMergeSplines.m`: checks if two splines are matching  
|- `ccvEvalBezSpline.m`: returns points on a spline given its control points  
|- `ccvGetLaneDetectionStats.m`: computes stats from detections and ground truth  
|- `ccvLabel.m`: handles the ground truth labels  
|- `ccvReadLaneDetectionResultsFile.m`: reads a detection file output from the
   binary file LaneDetector32/64  
|- `Stats.m`: computes stats for the detections on the Caltech Lanes Dataset and
   its ground truth labels

## Prerequisites

1. OpenCV 3.4 or higher https://opencv.org/
2. (Optional) Gengetopt http://www.gnu.org/software/gengetopt/

## Compiling

Unzip the archive somewhere, let's say `~/lane-detector`:

```bash
unzip lane-detector.zip -d ~/lane-detector
cd ~/lane-detector/src
make release
```

This will generate LaneDetector32 or LaneDetector64 depending on your system.

## Caltech Lanes Dataset

To view the lane detector in action, you can download the Caltech Lanes Dataset
available at http://www.vision.caltech.edu/malaa/datasets/caltech-lanes

## Running

To run the detector on the Caltech Lanes dataset, which might be in
~/caltech-lanes/

```bash
cd ~/lane-detector/
ln -s ~/caltech-lanes/  clips
cd ~/lane-detector/src/
bash run.sh
```

This will create the results files inside
`~/caltech-lanes/*/list.txt_results.txt`

To view the statistics of the results, open Matlab and run the file:

```bash
cd ~/lane-detector/matlab/
matlab&
>>Stats
```

## Command line options

```text
LinePerceptor 1.0

Detects lanes in street images.

Usage: LinePerceptor [OPTIONS]... [FILES]...

  -h, --help                   Print help and exit
  -V, --version                Print version and exit

Basic options:
      --lanes-conf=STRING      Configuration file for lane detection
                                 (default=`Lanes.conf')
      --stoplines-conf=STRING  Configuration file for stopline detection
                                 (default=`StopLines.conf')
      --no-stoplines           Don't detect stop lines  (default=on)
      --no-lanes               Don't detect lanes  (default=off)
      --camera-conf=STRING     Configuration file for the camera paramters
                                 (default=`CameraInfo.conf')
      --list-file=STRING       Text file containing a list of images one per
                                 line
      --list-path=STRING       Path where the image files are located, this is
                                 just appended at the front of each line in
                                 --list-file  (default=`')
      --image-file=STRING      The path to an image

Debugging options:
      --wait=INT               Number of milliseconds to show the detected
                                 lanes. Put 0 for infinite i.e. waits for
                                 keypress.  (default=`0')
      --show                   Show the detected lines  (default=off)
      --step                   Step through each image (needs a keypress) or
                                 fall through (waits for --wait msecs)
                                 (default=off)
      --show-lane-numbers      Show the lane numbers on the output image
                                 (default=off)
      --output-suffix=STRING   Suffix of images and results
                                 (default=`_results')
      --save-images            Export all images with detected lanes to the by
                                 appending --output-suffix + '.png' to each
                                 input image  (default=off)
      --save-lanes             Export all detected lanes to a text file by
                                 appending --output-suffix + '.txt' to
                                 --list-file  (default=off)
      --debug                  Show debugging information and images
                                 (default=off)
```
