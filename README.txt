# Author: Mohamed Aly <malaa@caltech.edu>
# Date: 10/7/2010

============================================================================
                     REAL TIME LANE DETECTOR SOFTWARE
============================================================================

This package contains source code and dataset that implements the work in the
paper [1].

=========
Contents
=========
src/: contains the C/C++ source files

|_ CameraInfo.conf: contains the camera calibration ifo

|_ CameraInfoOpt.*: contain gengetopt files for parsing the camera info files

|_ cmdline.*: contains gengetopt files for parsing command lines

|_ InversePerspectiveMapping.*: code for obtainig the IPM of an image

|_ LaneDetector.*: code for the bulk of the algorithm, including Hough
      Transforms, Spline fitting, Post processing, ...

|_ Lanes.conf: the typical configuration file for lane detection

|_ main.*: code for the main binary

|_ Makefile: the Make file

|_ mcv.*: contain utility functions

|_ ranker.h: code for obtaining the median of a vector

|_ run.sh: Shell script for running the detector on the four clips in
    Caltech Lanes Dataset

matlab/: contains the Matlab source files

|_ ccvCheckMergeSplines.m: checks if two splines are matching

|_ ccvEvalBezSpline.m: returns points on a spline given its control points

|_ ccvGetLaneDetectionStats.m: computes stats from detections and ground truth

|_ ccvLabel.m: handles the ground truth labels

|_ ccvReadLaneDetectionResultsFile.m: reads a detection file output from the
    binary file LaneDetector32/64

|_ Stats.m: computes stats for the detections on the Caltech Lanes Dataset and
    its ground truth labels

==============
Prerequisites
==============
1. OpenCV 2.0 or higher http://sourceforge.net/projects/opencvlibrary/
3. (Optional) Gengetopt http://www.gnu.org/software/gengetopt/

===========
Compiling
===========
Unzip the archive somewhere, let's say ~/lane-detector:

unzip lane-detector.zip -d ~/lane-detector
cd ~/lane-detector/src
make release

This will generate LaneDetector32 or LaneDetector64 depending on your system.

======================
Caltech Lanes Dataset
======================
To view the lane detector in action, you can download the Caltech Lanes Dataset
available at http://www.vision.caltech.edu/malaa/datasets/caltech-lanes

===========
Running
===========
To run the detector on the Caltech Lanes dataset, which might be in
~/caltech-lanes/

cd ~/lane-detector/
ln -s ~/caltech-lanes/  clips
cd ~/lane-detector/src/
bash run.sh

This will create the results files inside
~/caltech-lanes/*/list.txt_results.txt

To view the statistics of the results, open Matlab and run the file:

cd ~/lane-detector/matlab/
matlab&
>>Stats

======================
Command line options
======================
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

===========
References
===========
[1] Mohamed Aly, Real time Detection of Lane Markers in Urban Streets,
  IEEE Intelligent Vehicles Symposium, Eindhoven, The Netherlands, June 2008.
