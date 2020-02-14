/**
 * \file main.cc
 * \author Mohamed Aly <malaa@caltech.edu>
 * \date Wed Oct 6, 2010
 *
 */

#include "main.hh"

#include "cmdline.h"
#include "LaneDetector.hh"

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// Useful message macro
#define MSG(fmt, ...) \
  (fprintf(stdout, "%s:%d msg   " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__) ? 0 : 0)

// Useful error macro
#define ERROR(fmt, ...) \
  (fprintf(stderr, "%s:%d error " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__) ? -1 : -1)


namespace LaneDetector
{

/**
 * This function reads lines from the input file into a vector of strings
 *
 * \param filename the input file name
 * \param lines the output vector of lines
 */
bool ReadLines(const char* filename, vector<string> *lines)
{
  // make sure it's not NULL
  if (!lines)
    return false;
  // resize
  lines->clear();

  ifstream  file;
  file.open(filename, ifstream::in);
  char buf[5000];
  // read lines and process
  while (file.getline(buf, 5000))
  {
    string str(buf);
    lines->push_back(str);
  }
  // close
  file.close();
  return true;
}


/**
 * This function processes an input image and detects lanes/stoplines
 * based on the passed in command line arguments
 *
 * \param filename the input file name
 * \param cameraInfo the camera calibration info
 * \param lanesConf the lane detection settings
 * \param stoplinesConf the stop line detection settings
 * \param options the command line arguments
 * \param outputFile the output file stream to write output lanes to
 * \param index the image index (used for saving output files)
 * \param elapsedTime if NOT NULL, it is accumulated with clock ticks for
 *        the detection operation
 */
void ProcessImage(const char* filename, CameraInfo& cameraInfo,
                  LaneDetectorConf& lanesConf, LaneDetectorConf& stoplinesConf,
                  gengetopt_args_info& options, ofstream* outputFile,
                  int index, clock_t *elapsedTime)
{
  // load the image
  cv::Mat *raw_mat = new cv::Mat(), *mat = new cv::Mat();
  mcvLoadImage(filename, raw_mat, mat);

  // detect lanes
  vector<LD_FLOAT> lineScores, splineScores;
  vector<Line> lanes;
  vector<Spline> splines;
  clock_t startTime = clock();
  mcvGetLanes(mat, raw_mat, &lanes, &lineScores, &splines, &splineScores,
              &cameraInfo, &lanesConf, NULL);
  clock_t endTime = clock();
  MSG("Found %d lanes in %f msec", (int)splines.size(),
      static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC * 1000.);
  // update elapsed time
  if (elapsedTime)
    (*elapsedTime) += endTime - startTime;

  // save results?
  if (options.save_lanes_flag && outputFile && outputFile->is_open())
  {
    (*outputFile) << "frame#" << setw(8) << setfill('0') << index <<
      " has " << splines.size() << " splines" << endl;
    for (int i=0; i<splines.size(); ++i)
    {
      (*outputFile) << "\tspline#" << i+1 << " has " <<
        splines[i].degree+1 << " points and score " <<
        splineScores[i] << endl;
      for (int j=0; j<=splines[i].degree; ++j)
        (*outputFile) << "\t\t" <<
          splines[i].points[j].x << ", " <<
          splines[i].points[j].y << endl;
    }
  }

  // show or save
  if (options.show_flag || options.save_images_flag)
  {
    // show detected lanes
    cv::Mat *imDisplay  = new cv::Mat();
    *imDisplay  = raw_mat->clone();
    // convert to BGR
//     cvcv::tColor(raw_mat, imDisplay, CV_RGB2BGR);
    if (lanesConf.ransacLine && !lanesConf.ransacSpline)
      for(int i=0; i<lanes.size(); i++)
        mcvDrawLine(imDisplay, lanes[i], CV_RGB(0,125,0), 3);
    // print lanes
    if (lanesConf.ransacSpline)
    {
      for(int i=0; i<splines.size(); i++)
      {
        if (splines[i].color == LINE_COLOR_YELLOW)
          mcvDrawSpline(imDisplay, splines[i], CV_RGB(255,255,0), 3);
        else
          mcvDrawSpline(imDisplay, splines[i], CV_RGB(0,255,0), 3);
        // print numbers?
        if (options.show_lane_numbers_flag)
        {
          char str[256];
          sprintf(str, "%d", i);
          mcvDrawText(imDisplay, str,
                      cv::Point(splines[i].points[splines[i].degree]),
                                      1, CV_RGB(0, 0, 255));
        }
      }
    }
    // show?
    if (options.show_flag)
    {
      // set the wait value
      int wait = options.step_flag ? 0 : options.wait_arg;
      // show image with detected lanes
      SHOW_IMAGE(imDisplay, "Detected Lanes", wait);
    }
    // save?
    if (options.save_images_flag)
    {
      // file name
      stringstream ss;
      ss << filename << options.output_suffix_arg << "_" << setw(6) <<
        setfill('0') << index << ".png";
      string outFilename = ss.str();
      // save the image file
      MSG("Writing output image: %s", outFilename.c_str());
      cv::imwrite(outFilename, *imDisplay);
    }
    // clear
    delete imDisplay;
  }

  delete raw_mat;
  delete mat;
}


int Process(int argc, char** argv)
{
  // parse the command line paramters
  gengetopt_args_info options;
  if (cmdline_parser (argc, argv,  &options) < 0)
    return -1;

  // read the camera configurations
  CameraInfo cameraInfo;
  mcvInitCameraInfo(options.camera_conf_arg, &cameraInfo);
  MSG("Loaded camera file");

  // read the configurations
  LaneDetectorConf lanesConf, stoplinesConf;
  if (!options.no_lanes_flag)
  {
    mcvInitLaneDetectorConf(options.lanes_conf_arg, &lanesConf);
    MSG("Loaded lanes config file");
  }
  if (!options.no_stoplines_flag)
  {
    mcvInitLaneDetectorConf(options.stoplines_conf_arg, &stoplinesConf);
    MSG("Loaded stop lines config file");
  }

  // set debug to true
  if (options.debug_flag)
    DEBUG_LINES = 1;

  // process a single image
  if (options.image_file_given)
  {
    // elapsed time
    clock_t elapsed = 0;
    ProcessImage(options.image_file_arg, cameraInfo, lanesConf, stoplinesConf,
                  options, NULL, elapsed, 0);
    double elapsedTime = static_cast<double>(elapsed) / CLOCKS_PER_SEC;
    MSG("Total time %f secs for 1 image = %f Hz", elapsedTime,
        1. / elapsedTime);
  }

  // process a list of images
  if (options.list_file_given)
  {
    // get the path if exists
    string path = "";
    if (options.list_path_given)
      path = options.list_path_arg;

    // read file
    vector<string> files;
    ReadLines(options.list_file_arg, &files);
    int numImages = files.size();
    if (numImages<1)
      ERROR("File %s is empty", options.list_file_arg);
    else
    {
      // save results?
      ofstream outputFile;
      stringstream ss;
      if (options.save_lanes_flag)
      {
        ss << options.list_file_arg << options.output_suffix_arg << ".txt";
        outputFile.open(ss.str().c_str(), ios_base::out);
      }

      // elapsed time
      clock_t elapsed = 0;
      // loop
      for (int i=0; i<numImages; ++i)
      {
        string imageFile = path + files[i];
        MSG("Processing image: %s", imageFile.c_str());
        ProcessImage(imageFile.c_str(), cameraInfo, lanesConf, stoplinesConf,
                     options, &outputFile, i, &elapsed);
      }
      double elapsedTime = static_cast<double>(elapsed) / CLOCKS_PER_SEC;
      MSG("Total time %f secs for %d images = %f Hz",
          elapsedTime, numImages, numImages / elapsedTime);

      // close results file (if open)
      if (options.save_lanes_flag)
      {
        outputFile.close();
        MSG("Results written to %s", ss.str().c_str());
      }
    }
  }

  return 0;
}

} // namespace LaneDetector

using LaneDetector::Process;

// main entry point
int main(int argc, char** argv)
{
  return Process(argc, argv);
}
