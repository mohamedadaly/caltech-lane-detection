/**
 * \file main.hh
 * \author Mohamed Aly <malaa@caltech.edu>
 * \date Wed Oct 6, 2010
 *
 */

#ifndef LANE_DETECTOR_HH
#define LANE_DETECTOR_HH

#include "mcv.hh"
#include "InversePerspectiveMapping.hh"
#include "LaneDetector.hh"
#include "cmdline.h"

#include <vector>
#include <string>

using namespace std;

namespace LaneDetector
{

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
                  gengetopt_args_info& options, ofstream* outputFile = NULL,
                  int index = 0, clock_t *elapsedTime = NULL);

/**
 * This function reads lines from the input file into a vector of strings
 *
 * \param filename the input file name
 * \param lines the output vector of lines
 */
bool ReadLines(const char* filename, vector<string> *lines);

} // namespace LaneDetector

#endif //define LANE_DETECTOR_HH

