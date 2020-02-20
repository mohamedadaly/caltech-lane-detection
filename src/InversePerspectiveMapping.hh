/***
 * \file InversePerspectiveMapping.hh
 * \author Mohamed Aly <malaa@caltech.edu>
 * \date 11/29/2006
 */

#ifndef INVERSEPERSPECTIVEMAPPING_HH_
#define INVERSEPERSPECTIVEMAPPING_HH_


#include <list>

#include <opencv2/core.hpp>

#include "mcv.hh"

//conf file for cameraInfo
#include "CameraInfoOpt.h"

using namespace std;

namespace LaneDetector
{

/**
 * Structure to hold the info about IPM transformation
 */
typedef struct IPMInfo
{
  ///min and max x-value on ground in world coordinates
  LD_FLOAT xLimits[2];
  ///min and max y-value on ground in world coordinates
  LD_FLOAT yLimits[2];
  ///conversion between mm in world coordinate on the ground
  ///in x-direction and pixel in image
  LD_FLOAT xScale;
  ///conversion between mm in world coordinate on the ground
  ///in y-direction and pixel in image
  LD_FLOAT yScale;
  ///width
  int width;
  ///height
  int height;
  ///portion of image height to add to y-coordinate of
  ///vanishing point
  float vpPortion;
  ///Left point in original image of region to make IPM for
  float ipmLeft;
  ///Right point in original image of region to make IPM for
  float ipmRight;
  ///Top point in original image of region to make IPM for
  float ipmTop;
  ///Bottom point in original image of region to make IPM for
  float ipmBottom;
  ///interpolation to use for IPM (0: bilinear, 1:nearest neighbor)
  int ipmInterpolation;
}IPMInfo;

///Camera Calibration info
typedef struct CameraInfo
{
  ///focal length in x and y
  FLOAT_POINT2D focalLength;
  ///optical center coordinates in image frame (origin is (0,0) at top left)
  FLOAT_POINT2D opticalCenter;
  ///height of camera above ground
  LD_FLOAT cameraHeight;
  ///pitch angle in radians (+ve downwards)
  LD_FLOAT pitch;
  ///yaw angle in radians (+ve clockwise)
  LD_FLOAT yaw;
  ///width of images
  LD_FLOAT imageWidth;
  ///height of images
  LD_FLOAT imageHeight;
}CameraInfo;

//functions definitions
/**
 * This function returns the Inverse Perspective Mapping
 * of the input image, assuming a flat ground plane, and
 * given the camera parameters.
 *
 * \param inImage the input image
 * \param outImage the output image in IPM
 * \param ipmInfo the returned IPM info for the transformation
 * \param focalLength focal length (in x and y direction)
 * \param cameraInfo the camera parameters
 */
void mcvGetIPM(const cv::Mat* inImage, cv::Mat* outImage,
               IPMInfo *ipmInfo, const CameraInfo *cameraInfo,
               list<cv::Point>* outPoints=NULL);


/**
 * Transforms points from the image frame (uv-coordinates)
 * into the real world frame on the ground plane (z=-height)
 *
 * \param inPoints input points in the image frame (2xN matrix)
 * \param outPoints output points in the world frame on the ground
 *          (z=-height) (2xN matrix with xw, yw and implicit z=-height)
 * \param cemaraInfo the input camera parameters
 *
 */
void mcvTransformImage2Ground(const cv::Mat *inPoints,
                              cv::Mat *outPoints, const CameraInfo *cameraInfo);


/**
 * Transforms points from the ground plane (z=-h) in the world frame
 * into points on the image in image frame (uv-coordinates)
 *
 * \param inPoints 2xN array of input points on the ground in world coordinates
 * \param outPoints 2xN output points in on the image in image coordinates
 * \param cameraInfo the camera parameters
 *
 */
void mcvTransformGround2Image(const cv::Mat *inPoints,
                              cv::Mat *outPoints, const CameraInfo *cameraInfo);

/**
 * Computes the vanishing point in the image plane uv. It is
 * the point of intersection of the image plane with the line
 * in the XY-plane in the world coordinates that makes an
 * angle yaw clockwise (form Y-axis) with Y-axis
 *
 * \param cameraInfo the input camera parameter
 *
 * \return the computed vanishing point in image frame
 *
 */
FLOAT_POINT2D mcvGetVanishingPoint(const CameraInfo *cameraInfo);

/**
 * Converts a point from IPM pixel coordinates into world coordinates
 *
 * \param point in/out point
 * \param ipmInfo the ipm info from mcvGetIPM
 *
 */
void mcvPointImIPM2World(FLOAT_POINT2D *point, const IPMInfo *ipmInfo);

/**
 * Initializes the cameraInfo structure with data read from the conf file
 *
 * \param fileName the input camera conf file name
 * \param cameraInfo the returned camera parametrs struct
 *
 */
void mcvInitCameraInfo (char *const fileName, CameraInfo *cameraInfo);

/**
 * Scales the cameraInfo according to the input image size
 *
 * \param cameraInfo the input/return structure
 * \param size the input image size
 *
 */
 void mcvScaleCameraInfo (CameraInfo *cameraInfo, cv::Size size);

/**
 * Gets the extent of the image on the ground plane given the camera parameters
 *
 * \param cameraInfo the input camera info
 * \param ipmInfo the IPM info containing the extent on ground plane:
 *  xLimits & yLimits only are changed
 *
 */
void mcvGetIPMExtent(const CameraInfo *cameraInfo, IPMInfo *ipmInfo);

/**
 * Converts from IPM pixel coordinates into world coordinates
 *
 * \param inMat input matrix 2xN
 * \param outMat output matrix 2xN
 * \param ipmInfo the ipm info from mcvGetIPM
 *
 */
void mcvTransformImIPM2Ground(const cv::Mat *inMat, cv::Mat* outMat,
                              const IPMInfo *ipmInfo);

/**
 * Converts from IPM pixel coordinates into Image coordinates
 *
 * \param inMat input matrix 2xN
 * \param outMat output matrix 2xN
 * \param ipmInfo the ipm info from mcvGetIPM
 * \param cameraInfo the camera info
 *
 */
void mcvTransformImIPM2Im(const cv::Mat *inMat, cv::Mat* outMat,
                          const IPMInfo *ipmInfo,
                          const CameraInfo *cameraInfo);

} // namespace LaneDetector

#endif /*INVERSEPERSPECTIVEMAPPING_HH_*/
