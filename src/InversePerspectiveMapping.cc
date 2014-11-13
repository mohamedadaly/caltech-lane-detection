/***
 * \file InversePerspectiveMapping.cc
 * \author Mohamed Aly <malaa@caltech.edu>
 * \date 11/29/2006
 */

#include "InversePerspectiveMapping.hh"

#include "CameraInfoOpt.h"

#include <iostream>
#include <math.h>
#include <assert.h>
#include <list>

using namespace std;
#include <cv.h>
#include <highgui.h>

namespace LaneDetector
{

#define VP_PORTION 0.05

/*
 We are assuming the world coordinate frame center is at the camera,
 the ground plane is at height -h, the X-axis is going right,
 the Y-axis is going forward, the Z-axis is going up. The
 camera is looking forward with optical axis in direction of
 Y-axis, with possible pitch angle (above or below the Y-axis)
 and yaw angle (left or right).
 The camera coordinates have the same center as the world, but the Xc-axis goes right,
 the  Yc-axis goes down, and the Zc-axis (optical cxis) goes forward. The
 uv-plane of the image is such that u is horizontal going right, v is
 vertical going down.
 The image coordinates uv are such that the pixels are at half coordinates
 i.e. first pixel is (.5,.5) ...etc where the top-left point is (0,0) i.e.
 the tip of the first pixel is (0,0)
*/

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
 * \param outPoints indices of points outside the image
 */
void mcvGetIPM(const CvMat* inImage, CvMat* outImage,
               IPMInfo *ipmInfo, const CameraInfo *cameraInfo,
               list<CvPoint> *outPoints)
{
  //check input images types
  //CvMat inMat, outMat;
  //cvGetMat(inImage, &inMat);
  //cvGetMat(outImage, &outMat);
  //cout << CV_MAT_TYPE(inImage->type) << " " << CV_MAT_TYPE(FLOAT_MAT_TYPE) <<  " " << CV_MAT_TYPE(INT_MAT_TYPE)<<"\n";
  if (!(CV_ARE_TYPES_EQ(inImage, outImage) &&
      (CV_MAT_TYPE(inImage->type)==CV_MAT_TYPE(FLOAT_MAT_TYPE) ||
      (CV_MAT_TYPE(inImage->type)==CV_MAT_TYPE(INT_MAT_TYPE)))))
  {
    cerr << "Unsupported image types in mcvGetIPM";
    exit(1);
  }

  //get size of input image
  FLOAT u, v;
  v = inImage->height;
  u = inImage->width;

  //get the vanishing point
  FLOAT_POINT2D vp;
  vp = mcvGetVanishingPoint(cameraInfo);
  vp.y = MAX(0, vp.y);
  //vp.y = 30;

  //get extent of the image in the xfyf plane
  FLOAT_MAT_ELEM_TYPE eps = ipmInfo->vpPortion * v;//VP_PORTION*v;
  ipmInfo->ipmLeft = MAX(0, ipmInfo->ipmLeft);
  ipmInfo->ipmRight = MIN(u-1, ipmInfo->ipmRight);
  ipmInfo->ipmTop = MAX(vp.y+eps, ipmInfo->ipmTop);
  ipmInfo->ipmBottom = MIN(v-1, ipmInfo->ipmBottom);
  FLOAT_MAT_ELEM_TYPE uvLimitsp[] = {vp.x,
    ipmInfo->ipmRight, ipmInfo->ipmLeft, vp.x,
    ipmInfo->ipmTop, ipmInfo->ipmTop,   ipmInfo->ipmTop,  ipmInfo->ipmBottom};
	//{vp.x, u, 0, vp.x,
	//vp.y+eps, vp.y+eps, vp.y+eps, v};
  CvMat uvLimits = cvMat(2, 4, FLOAT_MAT_TYPE, uvLimitsp);

  //get these points on the ground plane
  CvMat * xyLimitsp = cvCreateMat(2, 4, FLOAT_MAT_TYPE);
  CvMat xyLimits = *xyLimitsp;
  mcvTransformImage2Ground(&uvLimits, &xyLimits,cameraInfo);
  //SHOW_MAT(xyLimitsp, "xyLImits");

  //get extent on the ground plane
  CvMat row1, row2;
  cvGetRow(&xyLimits, &row1, 0);
  cvGetRow(&xyLimits, &row2, 1);
  double xfMax, xfMin, yfMax, yfMin;
  cvMinMaxLoc(&row1, (double*)&xfMin, (double*)&xfMax, 0, 0, 0);
  cvMinMaxLoc(&row2, (double*)&yfMin, (double*)&yfMax, 0, 0, 0);

  INT outRow = outImage->height;
  INT outCol = outImage->width;

  FLOAT_MAT_ELEM_TYPE stepRow = (yfMax-yfMin)/outRow;
  FLOAT_MAT_ELEM_TYPE stepCol = (xfMax-xfMin)/outCol;

  //construct the grid to sample
  CvMat *xyGrid = cvCreateMat(2, outRow*outCol, FLOAT_MAT_TYPE);
  INT i, j;
  FLOAT_MAT_ELEM_TYPE x, y;
  //fill it with x-y values on the ground plane in world frame
  for (i=0, y=yfMax-.5*stepRow; i<outRow; i++, y-=stepRow)
    for (j=0, x=xfMin+.5*stepCol; j<outCol; j++, x+=stepCol)
    {
      CV_MAT_ELEM(*xyGrid, FLOAT_MAT_ELEM_TYPE, 0, i*outCol+j) = x;
      CV_MAT_ELEM(*xyGrid, FLOAT_MAT_ELEM_TYPE, 1, i*outCol+j) = y;
    }
  //get their pixel values in image frame
  CvMat *uvGrid = cvCreateMat(2, outRow*outCol, FLOAT_MAT_TYPE);
  mcvTransformGround2Image(xyGrid, uvGrid, cameraInfo);
  //now loop and find the nearest pixel value for each position
  //that's inside the image, otherwise put it zero
  FLOAT_MAT_ELEM_TYPE ui, vi;
  //get mean of the input image
  CvScalar means = cvAvg(inImage);
  double mean = means.val[0];
  //generic loop to work for both float and int matrix types
  #define MCV_GET_IPM(type) \
  for (i=0; i<outRow; i++) \
      for (j=0; j<outCol; j++) \
      { \
          /*get pixel coordiantes*/ \
          ui = CV_MAT_ELEM(*uvGrid, FLOAT_MAT_ELEM_TYPE, 0, i*outCol+j); \
          vi = CV_MAT_ELEM(*uvGrid, FLOAT_MAT_ELEM_TYPE, 1, i*outCol+j); \
          /*check if out-of-bounds*/ \
          /*if (ui<0 || ui>u-1 || vi<0 || vi>v-1) \*/ \
          if (ui<ipmInfo->ipmLeft || ui>ipmInfo->ipmRight || \
              vi<ipmInfo->ipmTop || vi>ipmInfo->ipmBottom) \
          { \
              CV_MAT_ELEM(*outImage, type, i, j) = (type)mean; \
          } \
          /*not out of bounds, then get nearest neighbor*/ \
          else \
          { \
              /*Bilinear interpolation*/ \
              if (ipmInfo->ipmInterpolation == 0) \
              { \
                  int x1 = int(ui), x2 = int(ui+1); \
                  int y1 = int(vi), y2 = int(vi+1); \
                  float x = ui - x1, y = vi - y1;   \
                  float val = CV_MAT_ELEM(*inImage, type, y1, x1) * (1-x) * (1-y) + \
                      CV_MAT_ELEM(*inImage, type, y1, x2) * x * (1-y) + \
                      CV_MAT_ELEM(*inImage, type, y2, x1) * (1-x) * y + \
                      CV_MAT_ELEM(*inImage, type, y2, x2) * x * y;   \
                  CV_MAT_ELEM(*outImage, type, i, j) =  (type)val; \
  } \
              /*nearest-neighbor interpolation*/ \
              else \
                  CV_MAT_ELEM(*outImage, type, i, j) = \
                      CV_MAT_ELEM(*inImage, type, int(vi+.5), int(ui+.5)); \
          } \
          if (outPoints && \
              (ui<ipmInfo->ipmLeft+10 || ui>ipmInfo->ipmRight-10 || \
              vi<ipmInfo->ipmTop || vi>ipmInfo->ipmBottom-2) )\
              outPoints->push_back(cvPoint(j, i)); \
      }
  if (CV_MAT_TYPE(inImage->type)==FLOAT_MAT_TYPE)
  {
      MCV_GET_IPM(FLOAT_MAT_ELEM_TYPE)
  }
  else
  {
      MCV_GET_IPM(INT_MAT_ELEM_TYPE)
  }
  //return the ipm info
  ipmInfo->xLimits[0] = CV_MAT_ELEM(*xyGrid, FLOAT_MAT_ELEM_TYPE, 0, 0);
  ipmInfo->xLimits[1] =
    CV_MAT_ELEM(*xyGrid, FLOAT_MAT_ELEM_TYPE, 0, (outRow-1)*outCol+outCol-1);
  ipmInfo->yLimits[1] = CV_MAT_ELEM(*xyGrid, FLOAT_MAT_ELEM_TYPE, 1, 0);
  ipmInfo->yLimits[0] =
    CV_MAT_ELEM(*xyGrid, FLOAT_MAT_ELEM_TYPE, 1, (outRow-1)*outCol+outCol-1);
  ipmInfo->xScale = 1/stepCol;
  ipmInfo->yScale = 1/stepRow;
  ipmInfo->width = outCol;
  ipmInfo->height = outRow;

  //clean
  cvReleaseMat(&xyLimitsp);
  cvReleaseMat(&xyGrid);
  cvReleaseMat(&uvGrid);
}


/**
 * Transforms points from the image frame (uv-coordinates)
 * into the real world frame on the ground plane (z=-height)
 *
 * \param inPoints input points in the image frame
 * \param outPoints output points in the world frame on the ground
 *          (z=-height)
 * \param cemaraInfo the input camera parameters
 *
 */
void mcvTransformImage2Ground(const CvMat *inPoints,
                              CvMat *outPoints, const CameraInfo *cameraInfo)
{

  //add two rows to the input points
  CvMat *inPoints4 = cvCreateMat(inPoints->rows+2, inPoints->cols,
      cvGetElemType(inPoints));

  //copy inPoints to first two rows
  CvMat inPoints2, inPoints3, inPointsr4, inPointsr3;
  cvGetRows(inPoints4, &inPoints2, 0, 2);
  cvGetRows(inPoints4, &inPoints3, 0, 3);
  cvGetRow(inPoints4, &inPointsr3, 2);
  cvGetRow(inPoints4, &inPointsr4, 3);
  cvSet(&inPointsr3, cvRealScalar(1));
  cvCopy(inPoints, &inPoints2);
  //create the transformation matrix
  float c1 = cos(cameraInfo->pitch);
  float s1 = sin(cameraInfo->pitch);
  float c2 = cos(cameraInfo->yaw);
  float s2 = sin(cameraInfo->yaw);
  float matp[] = {
    -cameraInfo->cameraHeight*c2/cameraInfo->focalLength.x,
    cameraInfo->cameraHeight*s1*s2/cameraInfo->focalLength.y,
    (cameraInfo->cameraHeight*c2*cameraInfo->opticalCenter.x/
      cameraInfo->focalLength.x)-
      (cameraInfo->cameraHeight *s1*s2* cameraInfo->opticalCenter.y/
      cameraInfo->focalLength.y) - cameraInfo->cameraHeight *c1*s2,

    cameraInfo->cameraHeight *s2 /cameraInfo->focalLength.x,
    cameraInfo->cameraHeight *s1*c2 /cameraInfo->focalLength.y,
    (-cameraInfo->cameraHeight *s2* cameraInfo->opticalCenter.x
      /cameraInfo->focalLength.x)-(cameraInfo->cameraHeight *s1*c2*
      cameraInfo->opticalCenter.y /cameraInfo->focalLength.y) -
      cameraInfo->cameraHeight *c1*c2,

    0,
    cameraInfo->cameraHeight *c1 /cameraInfo->focalLength.y,
    (-cameraInfo->cameraHeight *c1* cameraInfo->opticalCenter.y /
      cameraInfo->focalLength.y) + cameraInfo->cameraHeight *s1,

    0,
    -c1 /cameraInfo->focalLength.y,
    (c1* cameraInfo->opticalCenter.y /cameraInfo->focalLength.y) - s1,
  };
  CvMat mat = cvMat(4, 3, CV_32FC1, matp);
  //multiply
  cvMatMul(&mat, &inPoints3, inPoints4);
  //divide by last row of inPoints4
  for (int i=0; i<inPoints->cols; i++)
  {
    float div = CV_MAT_ELEM(inPointsr4, float, 0, i);
    CV_MAT_ELEM(*inPoints4, float, 0, i) =
        CV_MAT_ELEM(*inPoints4, float, 0, i) / div ;
    CV_MAT_ELEM(*inPoints4, float, 1, i) =
        CV_MAT_ELEM(*inPoints4, float, 1, i) / div;
  }
  //put back the result into outPoints
  cvCopy(&inPoints2, outPoints);
  //clear
  cvReleaseMat(&inPoints4);
}


/**
 * Transforms points from the ground plane (z=-h) in the world frame
 * into points on the image in image frame (uv-coordinates)
 *
 * \param inPoints 2xN array of input points on the ground in world coordinates
 * \param outPoints 2xN output points in on the image in image coordinates
 * \param cameraInfo the camera parameters
 *
 */
void mcvTransformGround2Image(const CvMat *inPoints,
                              CvMat *outPoints, const CameraInfo *cameraInfo)
{
  //add two rows to the input points
  CvMat *inPoints3 = cvCreateMat(inPoints->rows+1, inPoints->cols,
      cvGetElemType(inPoints));

  //copy inPoints to first two rows
  CvMat inPoints2,  inPointsr3;
  cvGetRows(inPoints3, &inPoints2, 0, 2);
  cvGetRow(inPoints3, &inPointsr3, 2);
  cvSet(&inPointsr3, cvRealScalar(-cameraInfo->cameraHeight));
  cvCopy(inPoints, &inPoints2);
  //create the transformation matrix
  float c1 = cos(cameraInfo->pitch);
  float s1 = sin(cameraInfo->pitch);
  float c2 = cos(cameraInfo->yaw);
  float s2 = sin(cameraInfo->yaw);
  float matp[] = {
    cameraInfo->focalLength.x * c2 + c1*s2* cameraInfo->opticalCenter.x,
    -cameraInfo->focalLength.x * s2 + c1*c2* cameraInfo->opticalCenter.x,
    - s1 * cameraInfo->opticalCenter.x,

    s2 * (-cameraInfo->focalLength.y * s1 + c1* cameraInfo->opticalCenter.y),
    c2 * (-cameraInfo->focalLength.y * s1 + c1* cameraInfo->opticalCenter.y),
    -cameraInfo->focalLength.y * c1 - s1* cameraInfo->opticalCenter.y,

    c1*s2,
    c1*c2,
    -s1
  };
  CvMat mat = cvMat(3, 3, CV_32FC1, matp);
  //multiply
  cvMatMul(&mat, inPoints3, inPoints3);
  //divide by last row of inPoints4
  for (int i=0; i<inPoints->cols; i++)
  {
    float div = CV_MAT_ELEM(inPointsr3, float, 0, i);
    CV_MAT_ELEM(*inPoints3, float, 0, i) =
        CV_MAT_ELEM(*inPoints3, float, 0, i) / div ;
    CV_MAT_ELEM(*inPoints3, float, 1, i) =
        CV_MAT_ELEM(*inPoints3, float, 1, i) / div;
  }
  //put back the result into outPoints
  cvCopy(&inPoints2, outPoints);
  //clear
  cvReleaseMat(&inPoints3);
}


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
FLOAT_POINT2D mcvGetVanishingPoint(const CameraInfo *cameraInfo)
{
  //get the vp in world coordinates
  FLOAT_MAT_ELEM_TYPE vpp[] = {sin(cameraInfo->yaw)/cos(cameraInfo->pitch),
          cos(cameraInfo->yaw)/cos(cameraInfo->pitch), 0};
  CvMat vp = cvMat(3, 1, FLOAT_MAT_TYPE, vpp);

  //transform from world to camera coordinates
  //
  //rotation matrix for yaw
  FLOAT_MAT_ELEM_TYPE tyawp[] = {cos(cameraInfo->yaw), -sin(cameraInfo->yaw), 0,
                sin(cameraInfo->yaw), cos(cameraInfo->yaw), 0,
                0, 0, 1};
  CvMat tyaw = cvMat(3, 3, FLOAT_MAT_TYPE, tyawp);
  //rotation matrix for pitch
  FLOAT_MAT_ELEM_TYPE tpitchp[] = {1, 0, 0,
                  0, -sin(cameraInfo->pitch), -cos(cameraInfo->pitch),
                  0, cos(cameraInfo->pitch), -sin(cameraInfo->pitch)};
  CvMat transform = cvMat(3, 3, FLOAT_MAT_TYPE, tpitchp);
  //combined transform
  cvMatMul(&transform, &tyaw, &transform);

  //
  //transformation from (xc, yc) in camra coordinates
  // to (u,v) in image frame
  //
  //matrix to shift optical center and focal length
  FLOAT_MAT_ELEM_TYPE t1p[] = {
    cameraInfo->focalLength.x, 0,
    cameraInfo->opticalCenter.x,
    0, cameraInfo->focalLength.y,
    cameraInfo->opticalCenter.y,
    0, 0, 1};
  CvMat t1 = cvMat(3, 3, FLOAT_MAT_TYPE, t1p);
  //combine transform
  cvMatMul(&t1, &transform, &transform);
  //transform
  cvMatMul(&transform, &vp, &vp);

  //
  //clean and return
  //
  FLOAT_POINT2D ret;
  ret.x = cvGetReal1D(&vp, 0);
  ret.y = cvGetReal1D(&vp, 1);
  return ret;
}


/**
 * Converts a point from IPM pixel coordinates into world coordinates
 *
 * \param point in/out point
 * \param ipmInfo the ipm info from mcvGetIPM
 *
 */
void mcvPointImIPM2World(FLOAT_POINT2D *point, const IPMInfo *ipmInfo)
{
  //x-direction
  point->x /= ipmInfo->xScale;
  point->x += ipmInfo->xLimits[0];
  //y-direction
  point->y /= ipmInfo->yScale;
  point->y = ipmInfo->yLimits[1] - point->y;
}


/**
 * Converts from IPM pixel coordinates into world coordinates
 *
 * \param inMat input matrix 2xN
 * \param outMat output matrix 2xN
 * \param ipmInfo the ipm info from mcvGetIPM
 *
 */
void mcvTransformImIPM2Ground(const CvMat *inMat, CvMat* outMat, const IPMInfo *ipmInfo)
{
  CvMat *mat;
  mat = outMat;
  if(inMat != mat)
  {
    cvCopy(inMat, mat);
  }

  //work on the x-direction i.e. first row
  CvMat row;
  cvGetRow(mat, &row, 0);
  cvConvertScale(&row, &row, 1./ipmInfo->xScale, ipmInfo->xLimits[0]);

  //work on y-direction
  cvGetRow(mat, &row, 1);
  cvConvertScale(&row, &row, -1./ipmInfo->yScale, ipmInfo->yLimits[1]);
}

/**
 * Converts from IPM pixel coordinates into Image coordinates
 *
 * \param inMat input matrix 2xN
 * \param outMat output matrix 2xN
 * \param ipmInfo the ipm info from mcvGetIPM
 * \param cameraInfo the camera info
 *
 */
void mcvTransformImIPM2Im(const CvMat *inMat, CvMat* outMat, const IPMInfo *ipmInfo,
			  const CameraInfo *cameraInfo)
{
  //convert to world coordinates
  mcvTransformImIPM2Ground(inMat, outMat, ipmInfo);

  //convert to image coordinates
  mcvTransformGround2Image(outMat, outMat, cameraInfo);

}


/**
 * Initializes the cameraInfo structure with data read from the conf file
 *
 * \param fileName the input camera conf file name
 * \param cameraInfo the returned camera parametrs struct
 *
 */
void mcvInitCameraInfo (char * const fileName, CameraInfo *cameraInfo)
{
  //parsed camera data
  CameraInfoParserInfo camInfo;
  //read the data
  assert(cameraInfoParser_configfile(fileName, &camInfo, 0, 1, 1)==0);
  //init the strucure
  cameraInfo->focalLength.x = camInfo.focalLengthX_arg;
  cameraInfo->focalLength.y = camInfo.focalLengthY_arg;
  cameraInfo->opticalCenter.x = camInfo.opticalCenterX_arg;
  cameraInfo->opticalCenter.y = camInfo.opticalCenterY_arg;
  cameraInfo->cameraHeight = camInfo.cameraHeight_arg;
  cameraInfo->pitch = camInfo.pitch_arg * CV_PI/180;
  cameraInfo->yaw = camInfo.yaw_arg * CV_PI/180;
  cameraInfo->imageWidth = camInfo.imageWidth_arg;
  cameraInfo->imageHeight = camInfo.imageHeight_arg;
}


/**
 * Scales the cameraInfo according to the input image size
 *
 * \param cameraInfo the input/return structure
 * \param size the input image size
 *
 */
 void mcvScaleCameraInfo (CameraInfo *cameraInfo, CvSize size)
 {
  //compute the scale factor
  double scaleX = size.width/cameraInfo->imageWidth;
  double scaleY = size.height/cameraInfo->imageHeight;
  //scale
  cameraInfo->imageWidth = size.width;
  cameraInfo->imageHeight = size.height;
  cameraInfo->focalLength.x *= scaleX;
  cameraInfo->focalLength.y *= scaleY;
  cameraInfo->opticalCenter.x *= scaleX;
  cameraInfo->opticalCenter.y *= scaleY;
 }


/**
 * Gets the extent of the image on the ground plane given the camera parameters
 *
 * \param cameraInfo the input camera info
 * \param ipmInfo the IPM info containing the extent on ground plane:
 *  xLimits & yLimits only are changed
 *
 */
void mcvGetIPMExtent(const CameraInfo *cameraInfo, IPMInfo *ipmInfo )
{
  //get size of input image
  FLOAT u, v;
  v = cameraInfo->imageHeight;
  u = cameraInfo->imageWidth;

  //get the vanishing point
  FLOAT_POINT2D vp;
  vp = mcvGetVanishingPoint(cameraInfo);
  vp.y = MAX(0, vp.y);

  //get extent of the image in the xfyf plane
  FLOAT_MAT_ELEM_TYPE eps = VP_PORTION*v;
  FLOAT_MAT_ELEM_TYPE uvLimitsp[] = {vp.x, u, 0, vp.x,
                      vp.y+eps, vp.y+eps, vp.y+eps, v};
  CvMat uvLimits = cvMat(2, 4, FLOAT_MAT_TYPE, uvLimitsp);

  //get these points on the ground plane
  CvMat * xyLimitsp = cvCreateMat(2, 4, FLOAT_MAT_TYPE);
  CvMat xyLimits = *xyLimitsp;
  mcvTransformImage2Ground(&uvLimits, &xyLimits,cameraInfo);
  //SHOW_MAT(xyLimitsp, "xyLImits");

  //get extent on the ground plane
  CvMat row1, row2;
  cvGetRow(&xyLimits, &row1, 0);
  cvGetRow(&xyLimits, &row2, 1);
  double xfMax, xfMin, yfMax, yfMin;
  cvMinMaxLoc(&row1, (double*)&xfMin, (double*)&xfMax, 0, 0, 0);
  cvMinMaxLoc(&row2, (double*)&yfMin, (double*)&yfMax, 0, 0, 0);

  //return
  ipmInfo->xLimits[0] = xfMin;
  ipmInfo->xLimits[1] = xfMax;
  ipmInfo->yLimits[1] = yfMax;
  ipmInfo->yLimits[0] = yfMin;

}

} // namespace LaneDetector
