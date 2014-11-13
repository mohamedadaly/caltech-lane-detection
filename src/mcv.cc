/***
 * \file mcv.cc
 * \author Mohamed Aly <malaa@caltech.edu>
 * \date 11/29/2006
 */

#include "mcv.hh"

#include <iostream>
#include <math.h>
#include <assert.h>

using namespace std;

#include <cv.h>
#include <highgui.h>

namespace LaneDetector
{

//some helper functions for debugging

//print the matrix passed to it
void SHOW_MAT(const CvMat *pmat, char str[])
{
  cerr << str << "\n";
  for(int i=0; i<pmat->rows; i++)
  {
    for(int j=0; j<pmat->cols; j++)
      cerr << cvGetReal2D(pmat, i, j)  << " ";
    cerr << "\n";
  }
}

void SHOT_MAT_TYPE(const CvMat *pmat)
{
  cout << CV_MAT_TYPE(pmat->type) << "\n";
}

void SHOW_IMAGE(const CvMat *pmat, const char str[], int wait)
{
  //cout << "channels:" << CV_MAT_CN(pmat->type) << "\n";
  //scale it
  //CvMat *mat = cvCreateMat(pmat->height, pmat->width, pmat->type);
  //cvCopy(pmat, mat);
  CvMat *mat = cvCloneMat(pmat);//->rows, pmat->cols, INT_MAT_TYPE);//cvCloneMat(pmat);
  assert(mat);
  //convert to int type
  //cvConvert(pmat, mat);
  if(CV_MAT_CN(mat->type) == 1)//FLOAT_MAT_TYPE)
    mcvScaleMat(mat, mat);
  //show it
  //cout << "in\n";
  cvNamedWindow(str, CV_WINDOW_AUTOSIZE); //0 1
  cvShowImage(str, mat);
  cvWaitKey(wait);
  //cvDestroyWindow(str);
  //clear
  cvReleaseMat(&mat);
  //cout << "out\n";
}

void SHOW_IMAGE(const IplImage *pmat, char str[])
{
  //cout << "channels:" << CV_MAT_CN(pmat->type) << "\n";
  //scale it
  //CvMat *mat = cvCreateMat(pmat->height, pmat->width, pmat->type);
  //cvCopy(pmat, mat);
  //CvMat *mat = cvCloneMat(pmat);
  //assert(mat);
  //    mcvScaleMat(mat, mat);
  //show it
  //cout << "in\n";
  cvNamedWindow(str, 1);
  cvShowImage(str, pmat);
  cvWaitKey(0);
  //cvDestroyWindow(str);
  //clear
  //cvReleaseMat(&mat);
  //cout << "out\n";
}

void SHOW_POINT(const FLOAT_POINT2D pt, char str[])
{
  cerr << str << "(" << pt.x << "," << pt.y << ")\n";
  cerr.flush();
}


void SHOW_RECT(const CvRect rect, char str[])
{
  cerr << str << "(x=" << rect.x << ", y=" << rect.y
    << ", width=" << rect.width << ", height="
    << rect.height << ")" << endl;
  cerr.flush();
}

/**
 * This function reads in an image from file and returns the original color
 * image and the first (red) channel scaled to [0 .. 1] with float type.
 * images are allocated inside the function, so you will need to deallocate
 * them
 *
 * \param filename the input file name
 * \param clrImage the raw input image
 * \param channelImage the first channel
 */
void mcvLoadImage(const char *filename, CvMat **clrImage, CvMat** channelImage)
{
  // load the image
  IplImage* im;
  im = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);
  // convert to mat and get first channel
  CvMat temp;
  cvGetMat(im, &temp);
  *clrImage = cvCloneMat(&temp);
  // convert to single channel
  CvMat *schannel_mat;
  CvMat* tchannelImage = cvCreateMat(im->height, im->width, INT_MAT_TYPE);
  cvSplit(*clrImage, tchannelImage, NULL, NULL, NULL);
  // convert to float
  *channelImage = cvCreateMat(im->height, im->width, FLOAT_MAT_TYPE);
  cvConvertScale(tchannelImage, *channelImage, 1./255);
  // destroy
  cvReleaseMat(&tchannelImage);
  cvReleaseImage(&im);
}

/**
 * This function scales the input image to have values 0->1
 *
 * \param inImage the input image
 * \param outImage hte output iamge
 */
void mcvScaleMat(const CvMat *inMat, CvMat *outMat)
{
  //convert inMat type to outMat type
  cvConvert(inMat, outMat);
  //if (CV_MAT_DEPTH(inMat->type)
  //get the min and subtract it
  double min;
  cvMinMaxLoc(inMat, &min, 0, 0, 0, 0);
  cvSubS(inMat, cvRealScalar(min), outMat);

  //get max and divide by it
  double max;
  cvMinMaxLoc(outMat, 0, &max, 0, 0, 0);
  if(CV_MAT_TYPE(outMat->type) == FLOAT_MAT_TYPE)
    cvConvertScale(outMat, outMat,  1/max);
  else if(CV_MAT_TYPE(outMat->type) == INT_MAT_TYPE)
    cvConvertScale(outMat, outMat,  255/max);
}

/**
 * This function creates a double matrix from an input vector
 *
 * \param vec the input vector
 * \param mat the output matrix (column vector)
 *
 */
template <class T>
CvMat* mcvVector2Mat(const vector<T> &vec)
{
  CvMat *mat = 0;

  if (vec.size()>0)
  {
    //create the matrix
    mat = cvCreateMat(vec.size(), 1, CV_64FC1);
    //loop and get values
    for (int i=0; i<(int)vec.size(); i++)
      CV_MAT_ELEM(*mat, double, i, 0) =(double) vec[i];
  }

  //return
  return mat;
}

// template CvMat* mcvVector2Mat(const vector<double> &vec);
// template CvMat* mcvVector2Mat(const vector<int> &vec);

} // namespace LaneDetector
