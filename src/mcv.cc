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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace LaneDetector
{

//some helper functions for debugging

//print the matrix passed to it
void SHOW_MAT(const cv::Mat *pmat, const char str[])
{
  cerr << str << "\n";
  for(int i=0; i<pmat->rows; i++)
  {
    for(int j=0; j<pmat->cols; j++)
      cerr << pmat->at<float>(i, j)  << " ";
    cerr << "\n";
  }
}

void SHOT_MAT_TYPE(const cv::Mat *pmat)
{
  cout << CV_MAT_TYPE(pmat->type()) << "\n";
}

void SHOW_IMAGE(const cv::Mat *pmat, const char str[], int wait)
{
  //cout << "channels:" << CV_MAT_CN(pmat->type()) << "\n";
  //scale it
  //cv::Mat *mat = new cv::Mat(pmat->height, pmat->width, pmat->type());
  //cv::copy(pmat, mat);
  cv::Mat *mat = new cv::Mat();//->rows, pmat->cols, INT_MAT_TYPE);//cv::cloneMat(pmat);
  *mat = pmat->clone();
  assert(mat);
  //convert to int type
  //cv::convert(pmat, mat);
  if(CV_MAT_CN(mat->type()) == 1)//FLOAT_MAT_TYPE)
    mcvScaleMat(mat, mat);
  //show it
  //cout << "in\n";
  cv::namedWindow(str, cv::WINDOW_AUTOSIZE); //0 1:
  cv::imshow(str, *mat);
  cv::waitKey(wait);
  //cv::destroyWindow(str);
  //clear
  delete mat;
  //cout << "out\n";
}

void SHOW_IMAGE(const cv::Mat *pmat, const char str[])
{
  //cout << "channels:" << CV_MAT_CN(pmat->type()) << "\n";
  //scale it
  //cv::Mat *mat = new cv::Mat(pmat->height, pmat->width, pmat->type());
  //cv::copy(pmat, mat);
  //cv::Mat *mat = cv::cloneMat(pmat);
  //assert(mat);
  //    mcvScaleMat(mat, mat);
  //show it
  //cout << "in\n";
  cv::namedWindow(str, 1);
  cv::imshow(str, *pmat);
  cv::waitKey(0);
  //cv::destroyWindow(str);
  //clear
  //delete mat;
  //cout << "out\n";
}

void SHOW_POINT(const FLOAT_POINT2D pt, const char str[])
{
  cerr << str << "(" << pt.x << "," << pt.y << ")\n";
  cerr.flush();
}


void SHOW_RECT(const cv::Rect rect, const char str[])
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
void mcvLoadImage(const char *filename, cv::Mat *clrImage, cv::Mat* channelImage)
{
  // load the image
  cv::Mat* temp = new cv::Mat(cv::imread(filename));

  //memcpy(temp->data.ptr, img.ptr, img.getImageSize());
  *clrImage = temp->clone();
  // convert to single channel
  cv::Mat* tchannelImage = new cv::Mat(static_cast<int>(temp->rows), static_cast<int>(temp->cols), CV_8UC1);
  cv::extractChannel(*clrImage, *tchannelImage, 0);
  // convert to float
  *channelImage = cv::Mat(static_cast<int>(temp->rows), static_cast<int>(temp->cols), FLOAT_MAT_TYPE);
  tchannelImage->convertTo(*channelImage, FLOAT_MAT_TYPE, 1./255);
  // destroy
  delete tchannelImage;
  delete temp;
}

/**
 * This function scales the input image to have values 0->1
 *
 * \param inImage the input image
 * \param outImage hte output iamge
 */
void mcvScaleMat(const cv::Mat *inMat, cv::Mat *outMat)
{
  //convert inMat type to outMat type
  inMat->convertTo(*outMat, outMat->type());
  //if (CV_MAT_DEPTH(inMat->type())
  //get the min and subtract it
  double min, max;
  cv::minMaxLoc(*inMat, &min, &max);
  *outMat -= min;
  *outMat /= max - min;
}

/**
 * This function creates a double matrix from an input vector
 *
 * \param vec the input vector
 * \param mat the output matrix (column vector)
 *
 */
template <class T>
cv::Mat* mcvVector2Mat(const vector<T> &vec)
{
  cv::Mat *mat = 0;

  if (vec.size()>0)
  {
    //create the matrix
    mat = new cv::Mat(vec.size(), 1, CV_64FC1);
    //loop and get values
    for (int i=0; i<static_cast<int>(vec.size()); i++)
      mat->at<double>(i, 0) =static_cast<double>(vec[i]);
  }

  //return
  return mat;
}

// template cv::Mat* mcvVector2Mat(const vector<double> &vec);
// template cv::Mat* mcvVector2Mat(const vector<int> &vec);

} // namespace LaneDetector
