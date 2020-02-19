/**
 * \file LaneDetector.cc
 * \author Mohamed Aly <malaa@caltech.edu>
 * \date Thu 26 Jul, 2007
 *
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wformat="
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#ifdef __GNUC__
#if __GNUC__ > 5
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#endif
#endif

#include "LaneDetector.hh"

#include "mcv.hh"
#include "InversePerspectiveMapping.hh"
#include "LaneDetectorOpt.h"
#include "ranker.h"

#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <math.h>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace LaneDetector
{

/**
 * This function filters the input image looking for horizontal
 * or vertical lines with specific cols or rows.
 *
 * \param inImage the input image
 * \param outImage the output image in IPM
 * \param wx cols of kernel window in x direction = 2*wx+1
 * (default 2)
 * \param wy cols of kernel window in y direction = 2*wy+1
 * (default 2)
 * \param sigmax std deviation of kernel in x (default 1)
 * \param sigmay std deviation of kernel in y (default 1)
 * \param lineType type of the line
 *      LINE_HORIZONTAL (default)
 *      LINE_VERTICAL
 */
 void mcvFilterLines(const cv::Mat *inImage, cv::Mat *outImage,
                     unsigned char wx, unsigned char wy, LD_FLOAT sigmax,
                     LD_FLOAT sigmay, LineType lineType)
{
    //define the two kernels
    //this is for 7-pixels wide
//     FLOAT_MAT_ELEM_TYPE derivp[] = {-2.328306e-10, -6.984919e-09, -1.008157e-07, -9.313226e-07, -6.178394e-06, -3.129616e-05, -1.255888e-04, -4.085824e-04, -1.092623e-03, -2.416329e-03, -4.408169e-03, -6.530620e-03, -7.510213e-03, -5.777087e-03, -5.777087e-04, 6.932504e-03, 1.372058e-02, 1.646470e-02, 1.372058e-02, 6.932504e-03, -5.777087e-04, -5.777087e-03, -7.510213e-03, -6.530620e-03, -4.408169e-03, -2.416329e-03, -1.092623e-03, -4.085824e-04, -1.255888e-04, -3.129616e-05, -6.178394e-06, -9.313226e-07, -1.008157e-07, -6.984919e-09, -2.328306e-10};
//     int derivLen = 35;
//     FLOAT_MAT_ELEM_TYPE smoothp[] = {2.384186e-07, 5.245209e-06, 5.507469e-05, 3.671646e-04, 1.744032e-03, 6.278515e-03, 1.778913e-02, 4.066086e-02, 7.623911e-02, 1.185942e-01, 1.541724e-01, 1.681881e-01, 1.541724e-01, 1.185942e-01, 7.623911e-02, 4.066086e-02, 1.778913e-02, 6.278515e-03, 1.744032e-03, 3.671646e-04, 5.507469e-05, 5.245209e-06, 2.384186e-07};
//     int smoothLen = 23;
  cv::Mat fx;
  cv::Mat fy;
  //create the convoultion kernel
  switch (lineType)
  {
    case LINE_HORIZONTAL:
    {
      //this is for 5-pixels wide
      FLOAT_MAT_ELEM_TYPE derivp[] = {-2.384186e-07, -4.768372e-06, -4.482269e-05, -2.622604e-04, -1.064777e-03, -3.157616e-03, -6.976128e-03, -1.136112e-02, -1.270652e-02, -6.776810e-03, 6.776810e-03, 2.156258e-02, 2.803135e-02, 2.156258e-02, 6.776810e-03, -6.776810e-03, -1.270652e-02, -1.136112e-02, -6.976128e-03, -3.157616e-03, -1.064777e-03, -2.622604e-04, -4.482269e-05, -4.768372e-06, -2.384186e-07};
      int derivLen = 25;
      FLOAT_MAT_ELEM_TYPE smoothp[] = {2.384186e-07, 5.245209e-06, 5.507469e-05, 3.671646e-04, 1.744032e-03, 6.278515e-03, 1.778913e-02, 4.066086e-02, 7.623911e-02, 1.185942e-01, 1.541724e-01, 1.681881e-01, 1.541724e-01, 1.185942e-01, 7.623911e-02, 4.066086e-02, 1.778913e-02, 6.278515e-03, 1.744032e-03, 3.671646e-04, 5.507469e-05, 5.245209e-06, 2.384186e-07};
      int smoothLen = 23;
      //horizontal is smoothing and vertical is derivative
      fx = cv::Mat(1, smoothLen, FLOAT_MAT_TYPE, smoothp);
      fy = cv::Mat(derivLen, 1, FLOAT_MAT_TYPE, derivp);
    }
    break;

    case LINE_VERTICAL:
    {
      //this is for 5-pixels wide
      FLOAT_MAT_ELEM_TYPE derivp[] = //{1.000000e-11, 8.800000e-10, 3.531000e-08, 8.536000e-07, 1.383415e-05, 1.581862e-04, 1.306992e-03, 7.852691e-03, 3.402475e-02, 1.038205e-01, 2.137474e-01, 2.781496e-01, 2.137474e-01, 1.038205e-01, 3.402475e-02, 7.852691e-03, 1.306992e-03, 1.581862e-04, 1.383415e-05, 8.536000e-07, 3.531000e-08, 8.800000e-10, 1.000000e-11};
          //{1.000000e-06, 4.800000e-05, 9.660000e-04, 1.048000e-02, 6.529500e-02, 2.278080e-01, 3.908040e-01, 2.278080e-01, 6.529500e-02, 1.048000e-02, 9.660000e-04, 4.800000e-05, 1.000000e-06};
          {1.000000e-16, 1.280000e-14, 7.696000e-13, 2.886400e-11, 7.562360e-10, 1.468714e-08, 2.189405e-07, 2.558828e-06, 2.374101e-05, 1.759328e-04, 1.042202e-03, 4.915650e-03, 1.829620e-02, 5.297748e-02, 1.169560e-01, 1.918578e-01, 2.275044e-01, 1.918578e-01, 1.169560e-01, 5.297748e-02, 1.829620e-02, 4.915650e-03, 1.042202e-03, 1.759328e-04, 2.374101e-05, 2.558828e-06, 2.189405e-07, 1.468714e-08, 7.562360e-10, 2.886400e-11, 7.696000e-13, 1.280000e-14, 1.000000e-16};
      int derivLen = 33; //23; 13; 33;
      FLOAT_MAT_ELEM_TYPE smoothp[] = {-1.000000e-03, -2.200000e-02, -1.480000e-01, -1.940000e-01, 7.300000e-01, -1.940000e-01, -1.480000e-01, -2.200000e-02, -1.000000e-03};
      //{-1.000000e-07, -5.400000e-06, -1.240000e-04, -1.561000e-03, -1.149400e-02, -4.787020e-02, -9.073680e-02, 2.144300e-02, 2.606970e-01, 2.144300e-02, -9.073680e-02, -4.787020e-02, -1.149400e-02, -1.561000e-03, -1.240000e-04, -5.400000e-06, -1.000000e-07};
      int smoothLen = 9; //9; 17;
      //horizontal is derivative and vertical is smoothing
      fy = cv::Mat(1, smoothLen, FLOAT_MAT_TYPE, smoothp);
      fx = cv::Mat(derivLen, 1, FLOAT_MAT_TYPE, derivp);
    }
    break;
  }

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
  //SHOW_MAT(kernel, "Kernel:");
  }//#endif

//#warning "still check subtracting mean from image"
  //subtract mean
  cv::Scalar mean = cv::mean(*inImage);
  *outImage = *inImage - mean;


  //do the filtering
  cv::filter2D(*outImage, *outImage, -1, fx); //inImage outImage
  cv::filter2D(*outImage, *outImage, -1, fy);


//     cv::Mat *deriv = new cv::Mat
//     //define x
//     cv::Mat *x = new cv::Mat(2*wx+1, 1, FLOAT_MAT_TYPE);
//     //define y
//     cv::Mat *y = new cv::Mat(2*wy+1, 1, FLOAT_MAT_TYPE);

//     //create the convoultion kernel
//     switch (lineType)
//     {
//         case FILTER_LINE_HORIZONTAL:
//             //guassian in x direction
// 	    mcvGetGaussianKernel(x, wx, sigmax);
//             //derivative of guassian in y direction
//             mcvGet2DerivativeGaussianKernel(y, wy, sigmay);
//             break;

//         case FILTER_LINE_VERTICAL:
//             //guassian in y direction
//             mcvGetGaussianKernel(y, wy, sigmay);
//             //derivative of guassian in x direction
//             mcvGet2DerivativeGaussianKernel(x, wx, sigmax);
//             break;
//     }

//     //combine the 2D kernel
//     cv::Mat *kernel = new cv::Mat(2*wy+1, 2*wx+1, FLOAT_MAT_TYPE);
//     cv::gEMM(y, x, 1, 0, 1, kernel, CV_GEMM_B_T);

//     //subtract the mean
//     cv::Scalar mean = cv::mean(*kernel);
//     *kernel = *kernel - mean;

//     #ifdef DEBUG_GET_STOP_LINES
//     //SHOW_MAT(kernel, "Kernel:");
//     #endif

//     //do the filtering
//     cv::filter2D(inImage, outImage, kernel);

//     //clean
//     delete x;
//     delete y;
//     delete kernel;
}

/**
 * This function gets a 1-D gaussian filter with specified
 * std deviation and range
 *
 * \param kernel input mat to hold the kernel (2*w+1x1)
 *      column vector (already allocated)
 * \param w cols of kernel is 2*w+1
 * \param sigma std deviation
 */
void mcvGetGaussianKernel(cv::Mat *kernel, unsigned char w, LD_FLOAT sigma)
{
  //get variance
  sigma *= sigma;

  //get the kernel
  for (double i=-w; i<=w; i++)
      kernel->at<FLOAT_MAT_ELEM_TYPE>(int(i+w), 0) =
          (FLOAT_MAT_ELEM_TYPE) exp(-(.5/sigma)*(i*i));
}

/**
 * This function gets a 1-D second derivative gaussian filter
 * with specified std deviation and range
 *
 * \param kernel input mat to hold the kernel (2*w+1x1)
 *      column vector (already allocated)
 * \param w cols of kernel is 2*w+1
 * \param sigma std deviation
 */
void mcvGet2DerivativeGaussianKernel(cv::Mat *kernel,
                                     unsigned char w, LD_FLOAT sigma)
{
  //get variance
  sigma *= sigma;

  //get the kernel
  for (double i=-w; i<=w; i++)
      kernel->at<FLOAT_MAT_ELEM_TYPE>(int(i+w), 0) =
          (FLOAT_MAT_ELEM_TYPE)
          (exp(-.5*i*i)/sigma - (i*i)*exp(-(.5/sigma)*i*i)/(sigma*sigma));
}


/** This function groups the input filtered image into
 * horizontal or vertical lines.
 *
 * \param inImage input image
 * \param lines returned detected lines (vector of points)
 * \param lineScores scores of the detected lines (vector of floats)
 * \param lineType type of lines to detect
 *      LINE_HORIZONTAL (default) or LINE_VERTICAL
 * \param linePixelWidth cols (or rows) of lines to detect
 * \param localMaxima whether to detect local maxima or just get
 *      the maximum
 * \param detectionThreshold threshold for detection
 * \param smoothScores whether to smooth scores detected or not
 */
void mcvGetHVLines(const cv::Mat *inImage, vector <Line> *lines,
                   vector <LD_FLOAT> *lineScores, LineType lineType,
                   LD_FLOAT linePixelWidth, bool binarize, bool localMaxima,
                   LD_FLOAT detectionThreshold, bool smoothScores)
{
  cv::Mat *image;
  //binarize input image if to binarize
  if (binarize)
  {
    //mcvBinarizeImage(image);
    image = new cv::Mat(inImage->rows, inImage->cols, INT_MAT_TYPE);
    cv::threshold(*inImage, *image, 0, 1, cv::THRESH_BINARY); //0.05
  }
  else
  {
    image = new cv::Mat();
    *image = inImage->clone();
  }

  //get sum of lines through horizontal or vertical
  //sumLines is a column vector
  cv::Mat sumLines, *sumLinesp;
  int maxLineLoc = 0;
  switch (lineType)
  {
    case LINE_HORIZONTAL:
      sumLinesp = new cv::Mat(image->rows, 1, FLOAT_MAT_TYPE);
      cv::reduce(*image, *sumLinesp, 1, cv::REDUCE_SUM); //_AVG
      sumLines = sumLinesp->reshape(0, 0);
      //max location for a detected line
      maxLineLoc = image->rows-1;
      break;
    case LINE_VERTICAL:
      sumLinesp = new cv::Mat(1, image->cols, FLOAT_MAT_TYPE);
      cv::reduce(*image, *sumLinesp, 0, cv::REDUCE_SUM); //_AVG
      sumLines = sumLinesp->reshape(0, image->cols);
      //max location for a detected line
      maxLineLoc = image->cols-1;
      break;
  }
    //SHOW_MAT(&sumLines, "sumLines:");

    //smooth it

  float smoothp[] =	{
    0.000003726653172, 0.000040065297393, 0.000335462627903, 0.002187491118183,
    0.011108996538242, 0.043936933623407, 0.135335283236613, 0.324652467358350,
    0.606530659712633, 0.882496902584595, 1.000000000000000, 0.882496902584595,
    0.606530659712633, 0.324652467358350, 0.135335283236613, 0.043936933623407,
    0.011108996538242, 0.002187491118183, 0.000335462627903, 0.000040065297393,
    0.000003726653172};
// 	{0.000004,0.000010,0.000025,0.000063,0.000148,0.000335,0.000732,0.001534,0.003089,0.005976,0.011109,0.019841,0.034047,0.056135,0.088922,0.135335,0.197899,0.278037,0.375311,0.486752,0.606531,0.726149,0.835270,0.923116,0.980199,1.000000,0.980199,0.923116,0.835270,0.726149,0.606531,0.486752,0.375311,0.278037,0.197899,0.135335,0.088922,0.056135,0.034047,0.019841,0.011109,0.005976,0.003089,0.001534,0.000732,0.000335,0.000148,0.000063,0.000025,0.000010,0.000004};
  int smoothWidth = 21; //21; 51;
  cv::Mat smooth = cv::Mat(1, smoothWidth, CV_32FC1, smoothp);
  if (smoothScores)
    cv::filter2D(*&sumLines, *&sumLines, -1, smooth);
//     SHOW_MAT(&sumLines, "sumLines:");



  //get the max and its location
  vector <int> sumLinesMaxLoc;
  vector <double> sumLinesMax;
  int maxLoc; double max;
  //TODO: put the ignore in conf
  #define MAX_IGNORE 0 //(int(smoothWidth/2.)+1)
  #define LOCAL_MAX_IGNORE (int(MAX_IGNORE/4))
  mcvGetVectorMax(&sumLines, &max, &maxLoc, MAX_IGNORE);

  //put the local maxima stuff here
  if (localMaxima)
  {
    //loop to get local maxima
    for(int i=1+LOCAL_MAX_IGNORE; i<sumLines.rows-1-LOCAL_MAX_IGNORE; i++)
    {
	    //get that value
	    LD_FLOAT val = sumLines.at<FLOAT_MAT_ELEM_TYPE>(i, 0);
	    //check if local maximum
	    if( (val > sumLines.at<FLOAT_MAT_ELEM_TYPE>(i-1, 0))
        && (val > sumLines.at<FLOAT_MAT_ELEM_TYPE>(i+1, 0))
        //		&& (i != maxLoc)
        && (val >= detectionThreshold) )
	    {
        //iterators for the two vectors
        vector<double>::iterator j;
        vector<int>::iterator k;
        //loop till we find the place to put it in descendingly
        for(j=sumLinesMax.begin(), k=sumLinesMaxLoc.begin();
            j != sumLinesMax.end()  && val<= *j; j++,k++);
        //add its index
        sumLinesMax.insert(j, val);
        sumLinesMaxLoc.insert(k, i);
	    }
    }
  }

  //check if didnt find local maxima
  if(sumLinesMax.size()==0 && max>detectionThreshold)
  {
    //put maximum
    sumLinesMaxLoc.push_back(maxLoc);
    sumLinesMax.push_back(max);
  }

//     //sort it descendingly
//     sort(sumLinesMax.begin(), sumLinesMax.end(), greater<double>());
//     //sort the indices
//     for (int i=0; i<(int)sumLinesMax.size(); i++)
// 	for (int j=i; j<(int)sumLinesMax.size(); j++)
// 	    if(sumLinesMax[i] == CV_MAT_ELEM(sumLines, FLOAT_MAT_ELEM_TYPE,
// 					     sumLinesMaxLoc[j], 0))
// 	    {
// 		int k = sumLinesMaxLoc[j];
// 		sumLinesMaxLoc[j] = sumLinesMaxLoc[i];
// 		sumLinesMaxLoc[i] = k;
// 	    }
//     //sort(sumLinesMaxLoc.begin(), sumLinesMaxLoc.end(), greater<int>());

    //plot the line scores and the local maxima
    //if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
//     gnuplot_ctrl *h =  mcvPlotMat1D(nullptr, &sumLines, "Line Scores");
//     cv::Mat *y = mcvVector2Mat(sumLinesMax);
//     cv::Mat *x =  mcvVector2Mat(sumLinesMaxLoc);
//     mcvPlotMat2D(h, x, y);
//     //gnuplot_plot_xy(h, (double*)&sumLinesMaxLoc,(double*)&sumLinesMax, sumLinesMax.size(),"");
//     cin.get();
//     gnuplot_close(h);
//     delete x;
//     delete y;
//}//#endif
  //process the found maxima
  for (int i=0; i<(int)sumLinesMax.size(); i++)
  {
    //get subpixel accuracy
    double maxLocAcc = mcvGetLocalMaxSubPixel(
      sumLines.at<FLOAT_MAT_ELEM_TYPE>(MAX(sumLinesMaxLoc[i]-1,0), 0),
      sumLines.at<FLOAT_MAT_ELEM_TYPE>(sumLinesMaxLoc[i], 0),
      sumLines.at<FLOAT_MAT_ELEM_TYPE>(MIN(sumLinesMaxLoc[i]+1,maxLineLoc), 0) );
    maxLocAcc += sumLinesMaxLoc[i];
    maxLocAcc = MIN(MAX(0, maxLocAcc), maxLineLoc);


	//TODO: get line extent

	//put the extracted line
    Line line;
    switch (lineType)
    {
      case LINE_HORIZONTAL:
        line.startPoint.x = 0.5;
        line.startPoint.y = (LD_FLOAT)maxLocAcc + .5;//sumLinesMaxLoc[i]+.5;
        line.endPoint.x = inImage->cols-.5;
        line.endPoint.y = line.startPoint.y;
        break;
      case LINE_VERTICAL:
        line.startPoint.x = (LD_FLOAT)maxLocAcc + .5;//sumLinesMaxLoc[i]+.5;
        line.startPoint.y = .5;
        line.endPoint.x = line.startPoint.x;
        line.endPoint.y = inImage->rows-.5;
        break;
    }
    (*lines).push_back(line);
    if (lineScores)
        (*lineScores).push_back(sumLinesMax[i]);
  }//for

  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    cv::Mat *im, *im2 = new cv::Mat();
    *im2 = image->clone();
    if (binarize)
      im2->convertTo(*im2, 255, 0);

    if (binarize)
	    im = new cv::Mat(image->rows, image->cols, CV_8UC3);
    else
	    im = new cv::Mat(image->rows, image->cols, CV_32FC3);
    mcvScaleMat(im2, im2);
    cv::cvtColor(*im2, *im, cv::COLOR_GRAY2RGB);
    for (unsigned int i=0; i<lines->size(); i++)
    {
      Line line = (*lines)[i];
      mcvIntersectLineWithBB(&line, cv::Size(image->cols, image->rows), &line);
      if (binarize)
        mcvDrawLine(im, line, CV_RGB(255,0,0), 1);
      else
        mcvDrawLine(im, line, CV_RGB(1,0,0), 1);
    }

    char str[256];
    switch (lineType)
    {
      case LINE_HORIZONTAL:
        sprintf(str, "%s", "Horizontal Lines");
        break;
      case LINE_VERTICAL:
        sprintf(str, "%s", "Vertical Lines");
        break;
    }
    SHOW_IMAGE(im, str, 10);
    delete im;
    delete im2;
  }

  //clean
  delete sumLinesp;
  //delete smooth;
  sumLinesMax.clear();
  sumLinesMaxLoc.clear();
  delete image;
}


/** This function detects lines in images using Hough transform
 *
 * \param inImage input image
 * \param lines vector of lines to hold the results
 * \param lineScores scores of the detected lines (vector of floats)
 * \param rMin minimum r use for finding the lines (default 0)
 * \param rMax maximum r to find (default max(size(im)))
 * \param rStep step to use for binning (default is 2)
 * \param thetaMin minimum angle theta to look for (default 0) all in radians
 * \param thetaMax maximum angle theta to look for (default 2*pi)
 * \param thetaStep step to use for binning theta (default 5)
 * \param  binarize if to binarize the input image or use the raw values so that
 *	non-zero values are not treated as equal
 * \param localMaxima whether to detect local maxima or just get
 *      the maximum
 * \param detectionThreshold threshold for detection
 * \param smoothScores whether to smooth scores detected or not
 * \param group whether to group nearby detections (1) or not (0 default)
 * \param groupThreshold the minimum distance used for grouping (default 10)
 */

void mcvGetHoughTransformLines(const cv::Mat *inImage, vector <Line> *lines,
                               vector <LD_FLOAT> *lineScores,
                               LD_FLOAT rMin, LD_FLOAT rMax, LD_FLOAT rStep,
                               LD_FLOAT thetaMin, LD_FLOAT thetaMax,
                               LD_FLOAT thetaStep, bool binarize, bool localMaxima,
                               LD_FLOAT detectionThreshold, bool smoothScores,
                               bool group, LD_FLOAT groupThreshold)
{
  cv::Mat *image;

  //binarize input image if to binarize
  if (!binarize)
  {
    image = new cv::Mat();
    *image = inImage->clone();
    assert(image!=0);
    //         mcvBinarizeImage(image);
  }
  //binarize input image
  else
  {
    image = new cv::Mat(inImage->rows, inImage->cols, INT_MAT_TYPE);
    cv::threshold(*inImage, *image, 0, 1, cv::THRESH_BINARY); //0.05
    //get max of image
    //double maxval, minval;
    //cv::minMaxLoc(inImage, &minval, &maxval);
    //cout << "Max = " << maxval << "& Min=" << minval << "\n";
    //cv::Scalar mean = cv::mean(*inImage);
    //cout << "Mean=" << mean.val[0] << "\n";
  }

  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    SHOW_IMAGE(image, "Hough thresholded image", 10);
  }//#endif

  //define the accumulator array: rows correspond to r and columns to theta
  int rBins = int((rMax-rMin)/rStep);
  int thetaBins = int((thetaMax-thetaMin)/thetaStep);
  cv::Mat *houghSpace = new cv::Mat(rBins, thetaBins, CV_MAT_TYPE(image->type())); //FLOAT_MAT_TYPE);
  assert(houghSpace!=0);
  //init to zero
  houghSpace->setTo(0);

  //init values of r and theta
  LD_FLOAT *rs = new LD_FLOAT[rBins];
  LD_FLOAT *thetas = new LD_FLOAT[thetaBins];
  LD_FLOAT r, theta;
  int ri, thetai;
  for (r=rMin+rStep/2,  ri=0 ; ri<rBins; ri++,r+=rStep)
    rs[ri] = r;
  for (theta=thetaMin, thetai=0 ; thetai<thetaBins; thetai++,
    theta+=thetaStep)
    thetas[thetai] = theta;

  //get non-zero points in the image
  int nzCount = cv::countNonZero(*image);
  cv::Mat *nzPoints = new cv::Mat(nzCount, 2, CV_32SC1);
  int idx = 0;
  for (int i=0; i<image->cols; i++)
    for (int j=0; j<image->rows; j++)
      if ( image->at<float>(j, i) )
      {
        nzPoints->at<int>(idx, 0) = i;
        nzPoints->at<int>(idx, 1) = j;
        idx++;
      }

    //calculate r values for all theta and all points
    //cv::Mat *rPoints = new cv::Mat(image->cols*image->rows, thetaBins, CV_32SC1);//FLOAT_MAT_TYPE)
    //cv::Mat *rPoints = new cv::Mat(nzCount, thetaBins, CV_32SC1);//FLOAT_MAT_TYPE);
    //cv::set(rPoints, -1);
    //loop on x
    //float x=0.5, y=0.5;
    int i, k; //j
    for (i=0; i<nzCount; i++)
      for (k=0; k<thetaBins; k++)
      {
        //compute the r value for that point and that theta
        theta = thetas[k];
        float rval = nzPoints->at<int>(i, 0) * cos(theta) +
        nzPoints->at<int>(i, 1) * sin(theta); //x y
        int r = (int)( ( rval - rMin) / rStep);
        //	    rPoints->at<int>(i, k) =
        //(int)( ( rval - rMin) / rStep);

        //accumulate in the hough space if a valid value
        if (r>=0 && r<rBins) {
          if(binarize)
            houghSpace->at<INT_MAT_ELEM_TYPE>(r, k)++;
          //image->at<INT_MAT_ELEM_TYPE>(j, i);
        }
        else {
          houghSpace->at<FLOAT_MAT_ELEM_TYPE>(r, k)+=
          image->at<FLOAT_MAT_ELEM_TYPE>(nzPoints->at<int>(i, 1), nzPoints->at<int>(i, 0));
        }
      }

      //clear
      delete nzPoints;

//     bool inside;
//     for (i=0; i<image->cols; i++) //x=0; x++
// 	//loop on y
// 	for (j=0; j<image->rows; j++) //y=0 y++
// 	    //loop on theta
// 	    for (k=0; k<thetaBins; k++)
// 	    {
// 		//compute the r value and then subtract rMin and div by rStep
// 		//to get the r bin index to which it belongs (0->rBins-1)
// 		if (lineConf->binarize && image->at<INT_MAT_ELEM_TYPE>(j, i) !=0)
// 		    inside = true;
// 		else if (!lineConf->binarize && CV_MAT_ELEM(*image, FLOAT_MAT_ELEM_TYPE,
// 							    j, i) !=0)
// 		    inside = true;
// 		else
// 		    inside = false;
// 		if (inside)
// 		{
// 		    theta = thetas[k];
// 		    float rval = i * cos(theta) + j * sin(theta); //x y
// 		    CV_MAT_ELEM(*rPoints, int,
// 				i*image->rows + j, k) =
// 			(int)( ( rval - lineConf->rMin) / lineConf->rStep);
// 		}

// 	    }

//      SHOW_MAT(rPoints, "rPoints");
//      cin.get();

    //now we should accumulate the values into the approprate bins in the houghSpace
//     for (ri=0; ri<rBins; ri++)
// 	for (thetai=0; thetai<thetaBins; thetai++)
// 	    for (i=0; i<image->cols; i++)
// 		for (j=0; j<image->rows; j++)
// 		{
// 		    //check if this cell belongs to that bin or not
// 		    if (CV_MAT_ELEM(*rPoints, int,
// 				    i*image->rows + j , thetai)==ri)
// 		    {
// 			if(lineConf->binarize)
// 			    houghSpace->at<INT_MAT_ELEM_TYPE>(ri, thetai)++;
// 			//image->at<INT_MAT_ELEM_TYPE>(j, i);
// 			else
// 			    houghSpace->at<FLOAT_MAT_ELEM_TYPE>(ri, thetai)+=
// 				image->at<FLOAT_MAT_ELEM_TYPE>(j, i);
// 		    }
// 		}


  //smooth hough transform
  if (smoothScores)
//    cv::smooth(houghSpace, houghSpace, CV_GAUSSIAN, 3);
      cv::GaussianBlur(*houghSpace, *houghSpace, {3, 3}, 0);

  //get local maxima
  vector <double> maxLineScores;
  vector <cv::Point> maxLineLocs;
  if (localMaxima)
  {
    //get local maxima in the hough space
    mcvGetMatLocalMax(houghSpace, maxLineScores, maxLineLocs, detectionThreshold);
  }
  else
  {
    //get the maxima above the threshold
    mcvGetMatMax(houghSpace, maxLineScores, maxLineLocs, detectionThreshold);
  }

  //get the maximum value
  double maxLineScore;
  cv::Point maxLineLoc;
  cv::minMaxLoc(*houghSpace, 0, &maxLineScore);
  if (maxLineScores.size()==0 && maxLineScore>=detectionThreshold)
  {
    maxLineScores.push_back(maxLineScore);
    maxLineLocs.push_back(maxLineLoc);
  }


  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    // 	cout << "Local maxima = " << maxLineScores.size() << "\n";

    {
      cv::Mat *im, *im2 = new cv::Mat();
      *im2 = image->clone();
      if (binarize)
        im2->convertTo(*im2, 255, 0);

      if (binarize)
        im = new cv::Mat(image->rows, image->cols, CV_8UC3);//cv::cloneMat(image);
      else
        im = new cv::Mat(image->rows, image->cols, CV_32FC3);
      cv::cvtColor(*im2, *im, cv::COLOR_GRAY2RGB);
      for (int i=0; i<(int)maxLineScores.size(); i++)
      {
        Line line;
        assert(maxLineLocs[i].x>=0 && maxLineLocs[i].x<thetaBins);
        assert(maxLineLocs[i].y>=0 && maxLineLocs[i].y<rBins);
        mcvIntersectLineRThetaWithBB(rs[maxLineLocs[i].y], thetas[maxLineLocs[i].x],
                                    cv::Size(image->cols, image->rows), &line);
                                    if (binarize)
                                      mcvDrawLine(im, line, CV_RGB(255,0,0), 1);
                                    else
                                      mcvDrawLine(im, line, CV_RGB(1,0,0), 1);
      }
      SHOW_IMAGE(im, "Hough before grouping", 10);
      delete im;
      delete im2;

      // 	    //debug
      // 	    cout << "Maxima detected:\n";
      // 	    for(int ii=0; ii<(int)maxLineScores.size(); ii++)
      // 		cout << " " << maxLineScores[ii];
      // 	    cout << "\n";
    }
  }//#endif

  //group detected maxima
  if (group && maxLineScores.size()>1)
  {
    //flag for stopping
    bool stop = false;
    while (!stop)
    {
      //minimum distance so far
      float minDist = groupThreshold+5, dist = 0.;
      vector<cv::Point>::iterator iloc, jloc,
      minIloc=maxLineLocs.begin(), minJloc=minIloc+1;
      vector<double>::iterator iscore, jscore, minIscore, minJscore;
      //compute pairwise distance between detected maxima
      for (iloc=maxLineLocs.begin(), iscore=maxLineScores.begin();
      iloc!=maxLineLocs.end(); iloc++, iscore++)
      for (jscore=iscore+1, jloc=iloc+1; jscore!=maxLineScores.end();
           jloc++, jscore++)
      {
        //add pi if neg
        float t1 = thetas[iloc->x]<0 ? thetas[iloc->x] : thetas[iloc->x]+CV_PI;
        float t2 = thetas[jloc->x]<0 ? thetas[jloc->x] : thetas[jloc->x]+CV_PI;
        //get distance
        dist = fabs(rs[iloc->y]-rs[jloc->y]) +
        0.1 * fabs(t1 - t2);//fabs(thetas[iloc->x]-thetas[jloc->x]);
        //check if minimum
        if (dist<minDist)
        {
          minDist = dist;
          minIloc = iloc; minIscore = iscore;
          minJloc = jloc; minJscore = jscore;
        }
      }
      //check if minimum distance is less than groupThreshold
      if (minDist >= groupThreshold)
        stop = true;
      else
      {
        // 		//debug
        //  		cout << "Before grouping:\n";
        //  		for(int ii=0; ii<(int)maxLineScores.size(); ii++)
        //  		    cout << " " << maxLineScores[ii];
        //  		cout << "\n";

        //combine the two minimum ones with weighted average of
        //their scores
        double x =  (minIloc->x * *minIscore + minJloc->x * *minJscore) /
        (*minIscore + *minJscore);
        double y =  (minIloc->y * *minIscore + minJloc->y * *minJscore) /
        (*minIscore + *minJscore);
        //put into the first
        minIloc->x = (int)x;// ((minJloc->x + minJloc->x)/2.0); // (int) x;
        minIloc->y = (int)y;// ((minJloc->y + minIloc->y)/2.0); // (int) y;
        *minIscore = (*minJscore + *minIscore)/2;///2;
        //delete second one
        maxLineLocs.erase(minJloc);
        maxLineScores.erase(minJscore);

        // 		//debug
        //  		cout << "Before sorting:\n";
        //  		for(int ii=0; ii<(int)maxLineScores.size(); ii++)
        //  		    cout << " " << maxLineScores[ii];
        //  		cout << "\n";

        //check if to put somewhere else depending on the changed score
        for (iscore=maxLineScores.begin(), iloc=maxLineLocs.begin();
             iscore!=maxLineScores.end() && *minIscore <= *iscore;
             iscore++, iloc++);
        //swap the original location if different
        if (iscore!=minIscore )
        {
          //insert in new position
          maxLineScores.insert(iscore, *minIscore);
          maxLineLocs.insert(iloc, *minIloc);
          //delte old
          maxLineScores.erase(minIscore);
          maxLineLocs.erase(minIloc);
          // 		    //if at end, then back up one position
          // 		    if(iscore == maxLineScores.end())
          // 		    {
            // 			iscore--;
            // 			iloc--;
            // 		    }
            // 		    cv::Point tloc;
            // 		    double tscore;
            // 		    //swap
            // 		    tloc = *minIloc;
            // 		    tscore = *minIscore;

            // 		    *minIloc = *iloc;
            // 		    *minIscore = *iscore;

            // 		    *iloc = tloc;
            // 		    *iscore = tscore;
        }
        //  		cout << "after sorting:\n";
        //  		for(int ii=0; ii<(int)maxLineScores.size(); ii++)
        //  		    cout << " " << maxLineScores[ii];
        //  		cout << "\n";
      }
    }
  }

  if(DEBUG_LINES) //#ifdef DEBUG_GET_STOP_LINES
  {
    cv::Mat *im, *im2 = new cv::Mat();
    *im2 = image->clone();
    if (binarize)
      im2->convertTo(*im2, 255, 0);
    if (binarize)
      im = new cv::Mat(image->rows, image->cols, CV_8UC3);//cv::cloneMat(image);
    else
      im = new cv::Mat(image->rows, image->cols, CV_32FC3);
    cv::cvtColor(*im2, *im, cv::COLOR_GRAY2RGB);
    for (int i=0; i<(int)maxLineScores.size(); i++)
    {
      Line line;
      assert(maxLineLocs[i].x>=0 && maxLineLocs[i].x<thetaBins);
      assert(maxLineLocs[i].y>=0 && maxLineLocs[i].y<rBins);
      mcvIntersectLineRThetaWithBB(rs[maxLineLocs[i].y],
                                   thetas[maxLineLocs[i].x],
                                   cv::Size(image->cols, image->rows), &line);
      if (binarize)
        mcvDrawLine(im, line, CV_RGB(255,0,0), 1);
      else
        mcvDrawLine(im, line, CV_RGB(1,0,0), 1);
    }
    SHOW_IMAGE(im, "Hough after grouping", 10);
    delete im;
    delete im2;

    //put local maxima in image
    cv::Mat *houghSpaceClr;
    if(binarize)
      houghSpaceClr = new cv::Mat(houghSpace->rows, houghSpace->cols,
                                  CV_8UC3);
    else
      houghSpaceClr = new cv::Mat(houghSpace->rows, houghSpace->cols,
                                  CV_32FC3);
    mcvScaleMat(houghSpace, houghSpace);
    cv::cvtColor(*houghSpace, *houghSpaceClr, cv::COLOR_GRAY2RGB);
    for (int i=0; i<(int)maxLineLocs.size(); i++)
      cv::circle(*houghSpaceClr, cv::Point(maxLineLocs[i].x, maxLineLocs[i].y),
              1, CV_RGB(1, 0, 0), -1);
              // 	    if(lineConf->binarize)
              // 		CV_MAT_ELEM(*houghSpace, unsigned char, maxLineLocs[i].y,
              // 			    maxLineLocs[i].x) = 255;
              // 	    else
              // 		houghSpace->at<float>(maxLineLocs[i].y, maxLineLocs[i].x) = 1.f;
              //show the hough space
    SHOW_IMAGE(houghSpaceClr, "Hough Space", 10);
    delete houghSpaceClr;
              //SHOW_MAT(houghSpace, "Hough Space:");
  }//#endif

  //process detected maxima and return detected line(s)
  for(int i=0; i<int(maxLineScores.size()); i++)
  {
    //check if above threshold
    if (maxLineScores[i]>=detectionThreshold)
    {
      //get sub-pixel accuracy
      //
      //get the two end points from the r-theta
      Line line;
      assert(maxLineLocs[i].x>=0 && maxLineLocs[i].x<thetaBins);
      assert(maxLineLocs[i].y>=0 && maxLineLocs[i].y<rBins);
      mcvIntersectLineRThetaWithBB(rs[maxLineLocs[i].y],
                                   thetas[maxLineLocs[i].x],
                                   cv::Size(image->cols, image->rows), &line);
      //get line extent
      //put the extracted line
      lines->push_back(line);
      if (lineScores)
        (*lineScores).push_back(maxLineScores[i]);

    }
    //not above threshold
    else
    {
      //exit out of the for loop, as the scores are sorted descendingly
      break;
    }
  }

  //clean up
  delete image;
  delete houghSpace;
  //delete rPoints;
  delete [] rs;
  delete [] thetas;
  maxLineScores.clear();
  maxLineLocs.clear();
}



/** This function binarizes the input image i.e. nonzero elements
 * becomen 1 and others are 0.
 *
 * \param inImage input & output image
 */
void mcvBinarizeImage(cv::Mat *inImage)
{

  if (CV_MAT_TYPE(inImage->type())==FLOAT_MAT_TYPE)
  {
    for (int i=0; i<inImage->rows; i++)
      for (int j=0; j<inImage->cols; j++)
        if (inImage->at<FLOAT_MAT_ELEM_TYPE>(i, j) != 0.f)
          inImage->at<FLOAT_MAT_ELEM_TYPE>(i, j)=1;
  }
  else if (CV_MAT_TYPE(inImage->type())==INT_MAT_TYPE)
  {
    for (int i=0; i<inImage->rows; i++)
      for (int j=0; j<inImage->cols; j++)
        if (inImage->at<INT_MAT_ELEM_TYPE>(i, j) != 0)
          inImage->at<INT_MAT_ELEM_TYPE>(i, j)=1;
  }
  else
  {
    cerr << "Unsupported type in mcvBinarizeImage\n";
    exit(1);
  }
}


/** This function gets the maximum value in a vector (row or column)
 * and its location
 *
 * \param inVector the input vector
 * \param max the output max value
 * \param maxLoc the location (index) of the first max
 *
 */
#define MCV_VECTOR_MAX(type)  \
    /*row vector*/ \
    if (inVector->rows==1) \
    { \
        /*initial value*/ \
        tmax = (double) inVector->at<type>(0, inVector->cols-1); \
        tmaxLoc = inVector->cols-1; \
        /*loop*/ \
        for (int i=inVector->cols-1-ignore; i>=0+ignore; i--) \
        { \
            if (tmax<inVector->at<type>(0, i)) \
            { \
                tmax = inVector->at<type>(0, i); \
                tmaxLoc = i; \
            } \
        } \
    } \
    /*column vector */ \
    else \
    { \
        /*initial value*/ \
        tmax = (double) inVector->at<type>(inVector->rows-1, 0); \
        tmaxLoc = inVector->rows-1; \
        /*loop*/ \
        for (int i=inVector->rows-1-ignore; i>=0+ignore; i--) \
        { \
            if (tmax<inVector->at<type>(i, 0)) \
            { \
                tmax = (double) inVector->at<type>(i, 0); \
                tmaxLoc = i; \
            } \
        } \
    } \

void mcvGetVectorMax(const cv::Mat *inVector, double *max, int *maxLoc, int ignore)
{
  double tmax;
  int tmaxLoc;

  if (CV_MAT_TYPE(inVector->type())==FLOAT_MAT_TYPE)
  {
    MCV_VECTOR_MAX(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inVector->type())==INT_MAT_TYPE)
  {
    MCV_VECTOR_MAX(INT_MAT_ELEM_TYPE)
  }
  else
  {
    cerr << "Unsupported type in mcvGetVectorMax\n";
    exit(1);
  }

  //return
  if (max)
      *max = tmax;
  if (maxLoc)
      *maxLoc = tmaxLoc;
}



/** This function gets the local maxima in a matrix and their positions
 *  and its location
 *
 * \param inMat input matrix
 * \param localMaxima the output vector of local maxima
 * \param localMaximaLoc the vector of locations of the local maxima,
 *       where each location is cv::Point(x=col, y=row) zero-based
 *
 */

void mcvGetMatLocalMax(const cv::Mat *inMat, vector<double> &localMaxima,
		     vector<cv::Point> &localMaximaLoc, double threshold)
{

  double val;

#define MCV_MAT_LOCAL_MAX(type)  \
    /*loop on the matrix and get points that are larger than their*/ \
    /*neighboring 8 pixels*/ \
    for(int i=1; i<inMat->rows-1; i++) \
	for (int j=1; j<inMat->cols-1; j++) \
	{ \
	    /*get the current value*/ \
	    val = inMat->at<type>(i, j); \
	    /*check if it's larger than all its neighbors*/ \
	    if( val > inMat->at<type>(i-1, j-1) && \
		val > inMat->at<type>(i-1, j) && \
		val > inMat->at<type>(i-1, j+1) && \
		val > inMat->at<type>(i, j-1) && \
		val > inMat->at<type>(i, j+1) && \
		val > inMat->at<type>(i+1, j-1) && \
		val > inMat->at<type>(i+1, j) && \
		val > inMat->at<type>(i+1, j+1) && \
                val >= threshold) \
	    { \
		/*found a local maxima, put it in the return vector*/ \
		/*in decending order*/ \
		/*iterators for the two vectors*/ \
		vector<double>::iterator k; \
		vector<cv::Point>::iterator l; \
		/*loop till we find the place to put it in descendingly*/ \
		for(k=localMaxima.begin(), l=localMaximaLoc.begin(); \
		    k != localMaxima.end()  && val<= *k; k++,l++); \
		/*add its index*/ \
		localMaxima.insert(k, val); \
		localMaximaLoc.insert(l, cv::Point(j, i)); \
	    } \
	}

  //check type
  if (CV_MAT_TYPE(inMat->type())==FLOAT_MAT_TYPE)
  {
    MCV_MAT_LOCAL_MAX(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type())==INT_MAT_TYPE)
  {
    MCV_MAT_LOCAL_MAX(INT_MAT_ELEM_TYPE)
  }
  else
  {
    cerr << "Unsupported type in mcvGetMatLocalMax\n";
    exit(1);
  }
}

/** This function gets the locations and values of all points
 * above a certain threshold
 *
 * \param inMat input matrix
 * \param maxima the output vector of maxima
 * \param maximaLoc the vector of locations of the maxima,
 *       where each location is cv::Point(x=col, y=row) zero-based
 *
 */

void mcvGetMatMax(const cv::Mat *inMat, vector<double> &maxima,
                  vector<cv::Point> &maximaLoc, double threshold)
{

  double val;

#define MCV_MAT_MAX(type)  \
    /*loop on the matrix and get points that are larger than their*/ \
    /*neighboring 8 pixels*/ \
    for(int i=1; i<inMat->rows-1; i++) \
	for (int j=1; j<inMat->cols-1; j++) \
	{ \
	    /*get the current value*/ \
	    val = inMat->at<type>(i, j); \
	    /*check if it's larger than threshold*/ \
	    if (val >= threshold) \
	    { \
		/*found a maxima, put it in the return vector*/ \
		/*in decending order*/ \
		/*iterators for the two vectors*/ \
		vector<double>::iterator k; \
		vector<cv::Point>::iterator l; \
		/*loop till we find the place to put it in descendingly*/ \
		for(k=maxima.begin(), l=maximaLoc.begin(); \
		    k != maxima.end()  && val<= *k; k++,l++); \
		/*add its index*/ \
		maxima.insert(k, val); \
		maximaLoc.insert(l, cv::Point(j, i)); \
	    } \
	}

  //check type
  if (CV_MAT_TYPE(inMat->type())==FLOAT_MAT_TYPE)
  {
    MCV_MAT_MAX(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type())==INT_MAT_TYPE)
  {
    MCV_MAT_MAX(INT_MAT_ELEM_TYPE)
  }
  else
  {
    cerr << "Unsupported type in mcvGetMatMax\n";
    exit(1);
  }
}


/** This function gets the local maxima in a vector and their positions
 *
 * \param inVec input vector
 * \param localMaxima the output vector of local maxima
 * \param localMaximaLoc the vector of locations of the local maxima,
 *
 */
#define MCV_VECTOR_LOCAL_MAX(type)						\
    /*loop on the vector and get points that are larger than their*/		\
    /*neighboring points*/							\
    if(inVec->rows == 1)							\
    {										\
	for(int i=1; i<inVec->cols-1; i++)					\
	{									\
	    /*get the current value*/						\
	    val = inVec->at<type>(0, i);				\
	    /*check if it's larger than all its neighbors*/			\
	    if( val > inVec->at<type>(0, i-1) &&			\
		val > inVec->at<type>(0, i+1) )			\
	    {									\
		/*found a local maxima, put it in the return vector*/		\
		/*in decending order*/						\
		/*iterators for the two vectors*/				\
		vector<double>::iterator k;					\
		vector<int>::iterator l;					\
		/*loop till we find the place to put it in descendingly*/	\
		for(k=localMaxima.begin(), l=localMaximaLoc.begin();		\
		    k != localMaxima.end()  && val<= *k; k++,l++);		\
		/*add its index*/						\
		localMaxima.insert(k, val);					\
		localMaximaLoc.insert(l, i);					\
	    }									\
        }									\
    }										\
    else									\
    {										\
	for(int i=1; i<inVec->rows-1; i++)					\
	{									\
	    /*get the current value*/						\
	    val = inVec->at<type>(i, 0);				\
	    /*check if it's larger than all its neighbors*/			\
	    if( val > inVec->at<type>(i-1, 0) &&			\
		val > inVec->at<type>(i+1, 0) )			\
	    {									\
		/*found a local maxima, put it in the return vector*/		\
		/*in decending order*/						\
		/*iterators for the two vectors*/				\
		vector<double>::iterator k;					\
		vector<int>::iterator l;					\
		/*loop till we find the place to put it in descendingly*/	\
		for(k=localMaxima.begin(), l=localMaximaLoc.begin();		\
		    k != localMaxima.end()  && val<= *k; k++,l++);		\
		/*add its index*/		\
		localMaxima.insert(k, val);	\
		localMaximaLoc.insert(l, i);	\
	    }					\
        }					\
    }

void mcvGetVectorLocalMax(const cv::Mat *inVec, vector<double> &localMaxima,
                          vector<int> &localMaximaLoc)
{

  double val;

  //check type
  if (CV_MAT_TYPE(inVec->type())==FLOAT_MAT_TYPE)
  {
    MCV_VECTOR_LOCAL_MAX(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inVec->type())==INT_MAT_TYPE)
  {
    MCV_VECTOR_LOCAL_MAX(INT_MAT_ELEM_TYPE)
  }
  else
  {
    cerr << "Unsupported type in mcvGetVectorLocalMax\n";
    exit(1);
  }
}


/** This function gets the qtile-th quantile of the input matrix
 *
 * \param mat input matrix
 * \param qtile required input quantile probability
 * \return the returned value
 *
 */
LD_FLOAT mcvGetQuantile(const cv::Mat *mat, LD_FLOAT qtile)
{
  //make it a row vector
  cv::Mat rowMat = mat->reshape(0, 1);

  //get the quantile
  LD_FLOAT qval;
  qval = quantile((LD_FLOAT*) rowMat.data, rowMat.cols, qtile);

  return qval;
}


/** This function thresholds the image below a certain value to the threshold
 * so: outMat(i,j) = inMat(i,j) if inMat(i,j)>=threshold
 *                 = threshold otherwise
 *
 * \param inMat input matrix
 * \param outMat output matrix
 * \param threshold threshold value
 *
 */
void mcvThresholdLower(const cv::Mat *inMat, cv::Mat *outMat, LD_FLOAT threshold)
{

#define MCV_THRESHOLD_LOWER(type) \
     for (int i=0; i<inMat->rows; i++) \
        for (int j=0; j<inMat->cols; j++) \
            if ( inMat->at<type>(i, j)<threshold) \
                outMat->at<type>(i, j)=(type) 0; /*check it, was: threshold*/\

  //check if to copy into outMat or not
  if (inMat != outMat)
    inMat->copyTo(*outMat);

  //check type
  if (CV_MAT_TYPE(inMat->type())==FLOAT_MAT_TYPE)
  {
    MCV_THRESHOLD_LOWER(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type())==INT_MAT_TYPE)
  {
    MCV_THRESHOLD_LOWER(INT_MAT_ELEM_TYPE)
  }
  else
  {
    cerr << "Unsupported type in mcvGetVectorMax\n";
    exit(1);
  }
}

/** This function detects stop lines in the input image using IPM
 * transformation and the input camera parameters. The returned lines
 * are in a vector of Line objects, having start and end point in
 * input image frame.
 *
 * \param image the input image
 * \param stopLines a vector of returned stop lines in input image coordinates
 * \param linescores a vector of line scores returned
 * \param cameraInfo the camera parameters
 * \param stopLineConf parameters for stop line detection
 *
 *
 */
void mcvGetStopLines(const cv::Mat *inImage, vector<Line> *stopLines,
		     vector<LD_FLOAT> *lineScores, const CameraInfo *cameraInfo,
		      LaneDetectorConf *stopLineConf)

{
  //input size
  cv::Size inSize = cv::Size(inImage->cols, inImage->rows);

  //TODO: smooth image
  cv::Mat *image  = new cv::Mat();
  *image  = inImage->clone();
  //cv::smooth(image, image, CV_GAUSSIAN, 5, 5, 1, 1);

  IPMInfo ipmInfo;

//     //get the IPM size such that we have rows of the stop line
//     //is 3 pixels
//     double ipmWidth, ipmHeight;
//     mcvGetIPMExtent(cameraInfo, &ipmInfo);
//     ipmHeight = 3*(ipmInfo.yLimits[1]-ipmInfo.yLimits[0]) / (stopLineConf->lineHeight/3.);
//     ipmWidth = ipmHeight * 4/3;
//     //put into the conf
//     stopLineConf->ipmWidth = int(ipmWidth);
//     stopLineConf->ipmHeight = int(ipmHeight);

//         if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
//     cout << "IPM cols:" << stopLineConf->ipmWidth << " IPM rows:"
// 	 << stopLineConf->ipmHeight << "\n";
//     }//#endif


  //Get IPM
  cv::Size ipmSize = cv::Size((int)stopLineConf->ipmWidth,
                          (int)stopLineConf->ipmHeight);
  cv::Mat * ipm;
  ipm = new cv::Mat(ipmSize.height, ipmSize.width, inImage->type());
  //mcvGetIPM(inImage, ipm, &ipmInfo, cameraInfo);
  ipmInfo.vpPortion = stopLineConf->ipmVpPortion;
  ipmInfo.ipmLeft = stopLineConf->ipmLeft;
  ipmInfo.ipmRight = stopLineConf->ipmRight;
  ipmInfo.ipmTop = stopLineConf->ipmTop;
  ipmInfo.ipmBottom = stopLineConf->ipmBottom;
  ipmInfo.ipmInterpolation = stopLineConf->ipmInterpolation;
  list<cv::Point> outPixels;
  list<cv::Point>::iterator outPixelsi;
  mcvGetIPM(image, ipm, &ipmInfo, cameraInfo, &outPixels);

  //smooth the IPM
  //cv::smooth(ipm, ipm, CV_GAUSSIAN, 5, 5, 1, 1);

  //debugging
  cv::Mat *dbIpmImage;
  if(DEBUG_LINES) {//    #ifdef DEBUG_GET_STOP_LINES
      dbIpmImage = new cv::Mat(ipm->rows, ipm->cols, ipm->type());
      ipm->copyTo(*dbIpmImage);
  }//#endif


  //compute stop line cols: 2000 mm
  LD_FLOAT stopLinePixelWidth = stopLineConf->lineWidth * ipmInfo.xScale;
  //stop line pixel rows: 12 inches = 12*25.4 mm
  LD_FLOAT stopLinePixelHeight = stopLineConf->lineHeight *  ipmInfo.yScale;
  //kernel dimensions
  //unsigned char wx = 2;
  //unsigned char wy = 2;
  LD_FLOAT sigmax = stopLinePixelWidth;
  LD_FLOAT sigmay = stopLinePixelHeight;

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
  //cout << "Line cols:" << stopLinePixelWidth << "Line rows:"
  //	 << stopLinePixelHeight << "\n";
  }//#endif

  //filter the IPM image
  mcvFilterLines(ipm, ipm, stopLineConf->kernelWidth,
                 stopLineConf->kernelHeight, sigmax, sigmay,
                 LINE_HORIZONTAL);

    //zero out points outside the image in IPM view
  for(outPixelsi=outPixels.begin(); outPixelsi!=outPixels.end(); outPixelsi++)
    ipm->at<float>((*outPixelsi).y, (*outPixelsi).x) = 0;
  outPixels.clear();

  //zero out negative values
  mcvThresholdLower(ipm, ipm, 0);

  //compute quantile: .985
  LD_FLOAT qtileThreshold = mcvGetQuantile(ipm, stopLineConf->lowerQuantile);
  mcvThresholdLower(ipm, ipm, qtileThreshold);

  //debugging
  cv::Mat *dbIpmImageThresholded;
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    dbIpmImageThresholded = new cv::Mat(ipm->rows, ipm->cols, ipm->type());
    ipm->copyTo(*dbIpmImageThresholded);
  }//#endif



  //group stop lines
  switch(stopLineConf->groupingType)
  {
    //use HV grouping
    case GROUPING_TYPE_HV_LINES:
    //vector <Line> ipmStopLines;
    //vector <FLOAT> lineScores;
    mcvGetHVLines(ipm, stopLines, lineScores, LINE_HORIZONTAL,
                  stopLinePixelHeight, stopLineConf->binarize,
                  stopLineConf->localMaxima, stopLineConf->detectionThreshold,
                  stopLineConf->smoothScores);
    break;

  //use Hough Transform grouping
  case GROUPING_TYPE_HOUGH_LINES:
    //FLOAT rMin = 0.05*ipm->rows, rMax = 0.4*ipm->rows, rStep = 1;
    //FLOAT thetaMin = 88*CV_PI/180, thetaMax = 92*CV_PI/180, thetaStep = 1*CV_PI/180;
    //bool group = false; FLOAT groupThreshold = 1;
    mcvGetHoughTransformLines(ipm, stopLines, lineScores,
                              stopLineConf->rMin, stopLineConf->rMax,
                              stopLineConf->rStep, stopLineConf->thetaMin,
                              stopLineConf->thetaMax, stopLineConf->thetaStep,
                              stopLineConf->binarize, stopLineConf->localMaxima,
                              stopLineConf->detectionThreshold,
                              stopLineConf->smoothScores, stopLineConf->group,
                              stopLineConf->groupThreshold);
    break;
  }

  //get RANSAC lines
  if (stopLineConf->ransac)
  {
    mcvGetRansacLines(ipm, *stopLines, *lineScores, stopLineConf, LINE_HORIZONTAL);
  }

  if(stopLineConf->getEndPoints)
  {
    //get line extent in IPM image
    for(int i=0; i<(int)stopLines->size(); i++)
      mcvGetLineExtent(ipm, (*stopLines)[i], (*stopLines)[i]);
  }

  vector <Line> dbIpmStopLines;
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    dbIpmStopLines = *stopLines;
//         //print out lineScores
//         cout << "LineScores:";
//         //for (int i=0; i<(int)lineScores->size(); i++)
// 	for (int i=0; i<1 && lineScores->size()>0; i++)
//             cout << (*lineScores)[i] << " ";
//         cout << "\n";
  }//#endif

  //check if returned anything
  if (stopLines->size()!=0)
  {
    //convert the line into world frame
    for (unsigned int i=0; i<stopLines->size(); i++)
    {
        Line *line;
        line = &(*stopLines)[i];

        mcvPointImIPM2World(&(line->startPoint), &ipmInfo);
        mcvPointImIPM2World(&(line->endPoint), &ipmInfo);
    }
    //convert them from world frame into camera frame
    //
    //put a dummy line at the beginning till we check that cv::div bug
    Line dummy = {{1.,1.},{2.,2.}};
    stopLines->insert(stopLines->begin(), dummy);
    //convert to mat and get in image coordinates
    cv::Mat *mat = new cv::Mat(2, 2*stopLines->size(), FLOAT_MAT_TYPE);
    mcvLines2Mat(stopLines, mat);
    stopLines->clear();
    mcvTransformGround2Image(mat, mat, cameraInfo);
    //get back to vector
    mcvMat2Lines(mat, stopLines);
    //remove the dummy line at the beginning
    stopLines->erase(stopLines->begin());
    //clear
    delete mat;

    //clip the lines found and get their extent
    for (unsigned int i=0; i<stopLines->size(); i++)
    {
	    //clip
      mcvIntersectLineWithBB(&(*stopLines)[i], inSize, &(*stopLines)[i]);

	    //get the line extent
	    //mcvGetLineExtent(inImage, (*stopLines)[i], (*stopLines)[i]);
    }


  }

  //debugging
  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    //show the IPM image
    SHOW_IMAGE(dbIpmImage, "IPM image", 10);
    //thresholded ipm
    SHOW_IMAGE(dbIpmImageThresholded, "Stoplines thresholded IPM image", 10);
    //draw lines in IPM image
        //for (int i=0; i<(int)dbIpmStopLines.size(); i++)
    for (int i=0; i<1 && dbIpmStopLines.size()>0; i++)
    {
      mcvDrawLine(dbIpmImage, dbIpmStopLines[i], CV_RGB(0,0,0), 3);
    }
    SHOW_IMAGE(dbIpmImage, "Stoplines IPM with lines", 10);
    //draw lines on original image
    //cv::Mat *image = new cv::Mat(inImage->rows, inImage->cols, CV_32FC3);
    //cv::cvtColor(*inImage, *image, cv::COLOR_GRAY2RGB);
    //cv::Mat *image  = new cv::Mat();
*image  = inImage->clone();
    //for (int i=0; i<(int)stopLines->size(); i++)
    for (int i=0; i<1 && stopLines->size()>0; i++)
    {
      //SHOW_POINT((*stopLines)[i].startPoint, "start");
      //SHOW_POINT((*stopLines)[i].endPoint, "end");
      mcvDrawLine(image, (*stopLines)[i], CV_RGB(255,0,0), 3);
    }
    SHOW_IMAGE(image, "Detected Stoplines", 10);
    //delete image;
    delete dbIpmImage;
    delete dbIpmImageThresholded;
    dbIpmStopLines.clear();
  }//#endif //DEBUG_GET_STOP_LINES

  //clear
  delete ipm;
  delete image;
  //ipmStopLines.clear();
}

/** This function detects lanes in the input image using IPM
 * transformation and the input camera parameters. The returned lines
 * are in a vector of Line objects, having start and end point in
 * input image frame.
 *
 * \param image the input image
 * \param lanes a vector of returned stop lines in input image coordinates
 * \param linescores a vector of line scores returned
 * \param cameraInfo the camera parameters
 * \param stopLineConf parameters for stop line detection
 * \param state returns the current state and inputs the previous state to
 *   initialize the current detection (nullptr to ignore)
 *
 *
 */
void mcvGetLanes(const cv::Mat *inImage, const cv::Mat* clrImage,
                 vector<Line> *lanes, vector<LD_FLOAT> *lineScores,
                 vector<Spline> *splines, vector<float> *splineScores,
                 CameraInfo *cameraInfo, LaneDetectorConf *stopLineConf,
                 LineState* state)
{
  //input size
  cv::Size inSize = cv::Size(inImage->cols, inImage->rows);

  //TODO: smooth image
  cv::Mat *image  = new cv::Mat();
  *image  = inImage->clone();
  //cv::smooth(image, image, CV_GAUSSIAN, 5, 5, 1, 1);

  //SHOW_IMAGE(image, "Input image", 10);

  IPMInfo ipmInfo;

  //state: create a new structure, and put pointer to it if it's null
  LineState newState;
  if(!state) state = &newState;

//     //get the IPM size such that we have rows of the stop line
//     //is 3 pixels
//     double ipmWidth, ipmHeight;
//     mcvGetIPMExtent(cameraInfo, &ipmInfo);
//     ipmHeight = 3*(ipmInfo.yLimits[1]-ipmInfo.yLimits[0]) / (stopLineConf->lineHeight/3.);
//     ipmWidth = ipmHeight * 4/3;
//     //put into the conf
//     stopLineConf->ipmWidth = int(ipmWidth);
//     stopLineConf->ipmHeight = int(ipmHeight);

//     #ifdef DEBUG_GET_STOP_LINES
//     cout << "IPM cols:" << stopLineConf->ipmWidth << " IPM rows:"
// 	 << stopLineConf->ipmHeight << "\n";
//     #endif


  //Get IPM
  cv::Size ipmSize = cv::Size((int)stopLineConf->ipmWidth,
      (int)stopLineConf->ipmHeight);
  cv::Mat * ipm;
  ipm = new cv::Mat(ipmSize.height, ipmSize.width, inImage->type());
  //mcvGetIPM(inImage, ipm, &ipmInfo, cameraInfo);
  ipmInfo.vpPortion = stopLineConf->ipmVpPortion;
  ipmInfo.ipmLeft = stopLineConf->ipmLeft;
  ipmInfo.ipmRight = stopLineConf->ipmRight;
  ipmInfo.ipmTop = stopLineConf->ipmTop;
  ipmInfo.ipmBottom = stopLineConf->ipmBottom;
  ipmInfo.ipmInterpolation = stopLineConf->ipmInterpolation;
  list<cv::Point> outPixels;
  list<cv::Point>::iterator outPixelsi;
  mcvGetIPM(image, ipm, &ipmInfo, cameraInfo, &outPixels);

  //smooth the IPM image with 5x5 gaussian filter
//#warning "Check: Smoothing IPM image"
  //cv::smooth(ipm, ipm, CV_GAUSSIAN, 3);
  //      SHOW_MAT(ipm, "ipm");

  //     //subtract mean
  //     cv::Scalar mean = cv::mean(*ipm);
  //     *ipm = *ipm - mean;

  //keep copy
  cv::Mat* rawipm  = new cv::Mat();
* rawipm  = ipm->clone();

  //smooth the IPM
  //cv::smooth(ipm, ipm, CV_GAUSSIAN, 5, 5, 1, 1);

  //debugging
  cv::Mat *dbIpmImage;
  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    dbIpmImage = new cv::Mat(ipm->rows, ipm->cols, ipm->type());
    ipm->copyTo(*dbIpmImage);
    //show the IPM image
    SHOW_IMAGE(dbIpmImage, "IPM image", 10);
  }//#endif

  //compute stop line cols: 2000 mm
  LD_FLOAT stopLinePixelWidth = stopLineConf->lineWidth *
      ipmInfo.xScale;
  //stop line pixel rows: 12 inches = 12*25.4 mm
  LD_FLOAT stopLinePixelHeight = stopLineConf->lineHeight  *
      ipmInfo.yScale;
  //kernel dimensions
  //unsigned char wx = 2;
  //unsigned char wy = 2;
  LD_FLOAT sigmax = stopLinePixelWidth;
  LD_FLOAT sigmay = stopLinePixelHeight;

//     //filter in the horizontal direction
//     cv::Mat * ipmt = new cv::Mat(ipm->cols, ipm->rows, ipm->type());
//     *ipmt = ipm->t();
//     mcvFilterLines(ipmt, ipmt, stopLineConf->kernelWidth,
// 		   stopLineConf->kernelHeight, sigmax, sigmay,
// 		   LINE_VERTICAL);
//     //retranspose
//     cv::Mat *ipm2 = new cv::Mat(ipm->rows, ipm->cols, ipm->type());
//     *ipm2 = ipmt->t();
//     delete ipmt;

  //filter the IPM image
  mcvFilterLines(ipm, ipm, stopLineConf->kernelWidth,
                 stopLineConf->kernelHeight, sigmax, sigmay,
                 LINE_VERTICAL);
//     mcvFilterLines(ipm, ipm, stopLineConf->kernelWidth,
// 		   stopLineConf->kernelHeight, sigmax, sigmay,
// 		   LINE_VERTICAL);

  //zero out points outside the image in IPM view
  for(outPixelsi=outPixels.begin(); outPixelsi!=outPixels.end(); outPixelsi++)
  {
    ipm->at<float>((*outPixelsi).y, (*outPixelsi).x) = 0;
  // 	ipm2->at<float>((*outPixelsi).y, (*outPixelsi).x) = 0;
  }
  outPixels.clear();

//#warning "Check this clearing of IPM image for 2 lanes"
  if (stopLineConf->ipmWindowClear)
  {
    //check to blank out other periferi of the image
    //blank from 60->100 (cols 40)
    cv::Rect mask = cv::Rect(stopLineConf->ipmWindowLeft, 0,
                         stopLineConf->ipmWindowRight -
                         stopLineConf->ipmWindowLeft + 1,
                         ipm->rows);
    mcvSetMat(ipm, mask, 0);
  }

  //show filtered image
  if (DEBUG_LINES) {
    SHOW_IMAGE(ipm, "Lane unthresholded filtered", 10);
  }

  //take the negative to get double yellow lines
  //cv::scale(ipm, ipm, -1);

  cv::Mat *fipm  = new cv::Mat();
*fipm  = ipm->clone();

    //zero out negative values
//     SHOW_MAT(fipm, "fipm");
//#warning "clean negative parts in filtered image"
  mcvThresholdLower(ipm, ipm, 0);
//     mcvThresholdLower(ipm2, ipm2, 0);

//     //add the two images
//     cv::add(ipm, ipm2, ipm);

//     //clear the horizontal filtered image
//     delete ipm2;

  //fipm was here
  //make copy of filteed ipm image

  vector <Line> dbIpmStopLines;
  vector<Spline> dbIpmSplines;

  //int numStrips = 2;
  int stripHeight = ipm->rows / stopLineConf->numStrips;
  for (int i=0; i<stopLineConf->numStrips; i++) //lines
  {
    //get the mask
    cv::Rect mask;
    mask = cv::Rect(0, i*stripHeight, ipm->cols,
            stripHeight);
  // 	SHOW_RECT(mask, "Mask");

    //get the subimage to work on
    cv::Mat *subimage  = new cv::Mat();
*subimage  = ipm->clone();
    //clear all but the mask
    mcvSetMat(subimage, mask, 0);

    //compute quantile: .985
    LD_FLOAT qtileThreshold = mcvGetQuantile(subimage, stopLineConf->lowerQuantile);
    mcvThresholdLower(subimage, subimage, qtileThreshold);
  // 	FLOAT qtileThreshold = mcvGetQuantile(ipm, stopLineConf->lowerQuantile);
  // 	mcvThresholdLower(ipm, ipm, qtileThreshold);

  //     qtileThreshold = mcvGetQuantile(ipm2, stopLineConf->lowerQuantile);
  //     mcvThresholdLower(ipm2, ipm2, qtileThreshold);

      //and fipm was here last
  //     //make copy of filtered ipm image
  //     cv::Mat *fipm  = new cv::Mat();
*fipm  = ipm->clone();
    vector<Line> subimageLines;
    vector<Spline> subimageSplines;
    vector<float> subimageLineScores, subimageSplineScores;

	//check to blank out other periferi of the image
// 	mask = cv::Rect(40, 0, 80, subimage->rows);
// 	mcvSetMat(subimage, mask, 0);
    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
	    cv::Mat *dbIpmImageThresholded;
	    dbIpmImageThresholded = new cv::Mat(ipm->rows, ipm->cols, ipm->type());
	    subimage->copyTo(*dbIpmImageThresholded);    //ipm
	    char str[256];
	    sprintf(str, "Lanes #%d thresholded IPM", i);
	    //thresholded ipm
	    SHOW_IMAGE(dbIpmImageThresholded, str, 10);
	    delete dbIpmImageThresholded;
    }

    //get the lines/splines
    mcvGetLines(subimage, LINE_VERTICAL, subimageLines, subimageLineScores,
		    subimageSplines, subimageSplineScores, stopLineConf,
		    state);
// 	mcvGetLines(ipm, LINE_VERTICAL, *lanes, *lineScores,
// 		    *splines, *splineScores, stopLineConf,
// 		    state);
    //put back
    for (unsigned int k=0; k<subimageLines.size(); k++)
    {
      lanes->push_back(subimageLines[k]);
      lineScores->push_back(subimageLineScores[k]);
    }
    for (unsigned int k=0; k<subimageSplines.size(); k++)
    {
      splines->push_back(subimageSplines[k]);
      splineScores->push_back(subimageSplineScores[k]);
    }

	//debug
    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
	    for (unsigned int k=0; k<lanes->size(); k++)
        dbIpmStopLines.push_back((*lanes)[k]);
	    for (unsigned int k=0; k<splines->size(); k++)
        dbIpmSplines.push_back((*splines)[k]);

	    cv::Mat *dbIpmImageThresholded;
	    dbIpmImageThresholded = new cv::Mat(ipm->rows, ipm->cols, ipm->type());
	    subimage->copyTo(*dbIpmImageThresholded);    //ipm
	    char str[256];
	    sprintf(str, "Lanes #%d thresholded IPM", i);
	    //thresholded ipm
	    SHOW_IMAGE(dbIpmImageThresholded, str, 10);
	    delete dbIpmImageThresholded;

	    //dbIpmStopLines = *lanes;
	    //dbIpmSplines = *splines;
	    // 	//print out lineScores
	    // 	cout << "LineScores:";
	    // 	//for (int i=0; i<(int)lineScores->size(); i++)
	    // 	for (int i=0; i<(int)lineScores->size(); i++)
	    // 	    cout << (*lineScores)[i] << " ";
	    // 	cout << "\n";
    }//#endif

    //release
    delete subimage;
  }

  //postprocess lines/splines //rawipm
  mcvPostprocessLines(image, clrImage, fipm, ipm, *lanes, *lineScores,
                      *splines, *splineScores,
                      stopLineConf, state, ipmInfo, *cameraInfo); //rawipm

  if (DEBUG_LINES) {
    dbIpmSplines = state->ipmSplines;
  }

  //debugging
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    char str[256];
    //draw lines in IPM image
    //for (int i=0; i<(int)dbIpmStopLines.size(); i++)
    if (stopLineConf->ransacLine || !stopLineConf->ransacSpline)
        for (int i=0; i<(int)dbIpmStopLines.size(); i++)
          mcvDrawLine(dbIpmImage, dbIpmStopLines[i], CV_RGB(0,0,0), 1);
    if (stopLineConf->ransacSpline)
      for (int i=0; i<(int)dbIpmSplines.size(); i++)
        mcvDrawSpline(dbIpmImage, dbIpmSplines[i], CV_RGB(0,0,0), 1);

    SHOW_IMAGE(dbIpmImage, "Lanes IPM with lines", 10);
    //draw lines on original image
    cv::Mat *imageClr = new cv::Mat(inImage->rows, inImage->cols, CV_32FC3);
    cv::cvtColor(*image, *imageClr, cv::COLOR_GRAY2RGB);
    //cv::Mat *image  = new cv::Mat();
*image  = inImage->clone();
    //for (int i=0; i<(int)stopLines->size(); i++)
    if (stopLineConf->ransacLine || !stopLineConf->ransacSpline)
      for (int i=0; i<(int)lanes->size(); i++)
        mcvDrawLine(imageClr, (*lanes)[i], CV_RGB(255,0,0), 1);
    if (stopLineConf->ransacSpline)
      for (int i=0; i<(int)splines->size(); i++)
      {
        mcvDrawSpline(imageClr, (*splines)[i], CV_RGB(255,0,0), 1);
        sprintf(str, "%.2f", (*splineScores)[i]);
        mcvDrawText(imageClr, str,
              cv::Point((*splines)[i].points[(*splines)[i].degree]),
              .5, CV_RGB(0, 0, 255));
      }

    SHOW_IMAGE(imageClr, "Detected lanes", 0);
    //delete image;
    delete dbIpmImage;
    //delete dbIpmImageThresholded;
    delete imageClr;
    dbIpmStopLines.clear();
    dbIpmSplines.clear();
  }//#endif //DEBUG_GET_STOP_LINES

  //clear
  delete ipm;
  delete image;
  delete fipm;
  delete rawipm;
  //ipmStopLines.clear();
}


/** This function postprocesses the detected lines/splines to better localize
 * and extend them
 *
 * \param image the input image
 * \param clrImage the inpout color image
 * \param rawipm the raw ipm image
 * \param fipm the filtered ipm iamge
 * \param lines a vector of lines
 * \param lineScores the line scores
 * \param splines a vector of returned splines
 * \param splineScores the spline scores
 * \param lineConf the conf structure
 * \param state the state for RANSAC splines
 * \param ipmInfo the ipmInfo structure
 * \param cameraInfo the camera info structure
 *
 */
void mcvPostprocessLines(const cv::Mat* image, const cv::Mat* clrImage,
                         const cv::Mat* rawipm, const cv::Mat* fipm,
                         vector<Line> &lines, vector<float> &lineScores,
                         vector<Spline> &splines, vector<float> &splineScores,
                         LaneDetectorConf *lineConf, LineState *state,
                         IPMInfo &ipmInfo, CameraInfo &cameraInfo)
{
  cv::Size inSize = cv::Size(image->cols-1, image->rows-1);

  //vector of splines to keep
  vector<Spline> keepSplines;
  vector<float> keepSplineScores;

//     //get line extent
//     if(lineConf->getEndPoints)
//     {
// 	//get line extent in IPM image
// 	for(int i=0; i<(int)lines.size(); i++)
// 	    mcvGetLineExtent(rawipm, lines[i], lines[i]);
//     }

  //if return straight lines
  if (lineConf->ransacLine || !lineConf->ransacSpline)
  {
    mcvLinesImIPM2Im(lines, ipmInfo, cameraInfo, inSize);
  }
  //return spline
  if (lineConf->ransacSpline)
  {
    //localize splines
    for(int i=0; i<(int)splines.size(); i++)
    {
	    //get spline status
	    int splineStatus = lineConf->checkIPMSplines ?
        mcvCheckSpline(splines[i],
                       lineConf->checkIPMSplinesCurvenessThreshold,
                       lineConf->checkIPMSplinesLengthThreshold,
                       lineConf->checkIPMSplinesThetaDiffThreshold,
                       lineConf->checkIPMSplinesThetaThreshold)
          : 0;

      //check it
      if (!(((splineStatus & CurvedSpline) && (splineStatus & CurvedSplineTheta))
          || splineStatus & HorizontalSpline))
      {
        //better localize points
        cv::Mat *points = mcvEvalBezierSpline(splines[i], .1); //.05
        //mcvLocalizePoints(ipm, points, points); //inImage
        //extend spline
        cv::Mat* p = mcvExtendPoints(rawipm, points,
                lineConf->extendIPMAngleThreshold,
                lineConf->extendIPMMeanDirAngleThreshold,
                lineConf->extendIPMLinePixelsTangent,
                lineConf->extendIPMLinePixelsNormal,
                lineConf->extendIPMContThreshold,
                lineConf->extendIPMDeviationThreshold,
                cv::Rect(0, lineConf->extendIPMRectTop,
                  rawipm->cols-1,
                  lineConf->extendIPMRectBottom-lineConf->extendIPMRectTop),
                false);
        //refit spline
        Spline spline = mcvFitBezierSpline(p, lineConf->ransacSplineDegree);

		//save
//#warning "Check this later: extension in IPM. Check threshold value"
// 		splines[i] = spline;

		//calculate the score from fipm or ipm (thresholded)
		//float lengthRatio = 0.5; //.8
		//float angleRatio = 0.8; //.4
		//vector<int> jitter = mcvGetJitterVector(lineConf->splineScoreJitter);//2);

        float score = mcvGetSplineScore(fipm, splines[i],
                                        lineConf->splineScoreStep, //.1
                                        lineConf->splineScoreJitter, //jitter,
                                        lineConf->splineScoreLengthRatio,
                                        lineConf->splineScoreAngleRatio);
        //jitter.clear();
        splineScores[i] = score;

        //check score
        if (splineScores[i] >= lineConf->finalSplineScoreThreshold)
        {
          keepSplines.push_back(spline);
          keepSplineScores.push_back(splineScores[i]);
        }
        //clear
        delete points;
        delete p;
      } //if
    } //for

    //put back
    splines.clear();
    splineScores.clear();
    splines = keepSplines;
    splineScores = keepSplineScores;
    keepSplines.clear();
    keepSplineScores.clear();

// 	if (DEBUG_LINES) {
// 	    dbIpmSplines = *splines;
// 	}

    //save state
    //save IPM splines to use in next frame
    state->ipmSplines.clear();
    state->ipmSplines = splines;

    //convert to image coordinates
    mcvSplinesImIPM2Im(splines, ipmInfo, cameraInfo, inSize);


    //	fprintf(stderr, "start of splines------------------------\n");
    //	for (unsigned int i=0; i<splines.size(); i++)
    //	{
    //	    for (int j=0; j<=splines[i].degree; j++)
    //		fprintf(stderr, "%f %f ", splines[i].points[j].x,
    //			splines[i].points[j].y);
    //	    fprintf(stderr, "\n");
    //	}
    //	fprintf(stderr, "\nend of splines------------------------\n");


    //convert and extend in image coordinates
    //localize splines
    for(int i=0; i<(int)splines.size(); i++)
    {
	    //check the spline
	    int splineStatus = lineConf->checkSplines ?
      mcvCheckSpline(splines[i],
                     lineConf->checkSplinesCurvenessThreshold,
                     lineConf->checkSplinesLengthThreshold,
                     lineConf->checkSplinesThetaDiffThreshold,
                     lineConf->checkSplinesThetaThreshold)
        : 0;
	    //check if short, then put the corresponding line instead
	    if (splineStatus & (ShortSpline|CurvedSpline))
	    {
        for (unsigned int j=0; j<lines.size(); j++)
        {
          //convert to spline
          Spline sp = mcvLineXY2Spline(lines[j], lineConf->ransacSplineDegree);
          //check if that merges with the current one
          if (mcvCheckMergeSplines(splines[i], sp, .4, 50, .4, 50, 50))
          {
            //put the spline
            splines[i] = sp;
            splineStatus = 0;
            break;
          }
        }
        //splines[i] = mcvLineXY2Spline(lines[i], lineConf->ransacSplineDegree);
	    }
	    if (!(splineStatus & (CurvedSpline|CurvedSplineTheta)))
	    {
        //better localize points
        cv::Mat *points = mcvEvalBezierSpline(splines[i], .05);
        // 	    cv::Mat *points = mcvGetBezierSplinePixels((*splines)[i], .05,
        // 						     cv::Size(inImage->cols-1,
        // 							    inImage->rows-1),
        // 						     true);
        // 	    cv::Mat *p = new cv::Mat(points->rows, points->cols, CV_32FC1);
        // 	    cv::convert(points, p);
        mcvLocalizePoints(image, points, points, lineConf->localizeNumLinePixels,
              lineConf->localizeAngleThreshold); //inImage

        //get color
        cv::Mat* clrPoints = points;

        //extend spline
        cv::Mat* p = mcvExtendPoints(image, points,
                                   lineConf->extendAngleThreshold,
                                   lineConf->extendMeanDirAngleThreshold,
                                   lineConf->extendLinePixelsTangent,
                                   lineConf->extendLinePixelsNormal,
                                   lineConf->extendContThreshold,
                                   lineConf->extendDeviationThreshold,
                    cv::Rect(0, lineConf->extendRectTop,
                           image->cols,
                           lineConf->extendRectBottom-lineConf->extendRectTop));

        //refit
        Spline spline = mcvFitBezierSpline(p, lineConf->ransacSplineDegree);
        //splines[i] = spline;
        clrPoints = p;

        //check the extended one
        if (lineConf->checkSplines &&
            (mcvCheckSpline(spline,
                lineConf->checkSplinesCurvenessThreshold,
                lineConf->checkSplinesLengthThreshold,
                lineConf->checkSplinesThetaDiffThreshold,
                lineConf->checkSplinesThetaThreshold)
            & CurvedSpline))
        {
          //rfit using points before extending
          spline = mcvFitBezierSpline(points, lineConf->ransacSplineDegree);
          clrPoints = points;

          //check again
          if (mcvCheckSpline(spline,
                             lineConf->checkSplinesCurvenessThreshold,
                             lineConf->checkSplinesLengthThreshold,
                             lineConf->checkSplinesThetaDiffThreshold,
                             lineConf->checkSplinesThetaThreshold)
                & CurvedSpline)
            //use spline before localization
            spline = splines[i];
        }

        //get color
//  		fprintf(stderr, "Color for spline %d\n", i);
        LineColor clr = lineConf->checkColor ?
          mcvGetPointsColor(clrImage, clrPoints,
                            lineConf->checkColorWindow,
                            lineConf->checkColorNumYellowMin,
                            lineConf->checkColorRGMin,
                            lineConf->checkColorRGMax,
                            lineConf->checkColorGBMin,
                            lineConf->checkColorRBMin,
                            lineConf->checkColorRBF,
                            lineConf->checkColorRBFThreshold)
                            : LINE_COLOR_WHITE;

        //clear
        delete points;
        delete p;

        //put it
        spline.color = clr;
        keepSplines.push_back(spline);
        keepSplineScores.push_back(splineScores[i]);
      } //if
    } //for

    //put them back
    splines.clear();
    splineScores.clear();
    splines = keepSplines;
    splineScores = keepSplineScores;
    keepSplines.clear();
    keepSplineScores.clear();

//	fprintf(stderr, "start of splines------------------------\n");
//	for (unsigned int i=0; i<splines.size(); i++)
//	{
//	    for (int j=0; j<=splines[i].degree; j++)
//		fprintf(stderr, "%f %f ", splines[i].points[j].x,
//			splines[i].points[j].y);
//	    fprintf(stderr, "\n");
//	}
//	fprintf(stderr, "\nend of splines------------------------\n");
  } //if
  //check on splines detected, and if too curvy, replace
    //with corresponding line
}



/** This function extracts lines from the passed infiltered and thresholded
 * image
 *
 * \param image the input thresholded filtered image
 * \param lineType the line type to look for (LINE_VERTICAL or LINE_HORIZONTAL)
 * \param lines a vector of lines
 * \param lineScores the line scores
 * \param splines a vector of returned splines
 * \param splineScores the spline scores
 * \param lineConf the conf structure
 * \param state the state for RANSAC splines
 *
 */
void mcvGetLines(const cv::Mat* image, LineType lineType,
                 vector<Line> &lines, vector<float> &lineScores,
                 vector<Spline> &splines, vector<float> &splineScores,
                 LaneDetectorConf *lineConf, LineState *state)
{

  //initial grouping of lines
  switch(lineConf->groupingType)
  {
    //use HV grouping
    case GROUPING_TYPE_HV_LINES:
      //vector <Line> ipmStopLines;
      //vector <FLOAT> lineScores;
      mcvGetHVLines(image, &lines, &lineScores, lineType,
                    6, //stopLinePixelHeight,
                    lineConf->binarize, lineConf->localMaxima,
                    lineConf->detectionThreshold,
                    lineConf->smoothScores);
      break;

  //use Hough Transform grouping
  case GROUPING_TYPE_HOUGH_LINES:
	//FLOAT rMin = 0.05*ipm->rows, rMax = 0.4*ipm->rows, rStep = 1;
	//FLOAT thetaMin = 88*CV_PI/180, thetaMax = 92*CV_PI/180, thetaStep = 1*CV_PI/180;
	//	bool group = true; FLOAT groupThreshold = 15;
    mcvGetHoughTransformLines(image, &lines, &lineScores,
                              lineConf->rMin, lineConf->rMax,
                              lineConf->rStep, lineConf->thetaMin,
                              lineConf->thetaMax, lineConf->thetaStep,
                              lineConf->binarize, lineConf->localMaxima,
                              lineConf->detectionThreshold,
                              lineConf->smoothScores,
                              lineConf->group, lineConf->groupThreshold);

    break;
  }

    //get only two lines if in this mode
//     if (lineConf->group)
// 	mcvGroupLines(lines, lineScores,
// 		      lineConf->groupThreshold,
// 		      cv::Size(image->cols, image->rows));
  if (lineConf->checkLaneWidth)
    mcvCheckLaneWidth(lines, lineScores,
                      lineConf->checkLaneWidthMean,
                      lineConf->checkLaneWidthStd); //70&20 65&10 25&10

    //check if to do RANSAC
    if (lineConf->ransac)
    {
      //do RANSAC lines?
      if (lineConf->ransacLine)
        mcvGetRansacLines(image, lines, lineScores, lineConf, lineType);

      //do RANSAC splines?
      if (lineConf->ransacSpline)
        mcvGetRansacSplines(image, lines, lineScores,
                            lineConf, lineType, splines, splineScores, state);
    }

    //get bounding boxes around returned splines to pass to the next
    //frame
    state->ipmBoxes.clear();
    mcvGetSplinesBoundingBoxes(splines, lineType,
                               cv::Size(image->cols, image->rows),
                               state->ipmBoxes);
}

/** This function makes some checks on splines and decides
 * whether to keep them or not
 *
 * \param spline the input spline
 * \param curvenessThreshold minimum curveness score it should have
 * \param lengthThreshold mimimum threshold it should have
 * \param thetaDiffThreshold max theta diff it should have
 * \param thetaThreshold max theta it should have to be considered horizontal
 *
 * \return code that determines what to do with the spline
 *
 */
int mcvCheckSpline(const Spline &spline, float curvenessThreshold,
                   float lengthThreshold, float thetaDiffThreshold,
                   float thetaThreshold)
{

  //get the spline features
  float theta, r, meanTheta, meanR, length, curveness;
  mcvGetSplineFeatures(spline, 0, &theta, &r, &length,
      &meanTheta, &meanR, &curveness);
  float thetaDiff = fabs(meanTheta - theta);
//     thetaDiff = thetaDiff>CV_PI ? thetaDiff-CV_PI : thetaDiff;

//     float curvenessThreshold = .92; //.85;
//     float lengthThreshold = 30;
//     float thetaDiffThreshold = .1;
//     float thetaThreshold = 70. * CV_PI/180;


  char check = 0;
  if (curveness<curvenessThreshold)
    check |= CurvedSpline;
  if (thetaDiff > thetaDiffThreshold)
    check |= CurvedSplineTheta;
  if (length<lengthThreshold)
    check |= ShortSpline;
  if (meanTheta<thetaThreshold)//(fabs(meanTheta)>thetaThreshold)
    check |= HorizontalSpline;


  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

    fprintf(stderr, "thetaDiffThreshold=%f\n", thetaDiffThreshold);
    fprintf(stderr, "%s: curveness=%f, length=%f, thetaDiff=%f, meanTheta=%f, theta=%f\n",
      check & (ShortSpline|CurvedSpline|HorizontalSpline)
      ? "YES" : "NO ", curveness, length, thetaDiff, meanTheta, theta);
    fprintf(stderr, "\t%s\t%s\t%s\t%s\n",
      check&CurvedSpline? "curved" : "not curved",
      check&CurvedSplineTheta? "curved theta" : "not curved theta",
      check&ShortSpline? "short" : "not short",
      check&HorizontalSpline? "horiz" : "not horiz");

    cv::Mat* im = new cv::Mat(480, 640, CV_8UC3);
    im->setTo(0.);
    //draw splines
    mcvDrawSpline(im, spline, CV_RGB(255, 0, 0), 1);
    SHOW_IMAGE(im, "Check Splines", 10);
    //clear
    delete im;
  }//#endif

  return check;
}

/** This function makes some checks on points and decides
 * whether to keep them or not
 *
 * \param points the array of points to check
 *
 * \return code that determines what to do with the points
 *
 */
int mcvCheckPoints(const cv::Mat* points)
{

  //get the spline features
  float theta, r, meanTheta, meanR, curveness; //length
  mcvGetPointsFeatures(points, 0, &theta, &r, 0,
      &meanTheta, &meanR, &curveness);
  float thetaDiff = fabs(meanTheta - theta);

  float curvenessThreshold = .8;
  float thetaDiffThreshold = .3;

  int check = 0;
  if (curveness<curvenessThreshold)
    check |= CurvedSpline;
  if (thetaDiff > thetaDiffThreshold)
    check |= CurvedSplineTheta;

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

    fprintf(stderr, "%s: curveness=%f, thetaDiff=%f, meanTheta=%f\n",
      check & (ShortSpline|CurvedSpline|HorizontalSpline)
      ? "YES" : "NO ", curveness, thetaDiff, meanTheta);

    cv::Mat* im = new cv::Mat(480, 640, CV_8UC3);
    im->setTo(0.);
    //draw splines
    for (int i=0; i<points->rows-1; i++)
    {
	    Line line;
	    line.startPoint = cv::Point(points->at<float>(i, 0),
                                     points->at<float>(i, 1));
	    line.endPoint = cv::Point(points->at<float>(i+1, 0),
                                   points->at<float>(i+1, 1));
	    mcvDrawLine(im, line, CV_RGB(255, 0, 0), 1);
    }
    SHOW_IMAGE(im, "Check Points", 0);
    //clear
    delete im;
  }//#endif

  return check;
}


/** This function converts an array of lines to a matrix (already allocated)
 *
 * \param lines input vector of lines
 * \param size number of lines to convert
 * \return the converted matrix, it has 2x2*size where size is the
 *  number of lines, first row is x values (start.x, end.x) and second
 *  row is y-values
 *
 *
 */
void mcvLines2Mat(const vector<Line> *lines, cv::Mat *mat)
{
  //allocate the matrix
  //*mat = new cv::Mat(2, size*2, FLOAT_MAT_TYPE);

  //loop and put values
  int j;
  for (int i=0; i<(int)lines->size(); i++)
  {
    j = 2*i;
    mat->at<FLOAT_MAT_ELEM_TYPE>(0, j) = (*lines)[i].startPoint.x;
    mat->at<FLOAT_MAT_ELEM_TYPE>(1, j) = (*lines)[i].startPoint.y;
    mat->at<FLOAT_MAT_ELEM_TYPE>(0, j+1) = (*lines)[i].endPoint.x;
    mat->at<FLOAT_MAT_ELEM_TYPE>(1, j+1) = (*lines)[i].endPoint.y;
  }
}


/** This function converts matrix into n array of lines
 *
 * \param mat input matrix , it has 2x2*size where size is the
 *  number of lines, first row is x values (start.x, end.x) and second
 *  row is y-values
 * \param  lines the rerurned vector of lines
 *
 *
 */
void mcvMat2Lines(const cv::Mat *mat, vector<Line> *lines)
{

  Line line;
  //loop and put values
  for (int i=0; i<int(mat->cols/2); i++)
  {
    int j = 2*i;
    //get the line
    line.startPoint.x = mat->at<FLOAT_MAT_ELEM_TYPE>(0, j);
    line.startPoint.y =  mat->at<FLOAT_MAT_ELEM_TYPE>(1, j);
    line.endPoint.x = mat->at<FLOAT_MAT_ELEM_TYPE>(0, j+1);
    line.endPoint.y = mat->at<FLOAT_MAT_ELEM_TYPE>(1, j+1);
    //push it
    lines->push_back(line);
  }
}



/** This function intersects the input line with the given bounding box
 *
 * \param inLine the input line
 * \param bbox the bounding box
 * \param outLine the output line
 *
 */
void mcvIntersectLineWithBB(const Line *inLine, const cv::Size bbox,
                            Line *outLine)
{
  //put output
  outLine->startPoint.x = inLine->startPoint.x;
  outLine->startPoint.y = inLine->startPoint.y;
  outLine->endPoint.x = inLine->endPoint.x;
  outLine->endPoint.y = inLine->endPoint.y;

  //check which points are inside
  bool startInside, endInside;
  startInside = mcvIsPointInside(inLine->startPoint, bbox);
  endInside = mcvIsPointInside(inLine->endPoint, bbox);

  //now check
  if (!(startInside && endInside))
  {
    //difference
    LD_FLOAT deltax, deltay;
    deltax = inLine->endPoint.x - inLine->startPoint.x;
    deltay = inLine->endPoint.y - inLine->startPoint.y;
    //hold parameters
    LD_FLOAT t[4]={2,2,2,2};
    LD_FLOAT xup, xdown, yleft, yright;

    //intersect with top and bottom borders: y=0 and y=bbox.rows-1
    if (deltay==0) //horizontal line
    {
      xup = xdown = bbox.width+2;
    }
    else
    {
      t[0] = -inLine->startPoint.y/deltay;
      xup = inLine->startPoint.x + t[0]*deltax;
      t[1] = (bbox.height-inLine->startPoint.y)/deltay;
      xdown = inLine->startPoint.x + t[1]*deltax;
    }

    //intersect with left and right borders: x=0 and x=bbox.widht-1
    if (deltax==0) //horizontal line
    {
      yleft = yright = bbox.height+2;
    }
    else
    {
      t[2] = -inLine->startPoint.x/deltax;
      yleft = inLine->startPoint.y + t[2]*deltay;
      t[3] = (bbox.width-inLine->startPoint.x)/deltax;
      yright = inLine->startPoint.y + t[3]*deltay;
    }

    //points of intersection
    FLOAT_POINT2D pts[4] = {{xup, 0},{xdown,static_cast<float>(bbox.height)},
	{ 0, yleft }, { static_cast<float>(bbox.width), yright } };

    //now decide which stays and which goes
    int i;
    if (!startInside)
    {
      bool cont=true;
      for (i=0; i<4 && cont; i++)
      {
        if (t[i]>=0 && t[i]<=1 && mcvIsPointInside(pts[i],bbox) &&
          !(pts[i].x == outLine->endPoint.x &&
          pts[i].y == outLine->endPoint.y) )
        {
          outLine->startPoint.x = pts[i].x;
          outLine->startPoint.y = pts[i].y;
          t[i] = 2;
          cont = false;
        }
      }
	    //check if not replaced
	    if(cont)
	    {
        //loop again removing restriction on endpoint this time
        for (i=0; i<4 && cont; i++)
        {
          if (t[i]>=0 && t[i]<=1 && mcvIsPointInside(pts[i],bbox))
          {
            outLine->startPoint.x = pts[i].x;
            outLine->startPoint.y = pts[i].y;
            t[i] = 2;
            cont = false;
          }
        }
      }
    }
    if (!endInside)
    {
      bool cont=true;
      for (i=0; i<4 && cont; i++)
      {
        if (t[i]>=0 && t[i]<=1 && mcvIsPointInside(pts[i],bbox) &&
          !(pts[i].x == outLine->startPoint.x &&
          pts[i].y == outLine->startPoint.y) )
        {
          outLine->endPoint.x = pts[i].x;
          outLine->endPoint.y = pts[i].y;
          t[i] = 2;
          cont = false;
        }
      }
      //check if not replaced
      if(cont)
      {
        //loop again removing restriction on endpoint this time
        for (i=0; i<4 && cont; i++)
        {
          if (t[i]>=0 && t[i]<=1 && mcvIsPointInside(pts[i],bbox))
          {
            outLine->endPoint.x = pts[i].x;
            outLine->endPoint.y = pts[i].y;
            t[i] = 2;
            cont = false;
          }
        }
      }
    }
  }
}


/** This function intersects the input line (given in r and theta) with
 *  the given bounding box where the line is represented by:
 *  x cos(theta) + y sin(theta) = r
 *
 * \param r the r value for the input line
 * \param theta the theta value for the input line
 * \param bbox the bounding box
 * \param outLine the output line
 *
 */
void mcvIntersectLineRThetaWithBB(LD_FLOAT r, LD_FLOAT theta, const cv::Size bbox,
                                  Line *outLine)
{
  //hold parameters
  double xup, xdown, yleft, yright;

  //intersect with top and bottom borders: y=0 and y=bbox.rows-1
  if (cos(theta)==0) //horizontal line
  {
    xup = xdown = bbox.width+2;
  }
  else
  {
    xup = r / cos(theta);
    xdown = (r-bbox.height*sin(theta))/cos(theta);
  }

  //intersect with left and right borders: x=0 and x=bbox.widht-1
  if (sin(theta)==0) //horizontal line
  {
    yleft = yright = bbox.height+2;
  }
  else
  {
    yleft = r/sin(theta);
    yright = (r-bbox.width*cos(theta))/sin(theta);
  }

  //points of intersection
  FLOAT_POINT2D pts[4] = { { static_cast<float>(xup), 0 }, { static_cast<float>(xdown), static_cast<float>(bbox.height) },
  { 0, static_cast<float>(yleft) }, { static_cast<float>(bbox.width), static_cast<float>(yright) } };

  //get the starting point
  int i;
  for (i=0; i<4; i++)
  {
    //if point inside, then put it
    if(mcvIsPointInside(pts[i], bbox))
    {
	    outLine->startPoint.x = pts[i].x;
	    outLine->startPoint.y = pts[i].y;
	    //get out of for loop
	    break;
    }
  }
  //get the ending point
  for (i++; i<4; i++)
  {
    //if point inside, then put it
    if(mcvIsPointInside(pts[i], bbox))
    {
	    outLine->endPoint.x = pts[i].x;
	    outLine->endPoint.y = pts[i].y;
	    //get out of for loop
	    break;
    }
  }
}


/** This function checks if the given point is inside the bounding box
 * specified
 *
 * \param inLine the input line
 * \param bbox the bounding box
 * \param outLine the output line
 *
 */
bool mcvIsPointInside(FLOAT_POINT2D point, cv::Size bbox)
{
  return (point.x>=0 && point.x<=bbox.width
      && point.y>=0 && point.y<=bbox.height) ? true : false;
}


/** This function intersects the input line (given in r and theta) with
 *  the given rectangle where the line is represented by:
 *  x cos(theta) + y sin(theta) = r
 *
 * \param r the r value for the input line
 * \param theta the theta value for the input line
 * \param rect the input rectangle (given two opposite points in the rectangle,
 *   upperleft->startPoint and bottomright->endPoint where x->right and y->down)
 * \param outLine the output line
 *
 */
void mcvIntersectLineRThetaWithRect(LD_FLOAT r, LD_FLOAT theta, const Line &rect,
                                    Line &outLine)
{
  //hold parameters
  double xup, xdown, yleft, yright;

  //intersect with top and bottom borders: y=rect->startPoint.y and y=rect->endPoint.y
  if (cos(theta)==0) //horizontal line
  {
    xup = xdown = rect.endPoint.x+2;
  }
  else
  {
    xup = (r-rect.startPoint.y*sin(theta)) / cos(theta);
    xdown = (r-rect.endPoint.y*sin(theta)) / cos(theta);
  }

  //intersect with left and right borders: x=rect->startPoint.x and x=rect->endPoint.x
  if (sin(theta)==0) //horizontal line
  {
    yleft = yright = rect.endPoint.y+2;
  }
  else
  {
    yleft = (r-rect.startPoint.x*cos(theta)) / sin(theta);
    yright = (r-rect.endPoint.x*cos(theta)) / sin(theta);
  }

  //points of intersection
  FLOAT_POINT2D pts[4] = { { static_cast<float>(xup), rect.startPoint.y }, { static_cast<float>(xdown), rect.endPoint.y },
  { rect.startPoint.x, static_cast<float>(yleft) }, { rect.endPoint.x, static_cast<float>(yright) } };

  //get the starting point
  int i;
  for (i=0; i<4; i++)
  {
    //if point inside, then put it
    if(mcvIsPointInside(pts[i], rect))
    {
	    outLine.startPoint.x = pts[i].x;
	    outLine.startPoint.y = pts[i].y;
	    //get out of for loop
	    break;
    }
  }
  //get the ending point
  for (i++; i<4; i++)
  {
    //if point inside, then put it
    if(mcvIsPointInside(pts[i], rect))
    {
	    outLine.endPoint.x = pts[i].x;
	    outLine.endPoint.y = pts[i].y;
	    //get out of for loop
	    break;
    }
  }
}


/** This function checks if the given point is inside the rectangle specified
 *
 * \param inLine the input line
 * \param rect the specified rectangle
 *
 */
bool mcvIsPointInside(FLOAT_POINT2D &point, const Line &rect)
{
  return (point.x>=rect.startPoint.x && point.x<=rect.endPoint.x
      && point.y>=rect.startPoint.y && point.y<=rect.endPoint.y) ? true : false;
}

/** This function checks if the given point is inside the rectangle specified
 *
 * \param inLine the input line
 * \param rect the specified rectangle
 *
 */
bool mcvIsPointInside(FLOAT_POINT2D &point, const cv::Rect &rect)
{
  return (point.x>=rect.x && point.x<=(rect.x+rect.width)
      && point.y>=rect.y && point.y<=(rect.y+rect.height)) ? true : false;
}

/** This function converts an INT mat into a FLOAT mat (already allocated)
 *
 * \param inMat input INT matrix
 * \param outMat output FLOAT matrix
 *
 */
void mcvMatInt2Float(const cv::Mat *inMat, cv::Mat *outMat)
{
  for (int i=0; i<inMat->rows; i++)
    for (int j=0; j<inMat->cols; j++)
      outMat->at<FLOAT_MAT_ELEM_TYPE>(i, j) =
            (FLOAT_MAT_ELEM_TYPE) inMat->at<INT_MAT_ELEM_TYPE>(i, j)/255;
}


/** This function draws a line onto the passed image
 *
 * \param image the input iamge
 * \param line input line
 * \param line color
 * \param cols line cols
 *
 */
void mcvDrawLine(cv::Mat *image, Line line, cv::Scalar color, int cols)
{
  cv::line(*image, cv::Point((int)line.startPoint.x,(int)line.startPoint.y),
          cv::Point((int)line.endPoint.x,(int)line.endPoint.y),
          color, cols);
}

/** This initializes the LaneDetectorinfo structure
 *
 * \param fileName the input file name
 * \param stopLineConf the structure to fill
 *
 *
 */
 void mcvInitLaneDetectorConf(char * const fileName,
    LaneDetectorConf *stopLineConf)
{
  //parsed camera data
  LaneDetectorParserInfo stopLineParserInfo;
  //read the data
  assert(LaneDetectorParser_configfile(fileName, &stopLineParserInfo, 0, 1, 1)==0);
  //init the strucure
  stopLineConf->ipmWidth = stopLineParserInfo.ipmWidth_arg;
  stopLineConf->ipmHeight = stopLineParserInfo.ipmHeight_arg;
  stopLineConf->ipmLeft = stopLineParserInfo.ipmLeft_arg;
  stopLineConf->ipmRight = stopLineParserInfo.ipmRight_arg;
  stopLineConf->ipmBottom = stopLineParserInfo.ipmBottom_arg;
  stopLineConf->ipmTop = stopLineParserInfo.ipmTop_arg;
  stopLineConf->ipmInterpolation = stopLineParserInfo.ipmInterpolation_arg;

  stopLineConf->lineWidth = stopLineParserInfo.lineWidth_arg;
  stopLineConf->lineHeight = stopLineParserInfo.lineHeight_arg;
  stopLineConf->kernelWidth = stopLineParserInfo.kernelWidth_arg;
  stopLineConf->kernelHeight = stopLineParserInfo.kernelHeight_arg;
  stopLineConf->lowerQuantile =
      stopLineParserInfo.lowerQuantile_arg;
  stopLineConf->localMaxima =
      stopLineParserInfo.localMaxima_arg;
  stopLineConf->groupingType = stopLineParserInfo.groupingType_arg;
  stopLineConf->binarize = stopLineParserInfo.binarize_arg;
  stopLineConf->detectionThreshold =
      stopLineParserInfo.detectionThreshold_arg;
  stopLineConf->smoothScores =
      stopLineParserInfo.smoothScores_arg;
  stopLineConf->rMin = stopLineParserInfo.rMin_arg;
  stopLineConf->rMax = stopLineParserInfo.rMax_arg;
  stopLineConf->rStep = stopLineParserInfo.rStep_arg;
  stopLineConf->thetaMin = stopLineParserInfo.thetaMin_arg * CV_PI/180;
  stopLineConf->thetaMax = stopLineParserInfo.thetaMax_arg * CV_PI/180;
  stopLineConf->thetaStep = stopLineParserInfo.thetaStep_arg * CV_PI/180;
  stopLineConf->ipmVpPortion = stopLineParserInfo.ipmVpPortion_arg;
  stopLineConf->getEndPoints = stopLineParserInfo.getEndPoints_arg;
  stopLineConf->group = stopLineParserInfo.group_arg;
  stopLineConf->groupThreshold = stopLineParserInfo.groupThreshold_arg;
  stopLineConf->ransac = stopLineParserInfo.ransac_arg;

  stopLineConf->ransacLineNumSamples = stopLineParserInfo.ransacLineNumSamples_arg;
  stopLineConf->ransacLineNumIterations = stopLineParserInfo.ransacLineNumIterations_arg;
  stopLineConf->ransacLineNumGoodFit = stopLineParserInfo.ransacLineNumGoodFit_arg;
  stopLineConf->ransacLineThreshold = stopLineParserInfo.ransacLineThreshold_arg;
  stopLineConf->ransacLineScoreThreshold = stopLineParserInfo.ransacLineScoreThreshold_arg;
  stopLineConf->ransacLineBinarize = stopLineParserInfo.ransacLineBinarize_arg;
  stopLineConf->ransacLineWindow = stopLineParserInfo.ransacLineWindow_arg;

  stopLineConf->ransacSplineNumSamples = stopLineParserInfo.ransacSplineNumSamples_arg;
  stopLineConf->ransacSplineNumIterations = stopLineParserInfo.ransacSplineNumIterations_arg;
  stopLineConf->ransacSplineNumGoodFit = stopLineParserInfo.ransacSplineNumGoodFit_arg;
  stopLineConf->ransacSplineThreshold = stopLineParserInfo.ransacSplineThreshold_arg;
  stopLineConf->ransacSplineScoreThreshold = stopLineParserInfo.ransacSplineScoreThreshold_arg;
  stopLineConf->ransacSplineBinarize = stopLineParserInfo.ransacSplineBinarize_arg;
  stopLineConf->ransacSplineWindow = stopLineParserInfo.ransacSplineWindow_arg;

  stopLineConf->ransacSplineDegree = stopLineParserInfo.ransacSplineDegree_arg;

  stopLineConf->ransacSpline = stopLineParserInfo.ransacSpline_arg;
  stopLineConf->ransacLine = stopLineParserInfo.ransacLine_arg;
  stopLineConf->ransacSplineStep = stopLineParserInfo.ransacSplineStep_arg;

  stopLineConf->overlapThreshold = stopLineParserInfo.overlapThreshold_arg;

  stopLineConf->localizeAngleThreshold = stopLineParserInfo.localizeAngleThreshold_arg;
  stopLineConf->localizeNumLinePixels = stopLineParserInfo.localizeNumLinePixels_arg;

  stopLineConf->extendAngleThreshold = stopLineParserInfo.extendAngleThreshold_arg;
  stopLineConf->extendMeanDirAngleThreshold = stopLineParserInfo.extendMeanDirAngleThreshold_arg;
  stopLineConf->extendLinePixelsTangent = stopLineParserInfo.extendLinePixelsTangent_arg;
  stopLineConf->extendLinePixelsNormal = stopLineParserInfo.extendLinePixelsNormal_arg;
  stopLineConf->extendContThreshold = stopLineParserInfo.extendContThreshold_arg;
  stopLineConf->extendDeviationThreshold = stopLineParserInfo.extendDeviationThreshold_arg;
  stopLineConf->extendRectTop = stopLineParserInfo.extendRectTop_arg;
  stopLineConf->extendRectBottom = stopLineParserInfo.extendRectBottom_arg;

  stopLineConf->extendIPMAngleThreshold = stopLineParserInfo.extendIPMAngleThreshold_arg;
  stopLineConf->extendIPMMeanDirAngleThreshold = stopLineParserInfo.extendIPMMeanDirAngleThreshold_arg;
  stopLineConf->extendIPMLinePixelsTangent = stopLineParserInfo.extendIPMLinePixelsTangent_arg;
  stopLineConf->extendIPMLinePixelsNormal = stopLineParserInfo.extendIPMLinePixelsNormal_arg;
  stopLineConf->extendIPMContThreshold = stopLineParserInfo.extendIPMContThreshold_arg;
  stopLineConf->extendIPMDeviationThreshold = stopLineParserInfo.extendIPMDeviationThreshold_arg;
  stopLineConf->extendIPMRectTop = stopLineParserInfo.extendIPMRectTop_arg;
  stopLineConf->extendIPMRectBottom = stopLineParserInfo.extendIPMRectBottom_arg;

  stopLineConf->splineScoreJitter = stopLineParserInfo.splineScoreJitter_arg;
  stopLineConf->splineScoreLengthRatio = stopLineParserInfo.splineScoreLengthRatio_arg;
  stopLineConf->splineScoreAngleRatio = stopLineParserInfo.splineScoreAngleRatio_arg;
  stopLineConf->splineScoreStep = stopLineParserInfo.splineScoreStep_arg;

  stopLineConf->splineTrackingNumAbsentFrames = stopLineParserInfo.splineTrackingNumAbsentFrames_arg;
  stopLineConf->splineTrackingNumSeenFrames = stopLineParserInfo.splineTrackingNumSeenFrames_arg;

  stopLineConf->mergeSplineThetaThreshold = stopLineParserInfo.mergeSplineThetaThreshold_arg;
  stopLineConf->mergeSplineRThreshold = stopLineParserInfo.mergeSplineRThreshold_arg;
  stopLineConf->mergeSplineMeanThetaThreshold = stopLineParserInfo.mergeSplineMeanThetaThreshold_arg;
  stopLineConf->mergeSplineMeanRThreshold = stopLineParserInfo.mergeSplineMeanRThreshold_arg;
  stopLineConf->mergeSplineCentroidThreshold = stopLineParserInfo.mergeSplineCentroidThreshold_arg;

  stopLineConf->lineTrackingNumAbsentFrames = stopLineParserInfo.lineTrackingNumAbsentFrames_arg;
  stopLineConf->lineTrackingNumSeenFrames = stopLineParserInfo.lineTrackingNumSeenFrames_arg;

  stopLineConf->mergeLineThetaThreshold = stopLineParserInfo.mergeLineThetaThreshold_arg;
  stopLineConf->mergeLineRThreshold = stopLineParserInfo.mergeLineRThreshold_arg;

  stopLineConf->numStrips = stopLineParserInfo.numStrips_arg;


  stopLineConf->checkSplines = stopLineParserInfo.checkSplines_arg;
  stopLineConf->checkSplinesCurvenessThreshold = stopLineParserInfo.checkSplinesCurvenessThreshold_arg;
  stopLineConf->checkSplinesLengthThreshold = stopLineParserInfo.checkSplinesLengthThreshold_arg;
  stopLineConf->checkSplinesThetaDiffThreshold = stopLineParserInfo.checkSplinesThetaDiffThreshold_arg;
  stopLineConf->checkSplinesThetaThreshold = stopLineParserInfo.checkSplinesThetaThreshold_arg;

  stopLineConf->checkIPMSplines = stopLineParserInfo.checkIPMSplines_arg;
  stopLineConf->checkIPMSplinesCurvenessThreshold = stopLineParserInfo.checkIPMSplinesCurvenessThreshold_arg;
  stopLineConf->checkIPMSplinesLengthThreshold = stopLineParserInfo.checkIPMSplinesLengthThreshold_arg;
  stopLineConf->checkIPMSplinesThetaDiffThreshold = stopLineParserInfo.checkIPMSplinesThetaDiffThreshold_arg;
  stopLineConf->checkIPMSplinesThetaThreshold = stopLineParserInfo.checkIPMSplinesThetaThreshold_arg;

  stopLineConf->finalSplineScoreThreshold = stopLineParserInfo.finalSplineScoreThreshold_arg;

  stopLineConf->useGroundPlane = stopLineParserInfo.useGroundPlane_arg;

  stopLineConf->checkColor = stopLineParserInfo.checkColor_arg;
  stopLineConf->checkColorNumBins = stopLineParserInfo.checkColorNumBins_arg;
  stopLineConf->checkColorWindow = stopLineParserInfo.checkColorWindow_arg;
  stopLineConf->checkColorNumYellowMin = stopLineParserInfo.checkColorNumYellowMin_arg;
  stopLineConf->checkColorRGMin = stopLineParserInfo.checkColorRGMin_arg;
  stopLineConf->checkColorRGMax = stopLineParserInfo.checkColorRGMax_arg;
  stopLineConf->checkColorGBMin = stopLineParserInfo.checkColorGBMin_arg;
  stopLineConf->checkColorRBMin = stopLineParserInfo.checkColorRBMin_arg;
  stopLineConf->checkColorRBFThreshold = stopLineParserInfo.checkColorRBFThreshold_arg;
  stopLineConf->checkColorRBF = stopLineParserInfo.checkColorRBF_arg;

  stopLineConf->ipmWindowClear = stopLineParserInfo.ipmWindowClear_arg;;
  stopLineConf->ipmWindowLeft = stopLineParserInfo.ipmWindowLeft_arg;;
  stopLineConf->ipmWindowRight = stopLineParserInfo.ipmWindowRight_arg;;

  stopLineConf->checkLaneWidth = stopLineParserInfo.checkLaneWidth_arg;;
  stopLineConf->checkLaneWidthMean = stopLineParserInfo.checkLaneWidthMean_arg;;
  stopLineConf->checkLaneWidthStd = stopLineParserInfo.checkLaneWidthStd_arg;;
}

void SHOW_LINE(const Line line, char str[])
{
  cerr << str;
  cerr << "(" << line.startPoint.x << "," << line.startPoint.y << ")";
  cerr << "->";
  cerr << "(" << line.endPoint.x << "," << line.endPoint.y << ")";
  cerr << "\n";
}

void SHOW_SPLINE(const Spline spline, char str[])
{
  cerr << str;
  cerr << "(" << spline.degree << ")";
  for (int i=0; i<spline.degree+1; i++)
    cerr << " (" << spline.points[i].x << "," << spline.points[i].y << ")";
  cerr << "\n";
}


/** This fits a parabola to the entered data to get
 * the location of local maximum with sub-pixel accuracy
 *
 * \param val1 first value
 * \param val2 second value
 * \param val3 third value
 *
 * \return the computed location of the local maximum
 */
double mcvGetLocalMaxSubPixel(double val1, double val2, double val3)
{
  //build an array to hold the x-values
  double Xp[] = {1, -1, 1, 0, 0, 1, 1, 1, 1};
  cv::Mat X = cv::Mat(3, 3, CV_64FC1, Xp);

  //array to hold the y values
  double yp[] = {val1, val2, val3};
  cv::Mat y = cv::Mat(3, 1, CV_64FC1, yp);

  //solve to get the coefficients
  double Ap[3];
  cv::Mat A = cv::Mat(3, 1, CV_64FC1, Ap);
  cv::solve(X, y, A, cv::DECOMP_SVD);

  //get the local max
  double max;
  max = -0.5 * Ap[1] / Ap[0];

  //return
  return max;
}

/** This functions implements Bresenham's algorithm for getting pixels of the
 * line given its two endpoints

 *
 * \param line the input line
  *
 */
//void mcvGetLinePixels(const Line &line, vector<int> &x, vector<int> &y)
cv::Mat * mcvGetLinePixels(const Line &line)
{
  //get two end points
  cv::Point start;
  start.x  = int(line.startPoint.x); start.y = int(line.startPoint.y);
  cv::Point end;
  end.x = int(line.endPoint.x); end.y = int(line.endPoint.y);

  //get deltas
  int deltay = end.y - start.y;
  int deltax = end.x - start.x;

  //check if slope is steep, then reflect the line along y=x i.e. swap x and y
  bool steep = false;
  if (abs(deltay) > abs(deltax))
  {
    steep = true;
    //swap x and y
    int t;
    t = start.x;
    start.x = start.y;
    start.y = t;
    t = end.x;
    end.x = end.y;
    end.y = t;
  }


  //check to make sure we are going right
  bool swap = false;
  if(start.x>end.x)
  {
    //swap the two points
    cv::Point t = start;
    start = end;
    end = t;
    //we swapped
    swap = true;
  }

  //get deltas again
  deltay = end.y - start.y;
  deltax = end.x - start.x;

  //error
  int error = 0;

  //delta error
  int deltaerror = abs(deltay);

  //ystep
  int ystep = -1;
  if (deltay>=0)
  {
    ystep = 1;
  }

  //create the return matrix
  cv::Mat *pixels = new cv::Mat(end.x-start.x+1, 2, CV_32SC1);

  //loop
  int i, j;
  j = start.y;
  //list<int> x, y;
  //index for array
  int k, kupdate;
  if (!swap)
  {
    k = 0;
    kupdate = 1;
  }
  else
  {
    k = pixels->rows-1;
    kupdate = -1;
  }

  for (i=start.x; i<=end.x; i++, k+=kupdate)
  {
    //put the new point
    if(steep)
    {
      pixels->at<int>(k, 0) = j;
      pixels->at<int>(k, 1) = i;
      // 	    x.push_back(j);
      // 	    y.push_back(i);
    }
    else
    {
      pixels->at<int>(k, 0) = i;
      pixels->at<int>(k, 1) = j;
      // 	    x.push_back(i);
      // 	    y.push_back(j);
    }

    //adjust error
    error += deltaerror;
    //check
    if(2*error>=deltax)
    {
	    j = j + ystep;
	    error -= deltax;
    }
  }

  //return
  return pixels;
}

/** This functions implements Bresenham's algorithm for getting pixels of the
 * line given its two endpoints

 *
 * \param im the input image
 * \param inLine the input line
 * \param outLine the output line
 *
 */
void mcvGetLineExtent(const cv::Mat *im, const Line &inLine, Line &outLine)
{
  //first clip the input line to the image coordinates
  Line line = inLine;
  mcvIntersectLineWithBB(&inLine, cv::Size(im->cols-1, im->rows-1), &line);

  //then get the pixel values of the line in the image
  cv::Mat *pixels; //vector<int> x, y;
  pixels = mcvGetLinePixels(line); //, x, y);

  //check which way to shift the line to get multiple readings
  bool changey = false;
  if (fabs(line.startPoint.x-line.endPoint.x) >
    fabs(line.startPoint.y-line.endPoint.y))
  {
    //change the y-coordiantes
    changey = true;
  }
  char changes[] = {0, -1, 1};//, -2, 2};
  int numChanges = 3;

  //loop on the changes and get possible extents
  vector<int> startLocs;
  vector<int> endLocs;
  int endLoc;
  int startLoc;
  cv::Mat *pix = new cv::Mat(1, im->cols, FLOAT_MAT_TYPE);
  cv::Mat *rstep = new cv::Mat(pix->rows, pix->cols, FLOAT_MAT_TYPE);
  cv::Mat *fstep = new cv::Mat(pix->rows, pix->cols, FLOAT_MAT_TYPE);
  for (int c=0; c<numChanges; c++)
  {
    //get the pixels
    //for(int i=0; i<(int)x.size(); i++)
    for(int i=0; i<pixels->rows; i++)
    {
      pix->at<FLOAT_MAT_ELEM_TYPE>(0, i) =
      im->at<FLOAT_MAT_ELEM_TYPE>(
                  changey ?
                  min(max(pixels->at<int>(i, 1)+
                  changes[c],0),im->rows-1) :
                  pixels->at<int>(i, 1),
                  changey ? pixels->at<int>(i, 0) :
                  min(max(pixels->at<int>(i, 0)+
                  changes[c],0),im->cols-1));
                  // 			    changey ? min(max(y[i]+changes[c],0),im->rows-1) : y[i],
                  // 			    changey ? x[i] : min(max(x[i]+changes[c],0),im->cols-1));
    }
    //remove the mean
    cv::Scalar mean = cv::mean(*pix);
    *pix = *pix - mean;

    //now convolve with rising step to get start point
    FLOAT_MAT_ELEM_TYPE stepp[] = {-0.3000, -0.2, -0.1, 0, 0, 0.1, 0.2, 0.3, 0.4};
    // {-0.6, -0.4, -0.2, 0.2, 0.4, 0.6};
    int stepsize = 9;
    //{-0.2, -0.4, -0.2, 0, 0, 0.2, 0.4, 0.2}; //{-.75, -.5, .5, .75};
    cv::Mat step = cv::Mat(1, stepsize, FLOAT_MAT_TYPE, stepp);
    //	  SHOW_MAT(&step,"step");
    //smooth
    //	  FLOAT_MAT_ELEM_TYPE smoothp[] = {.25, .5, .25};
    //cv::Mat smooth = cv::Mat(1, 3, FLOAT_MAT_TYPE, smoothp);
    //cv::filter2D(*&step, *&step, -1, smooth);
    //SHOW_MAT(&step,"smoothed step");
    //convolve
    cv::filter2D(*pix, *rstep, -1, step);
    //get local max
    //     vector<double> localMax;
    //     vector<int> localMaxLoc;
    //     mcvGetVectorLocalMax(rstep, localMax, localMaxLoc);
    //     int startLoc = localMaxLoc[0];
    double max;
    mcvGetVectorMax(rstep, &max, &startLoc, 0);
    //check if zero
    if(max==0)
      startLoc = startLocs[c-1];

    //convolve with falling step to get end point
    //cv::flip(&step, nullptr, 1);
    //convolve
    //cv::filter2D(*pix, *fstep, -1, step);
    //get local max
    //     localMax.clear();
    //     localMaxLoc.clear();
    //     mcvGetVectorLocalMax(fstep, localMax, localMaxLoc);
    //     int endLoc = localMaxLoc[0];
    //take the negative
    rstep->convertTo(*fstep, -1);
    mcvGetVectorMax(fstep, &max, &endLoc, 0);
    //check if zero
    if(max==0)
      endLoc = endLocs[c-1];
    if(endLoc<=startLoc)
      endLoc = im->cols-1;

    //put into vectors
    startLocs.push_back(startLoc);
    endLocs.push_back(endLoc);
  }

  //get median
  startLoc = quantile(startLocs, 0);
  endLoc = quantile(endLocs, 1);
  //    for (int i=0; i<(int)startLocs.size(); i++) cout << startLocs[i] << "  ";
  //cout << "\n";
  //for (int i=0; i<(int)endLocs.size(); i++) cout << endLocs[i] << "  ";

  //get the end-point
  outLine.startPoint.x = pixels->at<int>(startLoc, 0);
  outLine.startPoint.y = pixels->at<int>(startLoc, 1);
  outLine.endPoint.x = pixels->at<int>(endLoc, 0);
  outLine.endPoint.y = pixels->at<int>(endLoc, 1);
  //     outLine.startPoint.x = x[startLoc]; outLine.startPoint.y = y[startLoc];
  //     outLine.endPoint.x = x[endLoc]; outLine.endPoint.y = y[endLoc];

  //clear
  delete pix;
  delete rstep;
  delete fstep;
  delete pixels;
  //     localMax.clear();
  //     localMaxLoc.clear();
  startLocs.clear();
  endLocs.clear();
}



/** This functions converts a line defined by its two end-points into its
 *   r and theta (origin is at top-left corner with x right and y down and
 * theta measured positive clockwise(with y pointing down) -pi < theta < pi )
 *
 * \param line input line
 * \param r the returned r (normal distance to the line from the origin)
 * \param outLine the output line
 *
 */
void mcvLineXY2RTheta(const Line &line, float &r, float &theta)
{
  //check if vertical line x1==x2
  if(line.startPoint.x == line.endPoint.x)
  {
    //r is the x
    r = fabs(line.startPoint.x);
    //theta is 0 or pi
    theta = line.startPoint.x>=0 ? 0. : CV_PI;
  }
  //check if horizontal i.e. y1==y2
  else if(line.startPoint.y == line.endPoint.y)
  {
    //r is the y
    r = fabs(line.startPoint.y);
    //theta is pi/2 or -pi/2
    theta = (float) line.startPoint.y>=0 ? CV_PI/2 : -CV_PI/2;
  }
  //general line
  else
  {
    //tan(theta) = (x2-x1)/(y1-y2)
    theta =  atan2(line.endPoint.x-line.startPoint.x,
                   line.startPoint.y-line.endPoint.y);
    //r = x*cos(theta)+y*sin(theta)
    float r1 = line.startPoint.x * cos(theta) + line.startPoint.y * sin(theta);
    r = line.endPoint.x * cos(theta) + line.endPoint.y * sin(theta);
    //adjust to add pi if necessary
    if(r1<0 || r<0)
    {
      //add pi
      theta += CV_PI;
      if(theta>CV_PI)
        theta -= 2*CV_PI;
      //take abs
      r = fabs(r);
    }
  }
}

/** This functions fits a line using the orthogonal distance to the line
    by minimizing the sum of squares of this distance.

 *
 * \param points the input points to fit the line to which is
 *    2xN matrix with x values on first row and y values on second
 * \param lineRTheta the return line [r, theta] where the line is
 *    x*cos(theta)+y*sin(theta)=r
 * \param lineAbc the return line in [a, b, c] where the line is
 *    a*x+b*y+c=0
 *
 */
void mcvFitRobustLine(const cv::Mat *points, float *lineRTheta,
                      float *lineAbc)
{
  //clone the points
  cv::Mat *cpoints  = new cv::Mat();
*cpoints  = points->clone();
  //get mean of the points and subtract from the original points
  float meanX=0, meanY=0;
  cv::Scalar mean;
  cv::Mat row1, row2;
  //get first row, compute avg and store
  row1 = cpoints->row(0);
  mean = cv::mean(row1);
  meanX = (float) mean.val[0];
  row1 = row1 - mean;
  //same for second row
  row2 = cpoints->row(1);
  mean = cv::mean(row2);
  meanY = (float) mean.val[0];
  row2 = row2 - mean;

  //compute the SVD for the centered points array
  //cv::Mat *W = new cv::Mat(2, 1, CV_32FC1);
  //cv::Mat *V = new cv::Mat(2, 2, CV_32FC1);
  //    cv::Mat *V = new cv::Mat(2, 2, CV_32fC1);
  cv::Mat *cpointst = new cv::Mat(cpoints->cols, cpoints->rows, CV_32FC1);
  int m = cpointst->rows, n = cpointst->cols, nm = std::min(m, n);
  cv::Mat *W = new cv::Mat(nm, 1, CV_32FC1);
  cv::Mat U = cv::Mat();
  cv::Mat *V = new cv::Mat(2, 2, CV_32FC1);
  *cpointst = cpoints->t();
  cv::SVD::compute(*cpointst, *W, U, *V);
  *V = V->t();
  delete cpointst;

  //get the [a,b] which is the second column corresponding to
  //smaller singular value
  float a, b, c;
  a = V->at<float>(0, 1);
  b = V->at<float>(1, 1);

  //c = -meanX*a-meanY*b
  c = -(meanX * a + meanY * b);

  //compute r and theta
  //theta = atan(b/a)
  //r = meanX cos(theta) + meanY sin(theta)
  float r, theta;
  theta = atan2(b, a);
  r = meanX * cos(theta) + meanY * sin(theta);
  //correct
  if (r<0)
  {
    //correct r
    r = -r;
    //correct theta
    theta += CV_PI;
    if (theta>CV_PI)
      theta -= 2*CV_PI;
  }
  //return
  if (lineRTheta)
  {
    lineRTheta[0] = r;
    lineRTheta[1] = theta;
  }
  if (lineAbc)
  {
    lineAbc[0] = a;
    lineAbc[1] = b;
    lineAbc[2] = c;
  }
  //clear
  delete cpoints;
  delete W;
  delete V;
}



/** This functions implements RANSAC algorithm for line fitting
 *   given an image
 *
 *
 * \param image input image
 * \param numSamples number of samples to take every iteration
 * \param numIterations number of iterations to run
 * \param threshold threshold to use to assess a point as a good fit to a line
 * \param numGoodFit number of points close enough to say there's a good fit
 * \param getEndPoints whether to get the end points of the line from the data,
 *  just intersect with the image boundaries
 * \param lineType the type of line to look for (affects getEndPoints)
 * \param lineXY the fitted line
 * \param lineRTheta the fitted line [r; theta]
 * \param lineScore the score of the line detected
 *
 */
void mcvFitRansacLine(const cv::Mat *image, int numSamples, int numIterations,
                      float threshold, float scoreThreshold, int numGoodFit,
                      bool getEndPoints, LineType lineType,
                      Line *lineXY, float *lineRTheta, float *lineScore)
{

  //get the points with non-zero pixels
  cv::Mat *points;
  points = mcvGetNonZeroPoints(image,true);
  if (!points)
    return;
  //check numSamples
  if (numSamples>points->cols)
    numSamples = points->cols;
  //subtract half
  *points = *points + 0.5;

  //normalize pixels values to get weights of each non-zero point
  //get third row of points containing the pixel values
  cv::Mat w;
  w = points->row(2);
  //normalize it
  cv::Mat *weights  = new cv::Mat();
  *weights  = w.clone();
  cv::normalize(*weights, *weights, 1, 0, cv::NORM_L1);
  //get cumulative    sum
  mcvCumSum(weights, weights);

  //random number generator
  cv::RNG rng = cv::RNG(0xffffffff);
  //matrix to hold random sample
  cv::Mat *randInd = new cv::Mat(numSamples, 1, CV_32SC1);
  cv::Mat *samplePoints = new cv::Mat(2, numSamples, CV_32FC1);
  //flag for points currently included in the set
  cv::Mat *pointIn = new cv::Mat(1, points->cols, CV_8SC1);
  //returned lines
  float curLineRTheta[2], curLineAbc[3];
  float bestLineRTheta[2]={-1.f,0.f}, bestLineAbc[3];
  float bestScore=0, bestDist=1e5;
  float dist, score;
  Line curEndPointLine={{-1.,-1.},{-1.,-1.}},
  bestEndPointLine={{-1.,-1.},{-1.,-1.}};
  //variabels for getting endpoints
  //int mini, maxi;
  float minc=1e5f, maxc=-1e5f, mind, maxd;
  float x, y, c=0.;
  cv::Point2f minp={-1., -1.}, maxp={-1., -1.};
  //outer loop
  for (int i=0; i<numIterations; i++)
  {
    //set flag to zero
    //pointIn->setTo(0);
    pointIn->setTo(0);
    //get random sample from the points
    //#warning "Using weighted sampling for Ransac Line"
    // 	cv::randArr(&rng, randInd, CV_RAND_UNI, 0, points->cols);
    mcvSampleWeighted(weights, numSamples, randInd, &rng);

    for (int j=0; j<numSamples; j++)
    {
      //flag it as included
      pointIn->at<char>(0, randInd->at<int>(j, 0)) = 1;
      //put point
      samplePoints->at<float>(0, j) =
      points->at<float>(0, randInd->at<int>(j, 0));
      samplePoints->at<float>(1, j) =
      points->at<float>(1, randInd->at<int>(j, 0));
    }

    //fit the line
    mcvFitRobustLine(samplePoints, curLineRTheta, curLineAbc);

    //get end points from points in the samplePoints
    minc = 1e5; mind = 1e5; maxc = -1e5; maxd = -1e5;
    for (int j=0; getEndPoints && j<numSamples; ++j)
    {
      //get x & y
      x = samplePoints->at<float>(0, j);
      y = samplePoints->at<float>(1, j);

      //get the coordinate to work on
      if (lineType == LINE_HORIZONTAL)
        c = x;
      else if (lineType == LINE_VERTICAL)
        c = y;
      //compare
      if (c>maxc)
      {
        maxc = c;
        maxp = cv::Point(x, y);
      }
      if (c<minc)
      {
        minc = c;
        minp = cv::Point(x, y);
      }
    } //for

    // 	fprintf(stderr, "\nminx=%f, miny=%f\n", minp.x, minp.y);
    // 	fprintf(stderr, "maxp=%f, maxy=%f\n", maxp.x, maxp.y);

    //loop on other points and compute distance to the line
    score=0;
    for (int j=0; j<points->cols; j++)
    {
      // 	    //if not already inside
      // 	    if (!pointIn->at<char>(0, j))
      // 	    {
        //compute distance to line
        dist = fabs(points->at<float>(0, j) * curLineAbc[0] +
        points->at<float>(1, j) * curLineAbc[1] + curLineAbc[2]);
        //check distance
        if (dist<=threshold)
        {
          //add this point
          pointIn->at<char>(0, j) = 1;
          //update score
          score += image->at<float>((int)(points->at<float>(1, j)-.5),
                               (int)(points->at<float>(0, j)-.5));
        }
        // 	    }
    }

    //check the number of close points and whether to consider this a good fit
    int numClose = cv::countNonZero(*pointIn);
    //cout << "numClose=" << numClose << "\n";
    if (numClose >= numGoodFit)
    {
        //get the points included to fit this line
        cv::Mat *fitPoints = new cv::Mat(2, numClose, CV_32FC1);
        int k=0;
        //loop on points and copy points included
        for (int j=0; j<points->cols; j++)
      if(pointIn->at<char>(0, j))
      {
          fitPoints->at<float>(0, k) =
        points->at<float>(0, j);
          fitPoints->at<float>(1, k) =
        points->at<float>(1, j);
          k++;

      }

      //fit the line
      mcvFitRobustLine(fitPoints, curLineRTheta, curLineAbc);

      //compute distances to new line
      dist = 0.;
      for (int j=0; j<fitPoints->cols; j++)
      {
        //compute distance to line
        x = fitPoints->at<float>(0, j);
        y = fitPoints->at<float>(1, j);
        float d = fabs( x * curLineAbc[0] +
        y * curLineAbc[1] +
        curLineAbc[2])
        * image->at<float>((int)(y-.5), (int)(x-.5));
        dist += d;

        // 		//check min and max coordinates to get extent
        // 		if (getEndPoints)
        // 		{
          // 		    //get the coordinate to work on
          // 		    if (lineType == LINE_HORIZONTAL)
          // 			c = x;
          // 		    else if (lineType == LINE_VERTICAL)
          // 			c = y;
          // 		    //compare
          // 		    if (c>maxc)
          // 		    {
            // 			maxc = c;
            // 			maxd = d;
            // 			maxp = cv::Point(x, y);
            // 		    }
            // 		    if (c<minc)
            // 		    {
              // 			minc = c;
              // 			mind = d;
              // 			minp = cv::Point(x, y);
              // 		    }

              // // 		    fprintf(stderr, "minc=%f, mind=%f, mini=%d\n", minc, mind, mini);
              // // 		    fprintf(stderr, "maxc=%f, maxd=%f, maxi=%d\n", maxc, maxd, maxi);
              // 		}
      }

      //now check if we are getting the end points
      if (getEndPoints)
      {

        //get distances
        mind = minp.x * curLineAbc[0] +
        minp.y * curLineAbc[1] + curLineAbc[2];
        maxd = maxp.x * curLineAbc[0] +
        maxp.y * curLineAbc[1] + curLineAbc[2];

        //we have the index of min and max points, and
        //their distance, so just get them and compute
        //the end points
        curEndPointLine.startPoint.x = minp.x
        - mind * curLineAbc[0];
        curEndPointLine.startPoint.y = minp.y
        - mind * curLineAbc[1];

        curEndPointLine.endPoint.x = maxp.x
        - maxd * curLineAbc[0];
        curEndPointLine.endPoint.y = maxp.y
        - maxd * curLineAbc[1];

        // 		SHOW_MAT(fitPoints, "fitPoints");
        //  		SHOW_LINE(curEndPointLine, "line");
      }

      //dist /= score;

      //clear fitPoints
      delete fitPoints;

      //check if to keep the line as best
      if (score>=scoreThreshold && score>bestScore)//dist<bestDist //(numClose > bestScore)
      {
        //update max
        bestScore = score; //numClose;
        bestDist = dist;
        //copy
        bestLineRTheta[0] = curLineRTheta[0];
        bestLineRTheta[1] = curLineRTheta[1];
        bestLineAbc[0] = curLineAbc[0];
        bestLineAbc[1] = curLineAbc[1];
        bestLineAbc[2] = curLineAbc[2];
        bestEndPointLine = curEndPointLine;
      }
    } // if numClose

    //debug
    if (DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
      char str[256];
      //convert image to rgb
      cv::Mat* im  = new cv::Mat();
* im  = image->clone();
      mcvScaleMat(image, im);
      cv::Mat *imageClr = new cv::Mat(image->rows, image->cols, CV_32FC3);
      cv::cvtColor(*im, *imageClr, cv::COLOR_GRAY2RGB);

      Line line;
      //draw current line if there
      if (curLineRTheta[0]>0)
      {
        mcvIntersectLineRThetaWithBB(curLineRTheta[0], curLineRTheta[1],
                                    cv::Size(image->cols, image->rows), &line);
        mcvDrawLine(imageClr, line, CV_RGB(1,0,0), 1);
        if (getEndPoints)
          mcvDrawLine(imageClr, curEndPointLine, CV_RGB(0,1,0), 1);
      }

      //draw best line
      if (bestLineRTheta[0]>0)
      {
        mcvIntersectLineRThetaWithBB(bestLineRTheta[0], bestLineRTheta[1],
                                    cv::Size(image->cols, image->rows), &line);
        mcvDrawLine(imageClr, line, CV_RGB(0,0,1), 1);
        if (getEndPoints)
          mcvDrawLine(imageClr, bestEndPointLine, CV_RGB(1,1,0), 1);
      }
      sprintf(str, "scor=%.2f, best=%.2f", score, bestScore);
      mcvDrawText(imageClr, str, cv::Point(30, 30), .25, CV_RGB(255,255,255));

      SHOW_IMAGE(imageClr, "Fit Ransac Line", 10);

      //clear
      delete im;
      delete imageClr;
    }//#endif
  } // for i

  //return
  if (lineRTheta)
  {
    lineRTheta[0] = bestLineRTheta[0];
    lineRTheta[1] = bestLineRTheta[1];
  }
  if (lineXY)
  {
    if (getEndPoints)
      *lineXY = bestEndPointLine;
    else
      mcvIntersectLineRThetaWithBB(lineRTheta[0], lineRTheta[1],
                                   cv::Size(image->cols-1, image->rows-1),
                                   lineXY);
  }
  if (lineScore)
    *lineScore = bestScore;

  //clear
  delete points;
  delete samplePoints;
  delete randInd;
  delete pointIn;
  delete weights;
}




/** This function gets the indices of the non-zero values in a matrix
 *
 * \param inMat the input matrix
 * \param outMat the output matrix, with 2xN containing the x and y in
 *    each column and the pixels value [xs; ys; pixel values]
 * \param floatMat whether to return floating points or integers for
 *    the outMat
 */
cv::Mat* mcvGetNonZeroPoints(const cv::Mat *inMat, bool floatMat)
{


#define MCV_GET_NZ_POINTS(inMatType, outMatType) \
     /*loop and allocate the points*/ \
     for (int i=0; i<inMat->rows; i++) \
 	for (int j=0; j<inMat->cols; j++) \
 	    if (inMat->at<inMatType>(i, j)) \
 	    { \
 		outMat->at<outMatType>(0, k) = j; \
 		outMat->at<outMatType>(1, k) = i; \
                outMat->at<outMatType>(2, k) = \
                  (outMatType) inMat->at<inMatType>(i, j); \
                k++; \
 	    } \

  int k=0;

  //get number of non-zero points
  int numnz = cv::countNonZero(*inMat);

  //allocate the point array and get the points
  cv::Mat* outMat;
  if (numnz)
  {
    if (floatMat)
      outMat = new cv::Mat(3, numnz, CV_32FC1);
    else
      outMat = new cv::Mat(3, numnz, CV_32SC1);
  }
  else
    return nullptr;

  //check type
  if (CV_MAT_TYPE(inMat->type())==FLOAT_MAT_TYPE &&
    CV_MAT_TYPE(outMat->type())==FLOAT_MAT_TYPE)
  {
    MCV_GET_NZ_POINTS(FLOAT_MAT_ELEM_TYPE, FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type())==FLOAT_MAT_TYPE &&
    CV_MAT_TYPE(outMat->type())==INT_MAT_TYPE)
  {
    MCV_GET_NZ_POINTS(FLOAT_MAT_ELEM_TYPE, INT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type())==INT_MAT_TYPE &&
    CV_MAT_TYPE(outMat->type())==FLOAT_MAT_TYPE)
  {
    MCV_GET_NZ_POINTS(INT_MAT_ELEM_TYPE, FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type())==INT_MAT_TYPE &&
    CV_MAT_TYPE(outMat->type())==INT_MAT_TYPE)
  {
    MCV_GET_NZ_POINTS(INT_MAT_ELEM_TYPE, INT_MAT_ELEM_TYPE)
  }
  else
  {
    cerr << "Unsupported type in mcvGetMatLocalMax\n";
    exit(1);
  }

  //return
  return outMat;
}


/** This function groups nearby lines
 *
 * \param lines vector of lines
 * \param lineScores scores of input lines
 * \param groupThreshold the threshold used for grouping
 * \param bbox the bounding box to intersect with
 */
void mcvGroupLines(vector<Line> &lines, vector<float> &lineScores,
                   float groupThreshold, cv::Size bbox)
{

  //convert the lines into r-theta parameters
  int numInLines = lines.size();
  vector<float> rs(numInLines);
  vector<float> thetas(numInLines);
  for (int i=0; i<numInLines; i++)
    mcvLineXY2RTheta(lines[i], rs[i], thetas[i]);

  //flag for stopping
  bool stop = false;
  while (!stop)
  {
    //minimum distance so far
    float minDist = groupThreshold+5, dist;
    vector<float>::iterator ir, jr, itheta, jtheta, minIr, minJr, minItheta, minJtheta,
    iscore, jscore, minIscore, minJscore;
    //compute pairwise distance between detected maxima
    for (ir=rs.begin(), itheta=thetas.begin(), iscore=lineScores.begin();
    ir!=rs.end(); ir++, itheta++, iscore++)
    for (jr=ir+1, jtheta=itheta+1, jscore=iscore+1;
    jr!=rs.end(); jr++, jtheta++, jscore++)
    {
      //add pi if neg
      float t1 = *itheta<0 ? *itheta : *itheta+CV_PI;
      float t2 = *jtheta<0 ? *jtheta : *jtheta+CV_PI;
      //get distance
      dist = 1 * fabs(*ir - *jr) + 1 * fabs(t1 - t2);//fabs(*itheta - *jtheta);
      //check if minimum
      if (dist<minDist)
      {
        minDist = dist;
        minIr = ir; minItheta = itheta;
        minJr = jr; minJtheta = jtheta;
        minIscore = iscore; minJscore = jscore;
      }
    }
    //check if minimum distance is less than groupThreshold
    if (minDist >= groupThreshold)
      stop = true;
    else
    {
      //put into the first
      *minIr = (*minIr + *minJr)/2;
      *minItheta = (*minItheta + *minJtheta)/2;
      *minIscore = (*minIscore + *minJscore)/2;
      //delete second one
      rs.erase(minJr);
      thetas.erase(minJtheta);
      lineScores.erase(minJscore);
    }
  }//while

  //put back the lines
  lines.clear();
  //lines.resize(rs.size());
  vector<float> newScores=lineScores;
  lineScores.clear();
  for (int i=0; i<(int)rs.size(); i++)
  {
    //get the line
    Line line;
    mcvIntersectLineRThetaWithBB(rs[i], thetas[i], bbox, &line);
    //put in place descendingly
    vector<float>::iterator iscore;
    vector<Line>::iterator iline;
    for (iscore=lineScores.begin(), iline=lines.begin();
    iscore!=lineScores.end() && newScores[i]<=*iscore; iscore++, iline++);
    lineScores.insert(iscore, newScores[i]);
    lines.insert(iline, line);
  }
  //clear
  newScores.clear();
}

/** This function groups nearby splines
 *
 * \param splines vector of splines
 * \param lineScores scores of input lines
 */
void mcvGroupSplines(vector<Spline> &splines, vector<float> &scores)

{

  //debug
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

    cv::Mat* im = new cv::Mat(240, 320, CV_8UC3);
    im->setTo(0.);
    //draw splines
    for (unsigned int i=0; i<splines.size(); i++)
      mcvDrawSpline(im, splines[i], CV_RGB(255, 0, 0), 1);

    SHOW_IMAGE(im, "Splines Before grouping", 10);
    //clear
    delete im;

  }//#endif


  //flag for stopping
  bool stop = false;
  while (!stop)
  {

    stop = true;
    //check which splines can be merged with which
    vector<Spline>::iterator spi, spj;
    vector<float>::iterator si, sj;
    for (spi=splines.begin(), si=scores.begin();
    spi!=splines.end(); spi++, si++)
    for (spj=spi+1, sj=si+1; spj!=splines.end(); spj++, sj++)
      //if to merge them
      if (mcvCheckMergeSplines(*spi, *spj, .1, 5, .2, 10, 15))
      {
        stop = false;
        //keep straighter one
        float ci, cj;
        mcvGetSplineFeatures(*spi, 0, 0, 0, 0, 0, 0, &ci);
        mcvGetSplineFeatures(*spj, 0, 0, 0, 0, 0, 0, &cj);
        //put j in i if less curved
        if (cj>ci)
        {
          //put spline j into i
          *spi = *spj;
          *si = *sj;
        }
        //remove j
        splines.erase(spj);
        scores.erase(sj);

        //break
        break;
      }
  }//while

  //debug
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

    cv::Mat* im = new cv::Mat(240, 320, CV_8UC3);
    im->setTo(0.);
    //draw splines
    for (unsigned int i=0; i<splines.size(); i++)
      mcvDrawSpline(im, splines[i], CV_RGB(255, 0, 0), 1);

    SHOW_IMAGE(im, "Splines After grouping", 10);
    //clear
    delete im;

  }//#endif

}

/** \brief This function groups together bounding boxes
 *
 * \param size the size of image containing the lines
 * \param boxes a vector of output grouped bounding boxes
 * \param type the type of lines (LINE_HORIZONTAL or LINE_VERTICAL)
 * \param groupThreshold the threshold used for grouping (ratio of overlap)
 */
void mcvGroupBoundingBoxes(vector<cv::Rect> &boxes, LineType type,
                           float groupThreshold)
{
  bool cont = true;

  //Todo: check if to intersect with bounding box or not

  //save boxes
  //vector<cv::Rect> tboxes = boxes;

  //loop to get the largest overlap (according to type) and check
  //the overlap ratio
  float overlap, maxOverlap;
  while(cont)
  {
    maxOverlap =  overlap = -1e5;
    //loop on lines and get max overlap
    vector<cv::Rect>::iterator i, j, maxI, maxJ;
    for(i = boxes.begin(); i != boxes.end(); i++)
    {
      for(j = i+1; j != boxes.end(); j++)
      {
        switch(type)
        {
          case LINE_VERTICAL:
            //get one with smallest x, and compute the x2 - x1 / cols of smallest
            //i.e. (x12 - x21) / (x22 - x21)
            overlap = i->x < j->x  ?
            (i->x + i->width - j->x) / (float)j->width :
            (j->x + j->width - i->x) / (float)i->width;

            break;

          case LINE_HORIZONTAL:
            //get one with smallest y, and compute the y2 - y1 / height of smallest
            //i.e. (y12 - y21) / (y22 - y21)
            overlap = i->y < j->y  ?
            (i->y + i->height - j->y) / (float)j->height :
            (j->y + j->height - i->y) / (float)i->height;

            break;

        } //switch

        //get maximum
        if(overlap > maxOverlap)
        {
          maxI = i;
          maxJ = j;
          maxOverlap = overlap;
        }
      } //for j
    } // for i
    // 	//debug
    // 	if(DEBUG_LINES) {
    // 	    cout << "maxOverlap=" << maxOverlap << endl;
    // 	    cout << "Before grouping\n";
    // 	    for(unsigned int k=0; k<boxes.size(); ++k)
    // 		SHOW_RECT(boxes[k]);
    // 	}

    //now check the max overlap found against the threshold
    if (maxOverlap >= groupThreshold)
    {
      //combine the two boxes
      *maxI  = cv::Rect(min((*maxI).x, (*maxJ).x),
                      min((*maxI).y, (*maxJ).y),
                      max((*maxI).width, (*maxJ).width),
                      max((*maxI).height, (*maxJ).height));
                      //delete the second one
                      boxes.erase(maxJ);
    }
    else
      //stop
      cont = false;

    // 	//debug
    // 	if(DEBUG_LINES) {
    // 	    cout << "After grouping\n";
    // 	    for(unsigned int k=0; k<boxes.size(); ++k)
    // 		SHOW_RECT(boxes[k]);
    // 	}
  } //while
}

/** This function performs a RANSAC validation step on the detected lines
 *
 * \param image the input image
 * \param inLines vector of lines
 * \param outLines vector of grouped lines
 * \param groupThreshold the threshold used for grouping
 * \param bbox the bounding box to intersect with
 * \param lineType the line type to work on (horizontal or vertical)
 */
void mcvGetRansacLines(const cv::Mat *im, vector<Line> &lines,
                       vector<float> &lineScores, LaneDetectorConf *lineConf,
                       LineType lineType)
{
  //check if to binarize image
  cv::Mat *image  = new cv::Mat();
*image  = im->clone();
  if (lineConf->ransacLineBinarize)
    mcvBinarizeImage(image);

  int cols = image->cols-1;
  int rows = image->rows-1;
  //try grouping the lines into regions
  //float groupThreshold = 15;
  mcvGroupLines(lines, lineScores, lineConf->groupThreshold,
                cv::Size(cols, rows));

  //group bounding boxes of lines
  float overlapThreshold = lineConf->overlapThreshold; //0.5; //.8;
  vector<cv::Rect> boxes;
  mcvGetLinesBoundingBoxes(lines, lineType, cv::Size(cols, rows),
                           boxes);
  mcvGroupBoundingBoxes(boxes, lineType, overlapThreshold);
  //     mcvGroupLinesBoundingBoxes(lines, lineType, overlapThreshold,
  // 			       cv::Size(cols, rows), boxes);

  //     //check if there're no lines, then check the whole image
  //     if (boxes.size()<1)
  // 	boxes.push_back(cv::Rect(0, 0, cols-1, rows-1));

  int window = lineConf->ransacLineWindow; //15;
  vector<Line> newLines;
  vector<float> newScores;
  for (int i=0; i<(int)boxes.size(); i++) //lines
  {
    // 	fprintf(stderr, "i=%d\n", i);
    //Line line = lines[i];
    cv::Rect mask, box;
    //get box
    box = boxes[i];
    switch (lineType)
    {
      case LINE_HORIZONTAL:
      {
        //get extent
        //int ystart = (int)fmax(fmin(line.startPoint.y, line.endPoint.y)-window, 0);
        //int yend = (int)fmin(fmax(line.startPoint.y, line.endPoint.y)+window, rows-1);
        int ystart = (int)fmax(box.y - window, 0);
        int yend = (int)fmin(box.y + box.height+ window, rows-1);
        //get the mask
        mask = cv::Rect(0, ystart, cols, yend-ystart+1);
      }
      break;

      case LINE_VERTICAL:
      {
        //get extent of window to search in
        //int xstart = (int)fmax(fmin(line.startPoint.x, line.endPoint.x)-window, 0);
        //int xend = (int)fmin(fmax(line.startPoint.x, line.endPoint.x)+window, cols-1);
        int xstart = (int)fmax(box.x - window, 0);
        int xend = (int)fmin(box.x + box.height + window, cols-1);
        //get the mask
        mask = cv::Rect(xstart, 0, xend-xstart+1, rows);
      }
      break;
    }
    //get the subimage to work on
    cv::Mat *subimage  = new cv::Mat();
*subimage  = image->clone();
    //clear all but the mask
    mcvSetMat(subimage, mask, 0);

    //get the RANSAC line in this part
    //int numSamples = 5, numIterations = 10, numGoodFit = 15;
    //float threshold = 0.5;
    float lineRTheta[2]={-1,0};
    float lineScore;
    Line line;
    mcvFitRansacLine(subimage, lineConf->ransacLineNumSamples,
                     lineConf->ransacLineNumIterations,
                     lineConf->ransacLineThreshold,
                     lineConf->ransacLineScoreThreshold,
                     lineConf->ransacLineNumGoodFit,
                     lineConf->getEndPoints, lineType,
                     &line, lineRTheta, &lineScore);

    //store the line if found and make sure it's not
    //near horizontal or vertical (depending on type)
    //#warning "check this screening in ransacLines"
    if (lineRTheta[0]>=0)
    {
      bool put =true;
      switch(lineType)
      {
        case LINE_HORIZONTAL:
          //make sure it's not vertical
          if (fabs(lineRTheta[1]) < 30*CV_PI/180)
            put = false;
          break;

        case LINE_VERTICAL:
          //make sure it's not horizontal
          if((fabs(lineRTheta[1]) > 20*CV_PI/180))
            put = false;
          break;
      }
      if (put)
      {
        newLines.push_back(line);
        newScores.push_back(lineScore);
      }
    } // if

    //debug
    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

      //get string
      char str[256];
      switch (lineType)
      {
        case LINE_HORIZONTAL:
          sprintf(str, "Subimage Line H #%d", i);
          break;
        case LINE_VERTICAL:
          sprintf(str, "Subimage Line V #%d", i);
          break;
      }
      //convert image to rgb
      mcvScaleMat(subimage, subimage);
      cv::Mat *subimageClr = new cv::Mat(subimage->rows, subimage->cols,
                                       CV_32FC3);
      cv::cvtColor(*subimage, *subimageClr, cv::COLOR_GRAY2RGB);
      //draw rectangle
      //      	    mcvDrawRectangle(subimageClr, box,
                  // 			     CV_RGB(255, 255, 0), 1);
      mcvDrawRectangle(subimageClr, mask, CV_RGB(255, 255, 255), 1);

      //draw line
      if (lineRTheta[0]>0)
        mcvDrawLine(subimageClr, line, CV_RGB(1,0,0), 1);
      SHOW_IMAGE(subimageClr, str, 10);
      //clear
      delete subimageClr;
    }//#endif

    //clear
    delete subimage;
  } // for i

  //group lines
  vector<Line> oldLines;
  if (DEBUG_LINES)
    oldLines = lines;
  lines.clear();
  lineScores.clear();
  //#warning "not grouping at end of getRansacLines"
  //mcvGroupLines(newLines, newScores, lineConf->groupThreshold, cv::Size(cols, rows));
  lines = newLines;
  lineScores = newScores;

  //draw splines
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

    //get string
    char title[256]; //str[256],
    switch (lineType)
    {
      case LINE_HORIZONTAL:
        sprintf(title, "Lines H");
        break;
      case LINE_VERTICAL:
        sprintf(title, "Lines V");
        break;
    }
    //convert image to rgb
    cv::Mat* im2  = new cv::Mat();
* im2  = im->clone();
    mcvScaleMat(im2, im2);
    cv::Mat *imClr = new cv::Mat(im->rows, im->cols, CV_32FC3);
    cv::cvtColor(*im2, *imClr, cv::COLOR_GRAY2RGB);
    cv::Mat* imClr2  = new cv::Mat();
* imClr2  = imClr->clone();
    delete im2;

    //draw spline
    for (unsigned int j=0; j<lines.size(); j++)
      mcvDrawLine(imClr, lines[j], CV_RGB(0,1,0), 1);
    SHOW_IMAGE(imClr, title, 10);

    //draw spline
    for (unsigned int j=0; j<oldLines.size(); j++)
      mcvDrawLine(imClr2, oldLines[j], CV_RGB(1,0,0), 1);
    SHOW_IMAGE(imClr2, "Input Lines", 10);

    //clear
    delete imClr;
    delete imClr2;
    oldLines.clear();
  }//#endif

//     //put lines back in descending order of scores
//     lines.clear();
//     lineScores.clear();
//     vector<Line>::iterator li;
//     vector<float>::iterator si;
//     for (int i=0; i<(int)newLines.size(); i++)
//     {
  // 	//get its position
  // 	for (li=lines.begin(), si=lineScores.begin();
  // 	     si!=lineScores.end() && newScores[i]<=*si;
  // 	     si++, li++);
  // 	lines.insert(li, newLines[i]);
  // 	lineScores.insert(si, newScores[i]);
  //     }

  //clean
  boxes.clear();
  newLines.clear();
  newScores.clear();
  delete image;
}

/** This function sets the matrix to a value except for the mask window passed in
 *
 * \param inMat input matrix
 * \param mask the rectangle defining the mask: (xleft, ytop, cols, rows)
 * \param val the value to put
 */
void  mcvSetMat(cv::Mat *inMat, cv::Rect mask, double val)
{

  //get x-end points of region to work on, and work on the whole image rows
  //(int)fmax(fmin(line.startPoint.x, line.endPoint.x)-xwindow, 0);
  int xstart = mask.x, xend = mask.x + mask.width-1;
  //xend = (int)fmin(fmax(line.startPoint.x, line.endPoint.x), width-1);
  int ystart = mask.y, yend = mask.y + mask.height-1;

  //set other two windows to zero
  cv::Rect rect;
  //part to the left of required region
  rect = cv::Rect(0, 0, xstart-1, inMat->rows);
  if (rect.x<inMat->cols && rect.y<inMat->rows &&
    rect.x>=0 && rect.y>=0 && rect.width>0 && rect.height>0)
  {
    (*inMat)(rect).setTo(val);
  }
  //part to the right of required region
  rect = cv::Rect(xend+1, 0, inMat->cols-xend-1, inMat->rows);
  if (rect.x<inMat->cols && rect.y<inMat->rows &&
    rect.x>=0 && rect.y>=0 && rect.width>0 && rect.height>0)
  {
    (*inMat)(rect).setTo(val);
  }

  //part to the top
  rect = cv::Rect(xstart, 0, mask.width, ystart-1);
  if (rect.x<inMat->cols && rect.y<inMat->rows &&
    rect.x>=0 && rect.y>=0 && rect.width>0 && rect.height>0)
  {
    (*inMat)(rect).setTo(val);
  }

  //part to the bottom
  rect = cv::Rect(xstart, yend+1, mask.width, inMat->rows-yend-1);
  if (rect.x<inMat->cols && rect.y<inMat->rows &&
    rect.x>=0 && rect.y>=0 && rect.width>0 && rect.height>0)
  {
    (*inMat)(rect).setTo(val);
  }
}


/** This function sorts a set of points
 *
 * \param inPOints Nx2 matrix of points [x,y]
 * \param outPOints Nx2 matrix of points [x,y]
 * \param dim the dimension to sort on (0: x, 1:y)
 * \param dir direction of sorting (0: ascending, 1:descending)
 */
void mcvSortPoints(const cv::Mat *inPoints, cv::Mat *outPoints,
                   int dim, int dir)
{
  //make a copy of the input
  cv::Mat *pts  = new cv::Mat();
*pts  = inPoints->clone();

  //clear the output
  //outPoints->setTo(0);

  //make the list of sorted indices
  list<int> sorted;
  list<int>::iterator sortedi;
  int i, j;

  //loop on elements and adjust its index
  for (i=0; i<pts->rows; i++)
  {
    //if ascending
    if (dir==0)
      for (sortedi = sorted.begin();
           sortedi != sorted.end() &&
           (pts->at<float>(i, dim) >=
           outPoints->at<float>(*sortedi, dim));
           sortedi++);
    //descending
    else
      for (sortedi = sorted.begin();
           sortedi != sorted.end() &&
           (pts->at<float>(i, dim) <=
           outPoints->at<float>(*sortedi, dim));
           sortedi++);

    //found the position, so put it into sorted
    sorted.insert(sortedi, i);
  }

  //sorted the array, so put back
  for (i=0, sortedi=sorted.begin(); sortedi != sorted.end(); sortedi++, i++)
    for(j=0; j<outPoints->cols; j++)
      outPoints->at<float>(i, j) = pts->at<float>(*sortedi, j);

  //clear
  delete pts;
  sorted.clear();
}

/** This function fits a Bezier spline to the passed input points
 *
 * \param points the input points
 * \param degree the required spline degree
 * \return spline the returned spline
 */
Spline mcvFitBezierSpline(cv::Mat *points, int degree)
{

  //set the degree
  Spline spline;
  spline.degree = degree;

  //get number of points
  int n = points->rows;
  //float step = 1./(n-1);

  //sort the pointa
  mcvSortPoints(points, points, 1, 0);
  //     SHOW_MAT(points, "Points after sorting:");

  //get first point and distance between points
  cv::Point2f  p0 = cv::Point(points->at<float>(0, 0),
                                  points->at<float>(0, 1));

  float diff = 0.f;
  float *us = new float[points->rows];
  us[0] = 0;
  for (int i=1; i<points->rows; ++i)
  {
    float dx = points->at<float>(i, 0) -
      points->at<float>(i-1, 0);
    float dy = points->at<float>(i, 1) -
      points->at<float>(i-1, 1);
    us[i] = cv::sqrt(dx*dx + dy*dy) + us[i-1];
    // 	diff += us[i];;
  }
  diff = us[points->rows-1];

  //float y0 = points->at<float>(0, 1);
  //float ydiff = points->at<float>(points->rows-1, 1) - y0;

  //M matrices: M2 for quadratic (degree 2) and M3 for cubic
  float M2[] = {1, -2, 1,
                -2, 2, 0,
                1, 0, 0};
  float M3[] = {-1, 3, -3, 1,
                3, -6, 3, 0,
                -3, 3, 0, 0,
                1, 0, 0, 0};

  //M matrix for Bezier
  cv::Mat M;

  //Basis matrix
  cv::Mat *B;

  //u value for points to create the basis matrix
  float u = 0.f;

  //switch on the degree
  switch(degree)
  {
    //Quadratic spline
    case 2:
      //M matrix
      M = cv::Mat(3, 3, CV_32FC1, M2);

      //create the basis matrix
      B = new cv::Mat(n, 3, CV_32FC1);
      for (int i=0; i<B->rows; i++) //u+=step
      {
        //get u as ratio of y-coordinate
        // 	    u  = i / ((float)n-1);

        //  	    u = (points->at<float>(i, 1) - y0) / ydiff;

        // 	    float dx = points->at<float>(i, 0) - p0.x;
        // 	    float dy = points->at<float>(i, 1) - p0.y;
        // 	    u = cv::sqrt(dx*dx + dy*dy) / diff;
        u = us[i] / diff;

        B->at<float>(i, 2) = 1;  //1
        B->at<float>(i, 1) = u;  //u
        B->at<float>(i, 0) = u*u;  //u^2
      }
      break;

    //Cubic spline
    case 3:
      //M matrix
      M = cv::Mat(4, 4, CV_32FC1, M3);

      //create the basis matrix
      B = new cv::Mat(n, 4, CV_32FC1);
      for (int i=0; i<B->rows; i++) //, u+=step)
      {
        //get u as ratio of y-coordinate
        // 	    u  = i / ((float)n-1);

        //  	    u = (points->at<float>(i, 1) - y0) / ydiff;

        // 	    float dx = points->at<float>(i, 0) - p0.x;
        // 	    float dy = points->at<float>(i, 1) - p0.y;
        // 	    u = cv::sqrt(dx*dx + dy*dy) / diff;
        u = us[i] / diff;

        B->at<float>(i, 3) = 1;  //1
        B->at<float>(i, 2) = u;  //u
        B->at<float>(i, 1) = u*u;  //u^2
        B->at<float>(i, 0) = u*u*u;  //u^2
      }
      break;
  } // switch degree

  //multiply B by M
  *B = *B * M;;


  //return the required control points by LS
  cv::Mat *sp = new cv::Mat(degree+1, 2, CV_32FC1);
  cv::solve(*B, *points, *sp, cv::DECOMP_SVD);

  //     SHOW_MAT(sp, "Spline points:");

  //put back into spline
//  memcpy((float *)spline.points, ((float*)sp->data), sizeof(float)*(spline.degree+1)*2);
  for (int i = 0; i < spline.degree+1; i++) {
      spline.points[i] = cv::Point2f(sp->at<float>(i, 0), sp->at<float>(i, 1));
  }
  //     if(spline.points[0].x<0)
  // 	SHOW_MAT(points, "INput Points");

  //clear
  delete B;
  delete sp;
  delete [] us;

  //return
  return spline;
}



/** This function evaluates Bezier spline with given resolution
 *
 * \param spline input spline
 * \param h the input resolution
 * \param tangents compute tangents at the two endpoints [t0; t1]
 * \return computed points in an array Nx2 [x,y]
 */
cv::Mat* mcvEvalBezierSpline(const Spline &spline, float h, cv::Mat *tangents)
{
  //compute number of points to return
  int n = (int)(1./h)+1;

  //allocate the points
  cv::Mat *points = new cv::Mat(n, 2, CV_32FC1);

  //M matrices
  cv::Mat M;
  float M2[] = {1, -2, 1,
  -2, 2, 0,
  1, 0, 0};
  float M3[] = {-1, 3, -3, 1,
  3, -6, 3, 0,
  -3, 3, 0, 0,
  1, 0, 0, 0};

  //spline points
  cv::Mat *sp = new cv::Mat(spline.degree+1, 2, CV_32FC1);
//  memcpy(((float*)sp->data), (float *)spline.points,
//         sizeof(float)*(spline.degree+1)*2);
  for (int i = 0; i < spline.degree+1; i++) {
      sp->at<float>(i, 0) = spline.points[i].x;
      sp->at<float>(i, 1) = spline.points[i].y;
  }

  //abcd
  cv::Mat *abcd;

  float P[2], dP[2], ddP[2], dddP[2];
  float h2 = h*h, h3 = h2*h;

  //switch the degree
  switch(spline.degree)
  {
    //Quadratic
    case 2:
      //get M matrix
      M = cv::Mat(3, 3, CV_32FC1, M2);

      //get abcd where a=row 0, b=row 1, ...
      abcd = new cv::Mat(3, 2, CV_32FC1);
      *abcd = M * *sp;;

      //P = c
      P[0] = abcd->at<float>(2, 0);
      P[1] = abcd->at<float>(2, 1);

      //dP = b*h+a*h^2
      dP[0] = abcd->at<float>(1, 0)*h +
      abcd->at<float>(0, 0)*h2;
      dP[1] = abcd->at<float>(1, 1)*h +
      abcd->at<float>(0, 1)*h2;

      //ddP = 2*a*h^2
      ddP[0] = 2 * abcd->at<float>(0, 0)*h2;
      ddP[1] = 2 * abcd->at<float>(0, 1)*h2;

      //loop and put points
      for (int i=0; i<n; i++)
      {
        //put point
        points->at<float>(i, 0) = P[0];
        points->at<float>(i, 1) = P[1];

        //update
        P[0] += dP[0]; P[1] += dP[1];
        dP[0] += ddP[0]; dP[1] += ddP[1];
      }

      //put tangents
      if (tangents)
      {
        //t0 = b
        tangents->at<float>(0, 0) =
        abcd->at<float>(1, 0);
        tangents->at<float>(0, 1) =
        abcd->at<float>(1, 1);
        //t1 = 2*a + b
        tangents->at<float>(1, 0) = 2 *
        abcd->at<float>(0, 0) +
        abcd->at<float>(1, 0);
        tangents->at<float>(1, 1) = 2 *
        abcd->at<float>(0, 1) +
        abcd->at<float>(1, 1);
      }
      break;

    /*Cubic*/
    case 3:
      //get M matrix
      M = cv::Mat(4, 4, CV_32FC1, M3);

      //get abcd where a=row 0, b=row 1, ...
      abcd = new cv::Mat(4, 2, CV_32FC1);
      *abcd = M * *sp;;

      //P = d
      P[0] = abcd->at<float>(3, 0);
      P[1] = abcd->at<float>(3, 1);

      //dP = c*h + b*h^2+a*h^3
      dP[0] = abcd->at<float>(2, 0)*h +
      abcd->at<float>(1, 0)*h2 +
      abcd->at<float>(0, 0)*h3;
      dP[1] = abcd->at<float>(2, 1)*h +
      abcd->at<float>(1, 1)*h2 +
      abcd->at<float>(0, 1)*h3;

      //dddP = 6 * a * h3
      dddP[0] = 6 * abcd->at<float>(0, 0) * h3;
      dddP[1] = 6 * abcd->at<float>(0, 1) * h3;

      //ddP = 2*b*h2 + 6*a*h3
      ddP[0] = 2 * abcd->at<float>(1, 0) * h2 + dddP[0];
      ddP[1] = 2 * abcd->at<float>(1, 1) * h2 + dddP[1];

      //loop and put points
      for (int i=0; i<n; i++)
      {
        //put point
        points->at<float>(i, 0) = P[0];
        points->at<float>(i, 1) = P[1];

        //update
        P[0] += dP[0]; P[1] += dP[1];
        dP[0] += ddP[0]; dP[1] += ddP[1];
        ddP[0] += dddP[0]; ddP[1] += dddP[1];
      }

      //put tangents
      if (tangents)
      {
        //t0 = c
        tangents->at<float>(0, 0) = abcd->at<float>(2, 0);
        tangents->at<float>(0, 1) = abcd->at<float>(2, 1);
        //t1 = 3*a + 2*b + c
        tangents->at<float>(1, 0) =
          3 * abcd->at<float>(0, 0) +
          2 * abcd->at<float>(1, 0) +
          abcd->at<float>(2, 0);
        tangents->at<float>(1, 1) =
          3 * abcd->at<float>(0, 1) +
          2 * abcd->at<float>(1, 1) +
          abcd->at<float>(2, 1);
      }
      break;
    default:
      // avoid error when using cv::releaseMat below and degree is neither 2 or 3
      abcd = new cv::Mat(3, 2, CV_32FC1);
      break;
  }

  //clear
  delete abcd;
  delete sp;

  //return
  return points;
}


/** This function returns pixel coordinates for the Bezier
 * spline with the given resolution.
 *
 * \param spline input spline
 * \param h the input resolution
 * \param box the bounding box
 * \param extendSpline whether to extend spline with straight lines or
 *          not (default false)
 * \return computed points in an array Nx2 [x,y], returns nullptr if empty output
 */
cv::Mat* mcvGetBezierSplinePixels(Spline &spline, float h, cv::Size box,
                                bool extendSpline)
{
  //get the points belonging to the spline
  cv::Mat *tangents = new cv::Mat(2, 2, CV_32FC1);
  cv::Mat *points = mcvEvalBezierSpline(spline, h, tangents);

  //pixelize the spline
  //cv::Mat *inpoints = new cv::Mat(points->rows, 1, CV_8SC1);
  //cv::set(, cv::Scalar value, const cv::Arr* mask=nullptr);
  list<int> inpoints;
  list<int>::iterator inpointsi;
  int lastin = -1, numin = 0;
  for (int i=0; i<points->rows; i++)
  {
    //round
    points->at<float>(i, 0) = cvRound(points->at<float>(i, 0));
    points->at<float>(i, 1) = cvRound(points->at<float>(i, 1));

    //check boundaries
    if(points->at<float>(i, 0) >= 0 &&
      points->at<float>(i, 0) < box.width &&
      points->at<float>(i, 1) >= 0 &&
      points->at<float>(i, 1) < box.height)
    {
      //it's inside, so check if the same as last one
      if(lastin<0 ||
        (lastin>=0 &&
        !(points->at<float>(lastin, 1)==
        points->at<float>(i, 1) &&
        points->at<float>(lastin, 0)==
        points->at<float>(i, 0) )) )
      {
        //put inside
        //inpoints->at<char>(i, 0) = 1;
        inpoints.push_back(i);
        lastin = i;
        numin++;
      }
    }
  }

  //check if to extend the spline with lines
  cv::Mat *pixelst0, *pixelst1;
  if (extendSpline)
  {
    //get first point inside
    int p0 = inpoints.front();
    //extend from the starting point by going backwards along the tangent
    //line from that point to the start of spline
    Line line;
    line.startPoint = cv::Point(points->at<float>(p0, 0) - 10 *
                                   tangents->at<float>(0, 0),
                                   points->at<float>(p0, 1) - 10 *
                                   tangents->at<float>(0, 1));
    line.endPoint = cv::Point(points->at<float>(p0, 0),
                                 points->at<float>(p0, 1));
    //intersect the line with the bounding box
    mcvIntersectLineWithBB(&line, cv::Size(box.width-1, box.height-1), &line);
    //get line pixels
    pixelst0 = mcvGetLinePixels(line);
    numin += pixelst0->rows;

    //get last point inside
    int p1 = inpoints.back();
    //extend from end of spline along tangent
    line.endPoint = cv::Point(points->at<float>(p1, 0) + 10 *
                                 tangents->at<float>(1, 0),
                                 points->at<float>(p1, 1) + 10 *
                                 tangents->at<float>(1, 1));
    line.startPoint = cv::Point(points->at<float>(p1, 0),
                                   points->at<float>(p1, 1));
    //intersect the line with the bounding box
    mcvIntersectLineWithBB(&line, cv::Size(box.width-1, box.height-1), &line);
    //get line pixels
    pixelst1 = mcvGetLinePixels(line);
    numin += pixelst1->rows;
  }

  //put the results in another matrix
  cv::Mat *rpoints;
  if (numin>0)
    rpoints = new cv::Mat(numin, 2, CV_32SC1);
  else
  {
    return nullptr;
  }


  //first put extended line segment if available
  if(extendSpline)
  {
    //copy
//    memcpy(rpoints->ptr(0, 0), ((float*)pixelst0->data),
//           sizeof(float)*2*pixelst0->rows);
    pixelst0->copyTo(rpoints->rowRange(0, pixelst0->rows));
  }

  //put spline pixels
  int ri = extendSpline ? pixelst0->rows : 0;
  for (inpointsi=inpoints.begin();
  inpointsi!=inpoints.end(); ri++, inpointsi++)
  {
    rpoints->at<int>(ri, 0) = (int)points->at<float>(*inpointsi, 0);
    rpoints->at<int>(ri, 1) = (int)points->at<float>(*inpointsi, 1);
  }

  //put second extended piece of spline
  if(extendSpline)
  {
    //copy
//    memcpy(rpoints->ptr(ri, 0), ((float*)pixelst1->data),
//           sizeof(float)*2*pixelst1->rows);
    pixelst1->copyTo(rpoints->rowRange(ri, ri + pixelst1->rows));

    //clear
    delete pixelst0;
    delete pixelst1;
  }


  //release
  //    delete inpoints;
  delete points;
  delete tangents;
  inpoints.clear();

  //return
  return rpoints;
}


/** This function performs a RANSAC validation step on the detected lines to
 * get splines
 *
 * \param image the input image
 * \param lines vector of input lines to refine
 * \param lineSCores the line scores input
 * \param groupThreshold the threshold used for grouping
 * \param bbox the bounding box to intersect with
 * \param lineType the line type to work on (horizontal or vertical)
 * \param prevSplines the previous splines to use in initializing the detection
 */
void mcvGetRansacSplines(const cv::Mat *im, vector<Line> &lines,
                         vector<float> &lineScores, LaneDetectorConf *lineConf,
                         LineType lineType, vector<Spline> &splines,
                         vector<float> &splineScores, LineState* state)
{
  //check if to binarize image
  cv::Mat *image  = new cv::Mat();
*image  = im->clone();
  if (lineConf->ransacSplineBinarize)
    mcvBinarizeImage(image); // ((topmost-intro . 147431))

    int cols = image->cols;
  int rows = image->rows;
  //try grouping the lines into regions
  //float groupThreshold = 15;
  //#warning "no line grouping in getRansacSplines"
  vector<Line> tlines = lines;
  vector<float> tlineScores = lineScores;
  mcvGroupLines(tlines, tlineScores, lineConf->groupThreshold,
                cv::Size(cols-1, rows-1));

  //put the liens into the prevSplines to initialize it
  for (unsigned int i=0; state->ipmSplines.size() &&
    i<lines.size(); ++i)
  {
    //get spline and push back
    Spline spline = mcvLineXY2Spline
    (lines[i], lineConf->ransacSplineDegree);
    state->ipmSplines.push_back(spline);
  }

  //group bounding boxes of lines
  float overlapThreshold = lineConf->overlapThreshold; //0.5; //.8;
  vector<cv::Rect> boxes;
  cv::Size size = cv::Size(cols, rows);
  mcvGetLinesBoundingBoxes(tlines, lineType, size, boxes);
  mcvGroupBoundingBoxes(boxes, lineType, overlapThreshold);
  //     mcvGroupLinesBoundingBoxes(tlines, lineType, overlapThreshold,
  // 			       cv::Size(cols, rows), boxes);
  tlines.clear();
  tlineScores.clear();

  //add bounding boxes from previous frame
  //#warning "Turned off adding boxes from previous frame"
  //     boxes.insert(boxes.end(), state->ipmBoxes.begin(),
  // 		 state->ipmBoxes.end());

  //     //check if there're no lines, then check the whole image
  //     if (boxes.size()<1)
  // 	boxes.push_back(cv::Rect(0, 0, cols-1, rows-1));

  int window = lineConf->ransacSplineWindow; //15;
  vector<Spline> newSplines;
  vector<float> newSplineScores;
  for (int i=0; i<(int)boxes.size(); i++) //lines
  {
    //Line line = lines[i];

    cv::Rect mask, box;

    //get box
    box = boxes[i];

    switch (lineType)
    {
      case LINE_HORIZONTAL:
      {
        //get extent
        //int ystart = (int)fmax(fmin(line.startPoint.y, line.endPoint.y)-window, 0);
        //int yend = (int)fmin(fmax(line.startPoint.y, line.endPoint.y)+window, rows-1);
        int ystart = (int)fmax(box.y - window, 0);
        int yend = (int)fmin(box.y + box.height + window, rows-1);
        //get the mask
        mask = cv::Rect(0, ystart, cols, yend-ystart+1);
      }
      break;

      case LINE_VERTICAL:
      {
        //get extent of window to search in
        //int xstart = (int)fmax(fmin(line.startPoint.x, line.endPoint.x)-window, 0);
        //int xend = (int)fmin(fmax(line.startPoint.x, line.endPoint.x)+window, cols-1);
        int xstart = (int)fmax(box.x - window, 0);
        int xend = (int)fmin(box.x + box.width + window, cols-1);
        //get the mask
        mask = cv::Rect(xstart, 0, xend-xstart+1, rows);
      }
      break;
    }
    //get the subimage to work on
    cv::Mat *subimage  = new cv::Mat();
*subimage  = image->clone();
    //clear all but the mask
    mcvSetMat(subimage, mask, 0);

    //get the RANSAC spline in this part
    //int numSamples = 5, numIterations = 10, numGoodFit = 15;
    //float threshold = 0.5;
    Spline spline;
    float splineScore;
    //resolution to use in pixelizing the spline
    float h = lineConf->ransacSplineStep; // .1; //1. / max(image->cols, image->rows);
    mcvFitRansacSpline(subimage, lineConf->ransacSplineNumSamples,
                       lineConf->ransacSplineNumIterations,
                       lineConf->ransacSplineThreshold,
                       lineConf->ransacSplineScoreThreshold,
                       lineConf->ransacSplineNumGoodFit,
                       lineConf->ransacSplineDegree, h,
                       &spline, &splineScore,
                       lineConf->splineScoreJitter,
                       lineConf->splineScoreLengthRatio,
                       lineConf->splineScoreAngleRatio,
                       lineConf->splineScoreStep,
                       &state->ipmSplines);

    //store the line if found
    if (spline.degree > 0)
    {
      newSplines.push_back(spline);
      newSplineScores.push_back(splineScore);
    }

    //debug
    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

      //get string
      char str[256], title[256];;
      switch (lineType)
      {
        case LINE_HORIZONTAL:
          sprintf(title, "Subimage Spline H #%d", i);
          break;
        case LINE_VERTICAL:
          sprintf(title, "Subimage Spline V #%d", i);
          break;
      }
      //convert image to rgb
      mcvScaleMat(subimage, subimage);
      cv::Mat *subimageClr = new cv::Mat(subimage->rows, subimage->cols,
                                       CV_32FC3);
      cv::cvtColor(*subimage, *subimageClr, cv::COLOR_GRAY2RGB);

      //draw rectangle
      //mcvDrawRectangle(subimageClr, box,
      //	     CV_RGB(255, 255, 0), 1);
      mcvDrawRectangle(subimageClr, mask, CV_RGB(255, 255, 255), 1);

      //put text
      sprintf(str, "score=%.2f", splineScore);
      // 	    mcvDrawText(subimageClr, str, cv::Point(30, 30),
      // 			.25f, CV_RGB(1,1,1));

      //draw spline
      if (spline.degree > 0)
        mcvDrawSpline(subimageClr, spline, CV_RGB(1,0,0), 1);
      SHOW_IMAGE(subimageClr, title, 10);
      //clear
      delete subimageClr;
    }//#endif

    //clear
    delete subimage;
  }//for


  //put splines back in descending order of scores
  splines.clear();
  splineScores.clear();
  vector<Spline>::iterator li;
  vector<float>::iterator si;
  for (int i=0; i<(int)newSplines.size(); i++)
  {
    //get its position
    for (li=splines.begin(), si=splineScores.begin();
    si!=splineScores.end() && newSplineScores[i]<=*si;
    si++, li++);
    splines.insert(li, newSplines[i]);
    splineScores.insert(si, newSplineScores[i]);
  }

  //group the splines
  mcvGroupSplines(splines, splineScores);

  //draw splines
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

    //get string
    char title[256]; //str[256],
    switch (lineType)
    {
      case LINE_HORIZONTAL:
        sprintf(title, "Splines H");
        break;
      case LINE_VERTICAL:
        sprintf(title, "Splines V");
        break;
    }
    //convert image to rgb
    cv::Mat* im2  = new cv::Mat();
    *im2  = im->clone();
    mcvScaleMat(im2, im2);
    cv::Mat *imClr = new cv::Mat(im->rows, im->cols, CV_32FC3);
    cv::cvtColor(*im2, *imClr, cv::COLOR_GRAY2RGB);
    delete im2;

    //draw spline
    for (unsigned int j=0; j<splines.size(); j++)
      mcvDrawSpline(imClr, splines[j], CV_RGB(0,1,0), 1);
    SHOW_IMAGE(imClr, title, 10);
    //clear
    delete imClr;
  }//#endif


  //clean
  boxes.clear();
  newSplines.clear();
  newSplineScores.clear();
  delete image;
}

/** This functions implements RANSAC algorithm for spline fitting
 *  given an image
 *
 *
 * \param image input image
 * \param numSamples number of samples to take every iteration
 * \param numIterations number of iterations to run
 * \param threshold threshold to use to assess a point as a good fit to a line
 * \param numGoodFit number of points close enough to say there's a good fit
 * \param splineDegree the spline degree to fit
 * \param h the resolution to use for splines
 * \param spline the fitted line
 * \param splineScore the score of the line detected
 * \param splineScoreJitter Number of pixels to go around the spline to compute
 *          score
 * \param splineScoreLengthRatio Ratio of spline length to use
 * \param splineScoreAngleRatio Ratio of spline angle to use
 * \param splineScoreStep Step to use for spline score computation
 * \param prevSplines the splines from the previous frame, to use as initial
 *          seeds
 *   pass nullptr to ignore this input
 *
 */
void mcvFitRansacSpline(const cv::Mat *image, int numSamples, int numIterations,
                        float threshold, float scoreThreshold, int numGoodFit,
                        int splineDegree, float h, Spline *spline,
                        float *splineScore, int splineScoreJitter,
                        float splineScoreLengthRatio,
                        float splineScoreAngleRatio, float splineScoreStep,
                        vector<Spline> *prevSplines)
{
  //get the points with non-zero pixels
  cv::Mat *points = mcvGetNonZeroPoints(image, true);
  if (points==0 || points->cols < numSamples)
  {
    if (spline) spline->degree = -1;
    delete points;
    return;
  }
  //     fprintf(stderr, "num points=%d", points->cols);
  //subtract half
  //#warning "check adding half to points"
  cv::Mat p;
  p = points->rowRange(0, 2);
  p = p + 0.5;

  //normalize pixels values to get weights of each non-zero point
  //get third row of points containing the pixel values
  cv::Mat w;
  w = points->row(2);
  //normalize it
  cv::Mat *weights  = new cv::Mat();
*weights  = w.clone();
  cv::normalize(*weights, *weights, 1, 0, cv::NORM_L1);
  //get cumulative    sum
  mcvCumSum(weights, weights);

  //random number generator
  cv::RNG rng = cv::RNG(0xffffffff);
  //matrix to hold random sample
  cv::Mat *randInd = new cv::Mat(numSamples, 1, CV_32SC1);
  cv::Mat *samplePoints = new cv::Mat(numSamples, 2, CV_32FC1);
  //flag for points currently included in the set
  cv::Mat *pointIn = new cv::Mat(1, points->cols, CV_8SC1);
  //returned splines
  Spline curSpline, bestSpline;
  bestSpline.degree = 0;//initialize
  float bestScore=0; //, bestDist=1e5;

  //iterator for previous splines
  vector<Spline>::iterator prevSpline;
  bool randSpline = prevSplines==nullptr || prevSplines->size()==0;
  if (!randSpline) prevSpline = prevSplines->begin();

  //fprintf(stderr, "spline degree=%d\n", prevSpline->degree);

  //outer loop
  for (int i=0; i<numIterations; i++)
  {
    //check if to get a random spline or one of previous splines
    if (!randSpline)
    {
      //get spline
      curSpline = *prevSpline;
      //increment
      randSpline = ++prevSpline == prevSplines->end();
    } // if
    //get random spline
    else
    {
      //set flag to zero
      pointIn->setTo(0);
      //get random sample from the points
      //cv::randArr(&rng, randInd, CV_RAND_UNI, 0, points->cols);
      mcvSampleWeighted(weights, numSamples, randInd, &rng);
      // 	    SHOW_MAT(randInd, "randInd");
      for (int j=0; j<randInd->rows; j++) //numSamples
      {
        //flag it as included
        int p = randInd->at<int>(j, 0);
        pointIn->at<char>(0, p) = 1;
        //put point
        samplePoints->at<float>(j, 0) =
        points->at<float>(0, p);
        samplePoints->at<float>(j, 1) =
        points->at<float>(1, p);
      }

      //fit the spline
      curSpline = mcvFitBezierSpline(samplePoints, splineDegree);
      // 	    SHOW_MAT(samplePoints, "Sampled points");
    } // else


    //get score
    //float lengthRatio = 0.5; //.8
    //float angleRatio = 0.8; //.4
    //vector<int>jitter = mcvGetJitterVector(splineScoreJitter); //2);
    float score = mcvGetSplineScore(image, curSpline, splineScoreStep,//.05,//h,
                                    splineScoreJitter, //jitter,
                                    splineScoreLengthRatio,
                                    splineScoreAngleRatio);

    //jitter.clear();

    //check if better than best score so far
    //printf("Score=%.2f & scoreThreshold=%.2f\n", score, scoreThreshold);
    if (score>bestScore && score >= scoreThreshold)
    {
      //put it
      bestScore = score;
      bestSpline = curSpline;
    }

    //show image for debugging
    if(0) { //DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

      //get string
      char str[256];
      sprintf(str, "Spline Fit: score=%f, best=%f", score, bestScore);
      //fprintf(stderr, str);
      // 	    SHOW_SPLINE(curSpline, "curSpline:");


      //convert image to rgb
      cv::Mat *imageClr = new cv::Mat(image->rows, image->cols,
                                    CV_32FC3);
      cv::Mat *im  = new cv::Mat();
      *im  = image->clone();
      mcvScaleMat(image, im);
      cv::cvtColor(*im, *imageClr, cv::COLOR_GRAY2RGB);
	    //draw spline
	    //previous splines
 	    for (unsigned int k=0; prevSplines && k<prevSplines->size(); ++k)
        mcvDrawSpline(imageClr, (*prevSplines)[k], CV_RGB(0,1,0), 1);
	    if(curSpline.degree>0)
        mcvDrawSpline(imageClr, curSpline, CV_RGB(1,0,0), 1);
	    if(bestSpline.degree>0)
        mcvDrawSpline(imageClr, bestSpline, CV_RGB(0,0,1), 1);

	    //put text
	    sprintf(str, "score=%.2f bestScre=%.2f", score, bestScore);
	    cv::putText(*imageClr, std::string(str), cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.25f, CV_RGB(1,1,1));

	    sprintf(str, "Spline Fit");
	    SHOW_IMAGE(imageClr, str, 10);
	    //clear
	    delete imageClr;
	    delete im;
    }//#endif
  } //for

  //return
  if (spline)
    *spline = bestSpline;
  if (splineScore)
    *splineScore = bestScore;


  //clear
  delete points;
  delete samplePoints;
  delete randInd;
  delete pointIn;
  delete weights;
}

/** This function draws a spline onto the passed image
 *
 * \param image the input iamge
 * \param spline input spline
 * \param spline color
 *
 */
void mcvDrawSpline(cv::Mat *image, Spline spline, cv::Scalar color, int width)
{
  //get spline pixels
  cv::Mat *pixels = mcvGetBezierSplinePixels(spline, .05,
                                           cv::Size(image->cols, image->rows),
                                           false);
  //if no pixels
  if (!pixels)
    return;

  //draw pixels in image with that color
  for (int i=0; i<pixels->rows-1; i++)
    // 	cv::set2D(image,
    // 		(int)pixels->at<float>(i, 1),
    // 		(int)pixels->at<float>(i, 0),
    // 		color);
    cv::line(*image,
             cv::Point(pixels->at<int32_t>(i, 0),
                       pixels->at<int32_t>(i, 1)),
             cv::Point(pixels->at<int32_t>(i+1, 0),
                       pixels->at<int32_t>(i+1, 1)),
             color, width, cv::LINE_AA);

  //put the control points with circles
  for (int i=0; i<spline.degree+1; i++)
    cv::circle(*image, cv::Point(spline.points[i]), 3, color, -1);

  //release
  delete pixels;
}


/** This function draws a rectangle onto the passed image
 *
 * \param image the input iamge
 * \param rect the input rectangle
 * \param color the rectangle color
 * \param cols the rectangle cols
 *
 */
void mcvDrawRectangle (cv::Mat *image, cv::Rect rect, cv::Scalar color, int width)
{
  //draw the rectangle
  cv::rectangle(*image, cv::Point(rect.x, rect.y),
              cv::Point(rect.x + rect.width-1, rect.y + rect.height-1),
              color, width);

}

/** This function draws a spline onto the passed image
 *
 * \param image the input iamge
 * \param str the string to put
 * \param point the point where to put the text
 * \param size the font size
 * \param color the font color
 *
 */
void mcvDrawText(cv::Mat *image, char* str, cv::Point point,
		 float size, cv::Scalar color)
{

  cv::putText(*image, std::string(str), point, cv::FONT_HERSHEY_SIMPLEX, size, color);

}

/** This function converts lines from IPM image coordinates back to image
 * coordinates
 *
 * \param lines the input lines
 * \param ipmInfo the IPM info
 * \param cameraInfo the camera info
 * \param imSize the output image size (for clipping)
 *
 */
void mcvLinesImIPM2Im(vector<Line> &lines, IPMInfo &ipmInfo,
                      CameraInfo &cameraInfo, cv::Size imSize)
{
  //check if returned anything
  if (lines.size()!=0)
  {
    //convert the line into world frame
    for (unsigned int i=0; i<lines.size(); i++)
    {
      Line *line;
      line = & (lines[i]);

      mcvPointImIPM2World(&(line->startPoint), &ipmInfo);
      mcvPointImIPM2World(&(line->endPoint), &ipmInfo);
    }

    //convert them from world frame into camera frame
    //
    //put a dummy line at the beginning till we check that cv::div bug
    Line dummy = {{1.,1.},{2.,2.}};
    lines.insert(lines.begin(), dummy);
    //convert to mat and get in image coordinates
    cv::Mat *mat = new cv::Mat(2, 2*lines.size(), FLOAT_MAT_TYPE);
    mcvLines2Mat(&lines, mat);
    lines.clear();
    mcvTransformGround2Image(mat, mat, &cameraInfo);
    //get back to vector
    mcvMat2Lines(mat, &lines);
    //remove the dummy line at the beginning
    lines.erase(lines.begin());
    //clear
    delete mat;

    //clip the lines found and get their extent
    for (unsigned int i=0; i<lines.size(); i++)
    {
      //clip
      mcvIntersectLineWithBB(&(lines[i]), imSize, &(lines[i]));
      //get the line extent
      //mcvGetLineExtent(inImage, (*stopLines)[i], (*stopLines)[i]);
    }
  }
}


/** This function converts splines from IPM image coordinates back to image
 * coordinates
 *
 * \param splines the input splines
 * \param ipmInfo the IPM info
 * \param cameraInfo the camera info
 * \param imSize the output image size (for clipping)
 *
 */
void mcvSplinesImIPM2Im(vector<Spline> &splines, IPMInfo &ipmInfo,
                        CameraInfo &cameraInfo, cv::Size imSize)
{
  //loop on splines and convert
  for (int i=0; i<(int)splines.size(); i++)
  {
    //get points for this spline in IPM image
    cv::Mat *points = mcvEvalBezierSpline(splines[i], .1);

    //transform these points to image coordinates
    cv::Mat *points2 = new cv::Mat(2, points->rows, CV_32FC1);
    *points2 = points->t();
    //mcvPointImIPM2World(cv::Mat *mat, const IPMInfo *ipmInfo);
    //mcvTransformGround2Image(points2, points2, &cameraInfo);
    mcvTransformImIPM2Im(points2, points2, &ipmInfo, &cameraInfo);
    *points = points2->t();
    delete points2;

    //refit the points into a spline in output image
    splines[i] = mcvFitBezierSpline(points, splines[i].degree);
  }
}


/** This function samples uniformly with weights
 *
 * \param cumSum cumulative sum for normalized weights for the differnet
 *          samples (last is 1)
 * \param numSamples the number of samples
 * \param randInd a 1XnumSamples of int containing the indices
 * \param rng a pointer to a random number generator
 *
 */
void mcvSampleWeighted(const cv::Mat *cumSum, int numSamples, cv::Mat *randInd,
                       cv::RNG *rng)
{
//     //get cumulative sum of the weights
//     //OPTIMIZE:should pass it later instead of recomputing it
//     cv::Mat *cumSum  = new cv::Mat();
//     *cumSum  = weights->clone();
//     for (int i=1; i<weights->cols; i++)
// 	cumSum->at<float>(0, i) += cumSum->at<float>(0, i-1);

  //check if numSamples is equal or more
  int i=0;
  if (numSamples >= cumSum->cols)
  {
    for (; i<numSamples; i++)
      randInd->at<int>(i, 0) = i;
  }
  else
  {
    //loop
    while(i<numSamples)
    {
      //get random number
      double r = rng->uniform(0., 1.);//cv::randReal(rng);

      //get the index from cumSum
      int j;
      for (j=0; j<cumSum->cols && r>cumSum->at<float>(0, j); j++);

      //make sure this index wasnt chosen before
      volatile bool put = true;
      for (int k=0; k<i; k++)
        if (randInd->at<int>(k, 0) == j)
          //put it
          put = false;

      if (put)
      {
        //put it in array
        randInd->at<int>(i, 0) = j;
        //inc
        i++;
      }
    } //while
  } //if
}

/** This function computes the cumulative sum for a vector
 *
 * \param inMat input matrix
 * \param outMat output matrix
 *
 */
void mcvCumSum(const cv::Mat *inMat, cv::Mat *outMat)
{

#define MCV_CUM_SUM(type) 				\
    /*row vector*/ 					\
    if(inMat->rows == 1) 				\
	for (int i=1; i<outMat->cols; i++) 		\
	    outMat->at<type>(0, i) += 	\
		outMat->at<type>(0, i-1); 	\
    /*column vector*/					\
    else						\
	for (int i=1; i<outMat->rows; i++) 		\
	    outMat->at<type>(i, 0) += 	\
		outMat->at<type>(i-1, 0);

  //copy to output if not equal
  if(inMat != outMat)
    inMat->copyTo(*outMat);

  //check type
  if (CV_MAT_TYPE(inMat->type())==CV_32FC1)
  {
    MCV_CUM_SUM(float)
  }
  else if (CV_MAT_TYPE(inMat->type())==CV_32SC1)
  {
    MCV_CUM_SUM(int)
  }
  else
  {
    cerr << "Unsupported type in mcvCumSum\n";
    exit(1);
  }
}


/** This functions gives better localization of points along lines
 *
 * \param im the input image
 * \param inPoints the input points Nx2 matrix of points
 * \param outPoints the output points Nx2 matrix of points
 * \param numLinePixels Number of pixels to go in normal direction for
 *          localization
 * \param angleThreshold Angle threshold used for localization
 *          (cosine, 1: most restrictive, 0: most liberal)
 *
 */
void mcvLocalizePoints(const cv::Mat *im, const cv::Mat *inPoints,
                       cv::Mat *outPoints, int numLinePixels,
                       float angleThreshold)
{
  //size of inPoints must be at least 3
  if(inPoints->rows<3)
  {
    inPoints->copyTo(*outPoints);
    return;
  }

  //number of pixels in line around   each point
  //int numLinePixels = 20;
  //tangent and normal
  cv::Point2f tangent, normal;// peakTangent;

  //threshold for accepting new point (if not changing orientation too much)
  //float angleThreshold = .7;//.96;
  cv::Mat *imageClr;
  char str[256];
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    //get string
    sprintf(str, "Localize Points");

    //convert image to rgb
    imageClr = new cv::Mat(im->rows, im->cols, CV_32FC3);
    cv::cvtColor(*im, *imageClr, cv::COLOR_GRAY2RGB);
  }//#endif


  //loop on the points
  for (int i=0; i<inPoints->rows; i++)
  {

    //get tangent to current point
    if (i==0)
    {
      //first point, then tangent is vector to next point
      tangent = cv::Point(inPoints->at<float>(1, 0) -
      inPoints->at<float>(0, 0),
                             inPoints->at<float>(1, 1) -
                             inPoints->at<float>(0, 1));
    }
    else if (i==1)
      tangent = cv::Point(inPoints->at<float>(1, 0) -
                             outPoints->at<float>(0, 0),
                             inPoints->at<float>(1, 1) -
                             outPoints->at<float>(0, 1));

    else //if (i==inPoints->rows-1)
    {
      //last pointm then vector from previous two point
      tangent = cv::Point(outPoints->at<float>(i-1, 0) -
                             outPoints->at<float>(i-2, 0),
                             outPoints->at<float>(i-1, 1) -
                             outPoints->at<float>(i-2, 1));
      // 	    tangent = cv::Point(inPoints->at<float>(i, 0) -
      // 				   outPoints->at<float>(i-1, 0),
      // 				   inPoints->at<float>(i, 1) -
      // 				   outPoints->at<float>(i-1, 1));
    }
// 	else
// 	{
// 	    //general point, then take next - previous
// 	    tangent = cv::Point(inPoints->at<float>(i, 0) - //i+1
// 				   outPoints->at<float>(i-1, 0),
// 				   inPoints->at<float>(i, 1) - //i+1
// 				   outPoints->at<float>(i-1, 1));
// 	}

    //get normal
    float ss = 1./cv::sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
    tangent.x *= ss; tangent.y *= ss;
    normal.x = tangent.y; normal.y = -tangent.x;

    //get points in normal direction
    Line line;
    line.startPoint = cv::Point(inPoints->at<float>(i, 0) +
                numLinePixels * normal.x,
                inPoints->at<float>(i, 1) +
                numLinePixels * normal.y);
    line.endPoint = cv::Point(inPoints->at<float>(i, 0) -
              numLinePixels * normal.x,
              inPoints->at<float>(i, 1) -
              numLinePixels * normal.y);


    cv::Point2f prevPoint = {0., 0.};
    if (i>0)
      prevPoint = cv::Point(outPoints->at<float>(i-1, 0),
                               outPoints->at<float>(i-1, 1));

    //get line peak i.e. point in middle of bright line on dark background
    cv::Point2f peak;
  // 	float val = mcvGetLinePeak(im, line, peak);
    //get line peak
    vector<cv::Point2f> peaks;
    vector<float> peakVals;
    float val = mcvGetLinePeak(im, line, peaks, peakVals);

    //choose the best peak
    // 	int index = mcvChooseBestPeak(peaks, peakVals, peak, val,
    // 				      0, tangent,
    // 				      prevPoint, angleThreshold);
    peak = peaks.front();
    val = peakVals.front();
    //clear
    peaks.clear();
    peakVals.clear();

    //check new peak
    if (mcvIsPointInside(line.startPoint, cv::Size(im->cols, im->rows)) &&
        mcvIsPointInside(line.endPoint, cv::Size(im->cols, im->rows)) &&
        (//!i ||
        (i>0 &&
          mcvIsValidPeak(peak, tangent, prevPoint,
            angleThreshold))) )
    {
      //put new peak
      outPoints->at<float>(i, 0) = peak.x;
      outPoints->at<float>(i, 1) = peak.y;
    }
    else
    {
      //keep original point
      outPoints->at<float>(i, 0) = inPoints->at<float>(i, 0);
      outPoints->at<float>(i, 1) = inPoints->at<float>(i, 1);
    }

    //debugging
    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

      fprintf(stderr, "Localize val=%.3f\n", val);

      //draw original point, localized point, and line endpoints
      cv::line(*imageClr, cv::Point(line.startPoint),
            cv::Point(line.endPoint), CV_RGB(0, 0, 1));
      //output points
      cv::circle(*imageClr, cv::Point((int)outPoints->at<float>(i, 0),
                                (int)outPoints->at<float>(i, 1)),
              1, CV_RGB(0, 1, 0), -1);
      //input points
      cv::circle(*imageClr, cv::Point((int)(line.startPoint.x+line.endPoint.x)/2,
                                (int)(line.startPoint.y+line.endPoint.y)/2),
              1, CV_RGB(1, 0, 0), -1);
      //show image
      SHOW_IMAGE(imageClr, str, 10);
    }//#endif
  } // for i

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    SHOW_IMAGE(imageClr, str, 10);
    //clear
    delete imageClr;
  }//#endif
}


/** This functions checks the peak point if much change in orientation
 *
 * \param peak the input peak point
 * \param tangent the tangent line along which the peak was found normal to
 *          (normalized)
 * \param prevPoint the previous point along the tangent
 * \param angleThreshold the angle threshold to consider for valid peaks
 * \return true if useful peak, zero otherwise
 *
 */
bool mcvIsValidPeak(const cv::Point2f &peak, const cv::Point2f &tangent,
                    const cv::Point2f &prevPoint, float angleThreshold)
{
  //compute the tangent line for the peak
  cv::Point2f peakTangent;
  peakTangent.x = peak.x - prevPoint.x;
  peakTangent.y = peak.y - prevPoint.y;

  //normalize new tangent
  float ss = 1./cv::sqrt(peakTangent.x * peakTangent.x + peakTangent.y * peakTangent.y);
  peakTangent.x *= ss; peakTangent.y *= ss;

  //check angle between newTangent and tangent, and refuse peak if too far
  float angle = fabs(peakTangent.x*tangent.x + peakTangent.y*tangent.y);
  if (DEBUG_LINES)
    fprintf(stderr, "angle=%f\n", angle);
  //return
  return (angle >= angleThreshold) ? true : false;

}


/** This functions chooses the best peak that minimizes deviation
 * from the tangent direction given
 *
 * \param peaks the peaks found
 * \param peakVals the values for the peaks
 * \param peak the returned peak
 * \param peakVal the peak value for chosen peak, -1 if nothing
 * \param contThreshold the threshold to get peak above
 * \param tangent the tangent line along which the peak was found normal to
 *          (normalized)
 * \param prevPoint the previous point along the tangent
 * \param angleThreshold the angle threshold to consider for valid peaks
 * \return index of peak chosen, -1 if nothing
 *
 */
int mcvChooseBestPeak(const vector<cv::Point2f> &peaks,
                      const vector<float> &peakVals,
                      cv::Point2f &peak, float &peakVal,
                      float contThreshold, const cv::Point2f &tangent,
                      const cv::Point2f &prevPoint, float angleThreshold)
{
  int index=-1;
  float maxAngle=0;
  peakVal = -1;

  //loop and check
  for (unsigned int i=0; i<peaks.size(); ++i)
  {
    cv::Point2f peak = peaks[i];

    //compute the tangent line for the peak and normalize
    cv::Point2f peakTangent;
    peakTangent.x = peak.x - prevPoint.x;
    peakTangent.y = peak.y - prevPoint.y;
    peakTangent = mcvNormalizeVector(peakTangent);

    //compute angle
    float angle = fabs(peakTangent.x*tangent.x + peakTangent.y*tangent.y);

    //check if min angle so far and above both thresholds
    if (DEBUG_LINES)
      fprintf(stderr, "peak#%d/%lu (%f, %f): angle=%f, maxAngle=%f\n",
              i, peaks.size(), peaks[i].x, peaks[i].y,
              angle, maxAngle);
    if (peakVals[i]>=contThreshold && angle>=angleThreshold &&
      angle>maxAngle)
    {
      //mark it as chosen
      maxAngle = angle;
      index = i;
    }
  } // for i

  //return
  if (index>=0)
  {
    peak = peaks[index];
    peakVal = peakVals[index];
  }

  if (DEBUG_LINES)
    fprintf(stderr, "Chosen peak is: (%f, %f)\n", peak.x, peak.y);

  return index;
}


/** This functions extends the given set of points in both directions to
 * extend curves and lines in the image
 *
 * \param im the input image
 * \param inPoints the input points Nx2 matrix of points
 * \param angleThreshold angle threshold used for extending
 * \param meanDirAngleThreshold angle threshold from mean direction
 * \param linePixelsTangent number of pixels to go in tangent direction
 * \param linePixelsNormal number of pixels to go in normal direction
 * \param contThreshold number of pixels to go in tangent direction
 * \param deviationThreshold Stop extending when number of deviating points
 *          exceeds this threshold
 * \param bbox a bounding box not to get points outside
 * \param smoothPeak whether to smooth for calculating peaks or not
 *
 */
cv::Mat*  mcvExtendPoints(const cv::Mat *im, const cv::Mat *inPoints,
                        float angleThreshold, float meanDirAngleThreshold,
                        int linePixelsTangent, int linePixelsNormal,
                        float contThreshold, int deviationThreshold,
                        cv::Rect bbox, bool smoothPeaks)
{
  //size of inPoints must be at least 3
  if(inPoints->rows<4)
  {
    cv::Mat* ret = new cv::Mat();
    *ret = inPoints->clone();
    return ret;
  }


  char str[256];
  cv::Mat *imageClr;
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    //get string
    sprintf(str, "Extend Points");

    //convert image to rgb
    imageClr = new cv::Mat(im->rows, im->cols, CV_32FC3);
    cv::Mat *im2  = new cv::Mat();
*im2  = im->clone();
    mcvScaleMat(im, im2);
    cv::cvtColor(*im2, *imageClr, cv::COLOR_GRAY2RGB);
    delete im2;

    //show original points
    for(int i=0; i<inPoints->rows; i++)
        //input points
        cv::circle(*imageClr, cv::Point((int)(inPoints->at<float>(i, 0)),
                                   (int)(inPoints->at<float>(i, 1))),
                 1, CV_RGB(0, 1, 1), -1);
    //show image
    SHOW_IMAGE(imageClr, str, 10);
  }//#endif

  //tangent and normal
  cv::Point2f tangent, curPoint, peak, nextPoint, meanDir;
  //prevTangent, pprevTangent,

  //number of pixels away to look for points
  //int linePixelsTangent = 5, linePixelsNormal = 20;
  bool cont = true;

  //threshold for stopping
  //float contThreshold = 0.1; //.1 for gaussian top   //0.01;

  //angle threshold (max orientation change allowed)
  //float angleThreshold = .7;//.5 //.8;//.5 //.866;
  //float meanDirAngleThreshold = .7;

  //threshold to stop when deviating from normal orientation
  //int deviationThreshold = 2;

  //first go in one direction: from first point backward
  vector<cv::Point2f> backPoints;
  int numBack = 0;
  int deviationCount = 0;
  vector<cv::Point2f> peaks;
  vector<float> peakVals;
  //get mean direction of points
  meanDir = mcvGetPointsMeanVector(inPoints, false);
  while(cont)
  {
    int outSize = (int)backPoints.size();
    //get tangent from previous point in input points if no output points yet
    if(outSize==0)
    {
	    curPoint = cv::Point(inPoints->at<float>(0, 0),
                              inPoints->at<float>(0, 1));
	    tangent = cv::Point(inPoints->at<float>(0, 0) -
                             inPoints->at<float>(1, 0),
                             inPoints->at<float>(0, 1) -
                             inPoints->at<float>(1, 1));
      // 	    prevTangent = cv::Point(inPoints->at<float>(1, 0) -
      // 				       inPoints->at<float>(2, 0),
      // 				       inPoints->at<float>(1, 1) -
      // 				       inPoints->at<float>(2, 1));
      // 	    prevTangent = mcvNormalizeVector(prevTangent);

      // 	    pprevTangent = cv::Point(inPoints->at<float>(2, 0) -
      // 				       inPoints->at<float>(3, 0),
      // 				       inPoints->at<float>(2, 1) -
      // 				       inPoints->at<float>(3, 1));
      // 	    pprevTangent = mcvNormalizeVector(pprevTangent);
    }
    //only one out point till now
    else
    {
      // 	    pprevTangent = prevTangent;
      // 	    prevTangent = tangent;
	    curPoint = backPoints[outSize-1];
	    if (outSize==1)
	    {
        tangent = cv::Point(backPoints[outSize-1].x -
                               inPoints->at<float>(0, 0),
                               backPoints[outSize-1].y -
                               inPoints->at<float>(0, 1));
	    }
	    //more than one
	    else
	    {
        tangent = cv::Point(backPoints[outSize-1].x -
                               backPoints[outSize-2].x,
                               backPoints[outSize-1].y -
                               backPoints[outSize-2].y);
	    }
    }

    //get the line normal to tangent (tangent is normalized in function)
    Line line;
    line = mcvGetExtendedNormalLine(curPoint, tangent, linePixelsTangent,
                                    linePixelsNormal, nextPoint);

    //check if still inside
    //if (mcvIsPointInside(nextPoint, cv::Size(im->cols-1, im->rows-1)))
    if (mcvIsPointInside(nextPoint, bbox))
    {
	    //clip line
	    mcvIntersectLineWithBB(&line, cv::Size(im->cols-1, im->rows-1), &line);

	    //get line peak
	    float val = mcvGetLinePeak(im, line, peaks, peakVals,
				       true, smoothPeaks);

	    //choose the best peak
	    //int index =
	    mcvChooseBestPeak(peaks, peakVals, peak, val, contThreshold, tangent,
                        curPoint, 0); //angleThreshold);
	    //clear
	    peaks.clear();
	    peakVals.clear();

	    //check the peak
      //    !mcvIsValidPeak(peak, prevTangent, curPoint, angleThreshold) ||
      //    !mcvIsValidPeak(peak, pprevTangent, curPoint, angleThreshold) ||
	    if (!mcvIsValidPeak(peak, tangent, curPoint, angleThreshold) ||
          !mcvIsValidPeak(peak, meanDir, curPoint, meanDirAngleThreshold))
	    {
        peak = nextPoint;
        deviationCount++;
	    }
	    else
        deviationCount = 0;

	    if (DEBUG_LINES){
        fprintf(stderr, "Extension back #%d val=%.3f\n", outSize, val);
        fprintf(stderr, "Deviation Count=%d\n", deviationCount);
	    }

 	    //check value
	    //check value
	    if(val<contThreshold || deviationCount>deviationThreshold)
	    {
        cont = false;
        //  		for(int k=0; k<deviationCount; k++)
        // 		    backPoints.pop_back();
        if (deviationCount)
          backPoints.erase(backPoints.end() - (deviationCount-1),
                           backPoints.end());
      }
      // 	    if (val<contThreshold)
      // 		cont = false;
      // 	    //if exceeded threshold, then remove the latest additions
      // 	    else if (deviationCount>deviationThreshold)
      // 	    {
      // 		cont = false;
      // 		backPoints.erase(backPoints.end() - deviationThreshold,
      // 				 backPoints.end());
      // 		//numBack -= deviationThreshold;
      // 	    }
	    else
      {
        //push back
        backPoints.push_back(peak);
        //numBack++;
	    }
    } // if mcvIsPointInside
    //line got outside, so stop
    else
	    cont = false;

    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    //draw original point, localized point, and line endpoints
    cv::line(*imageClr, cv::Point(line.startPoint),
           cv::Point(line.endPoint),
    CV_RGB(0, 0, 1));
    //output points
    cv::circle(*imageClr, cv::Point(peak), 1, CV_RGB(0, 1, 0), -1);
    //input points
    cv::circle(*imageClr, cv::Point(nextPoint), 1, CV_RGB(1, 0, 0), -1);
    //show image
    SHOW_IMAGE(imageClr, str, 10);
    }//#endif
  }  // while cont

  //do the same for the opposite direction
  cont = true;
  vector<cv::Point2f> frontPoints;
  int numFront = 0;
  deviationCount = 0;
  //get mean direction in forward direction
  meanDir = mcvGetPointsMeanVector(inPoints, true);
  while(cont)
  {
    int outSize = (int)frontPoints.size();
    //get tangent from previous point in input points if no output points yet
    if(outSize==0)
    {
	    curPoint = cv::Point(inPoints->at<float>(inPoints->rows-1, 0),
                              inPoints->at<float>(inPoints->rows-1, 1));
	    tangent = cv::Point(inPoints->at<float>(inPoints->rows-1, 0) -
                             inPoints->at<float>(inPoints->rows-2, 0),
                             inPoints->at<float>(inPoints->rows-1, 1) -
                             inPoints->at<float>(inPoints->rows-2, 1));

      // 	    prevTangent = cv::Point(inPoints->at<float>(inPoints->rows-2, 0) -
      // 				       inPoints->at<float>(inPoints->rows-3, 0),
      // 				       inPoints->at<float>(inPoints->rows-2, 1) -
      // 				       inPoints->at<float>(inPoints->rows-3, 1));
      // 	    prevTangent = mcvNormalizeVector(prevTangent);

      // 	    pprevTangent = cv::Point(inPoints->at<float>(inPoints->rows-3, 0) -
      // 				       inPoints->at<float>(inPoints->rows-4, 0),
      // 				       inPoints->at<float>(inPoints->rows-3, 1) -
      // 				       inPoints->at<float>(inPoints->rows-4, 1));
      // 	    pprevTangent = mcvNormalizeVector(pprevTangent);
    }
    //only one out point till now
    else
    {
      // 	    pprevTangent = prevTangent;
      // 	    prevTangent = tangent;
	    curPoint = frontPoints[outSize-1];
	    if (outSize==1)
	    {
        tangent = cv::Point(frontPoints[outSize-1].x -
                               inPoints->at<float>(inPoints->rows-1, 0),
                               frontPoints[outSize-1].y -
                               inPoints->at<float>(inPoints->rows-1, 1));
	    }
	    //more than one
	    else
	    {
        tangent = cv::Point(frontPoints[outSize-1].x -
                               frontPoints[outSize-2].x,
                               frontPoints[outSize-1].y -
                               frontPoints[outSize-2].y);
	    }
    }

    Line line;
    line = mcvGetExtendedNormalLine(curPoint, tangent, linePixelsTangent,
                                    linePixelsNormal, nextPoint);

    //check if still inside
  // 	if (mcvIsPointInside(nextPoint, cv::Size(im->cols-1, im->rows-1)))
    if (mcvIsPointInside(nextPoint, bbox))
    {
      //clip line
      mcvIntersectLineWithBB(&line, cv::Size(im->cols-1, im->rows-1), &line);

	    //get line peak
      // 	    float val = mcvGetLinePeak(im, line, peak);
	    float val = mcvGetLinePeak(im, line, peaks, peakVals, true, smoothPeaks);

	    //choose the best peak
	    //int index =
	    mcvChooseBestPeak(peaks, peakVals, peak, val, contThreshold, tangent,
			      curPoint, 0); //angleThreshold);

	    //clear
	    peaks.clear();
	    peakVals.clear();

	    //check the peak
//         !mcvIsValidPeak(peak, prevTangent, curPoint, angleThreshold) ||
//         !mcvIsValidPeak(peak, pprevTangent, curPoint, angleThreshold) ||
	    if(!mcvIsValidPeak(peak, tangent, curPoint, angleThreshold) ||
	       !mcvIsValidPeak(peak, meanDir, curPoint, meanDirAngleThreshold))
	    {
        //put normal point
        peak = nextPoint;
        //increment deviation count
        deviationCount++;
      }
	    else
        deviationCount = 0;

	    if (DEBUG_LINES){
        fprintf(stderr, "Extension front #%d val=%.3f\n", outSize, val);
        fprintf(stderr, "Deviation Count=%d\n", deviationCount);
	    }

	    //check value
	    if(val<contThreshold || deviationCount>deviationThreshold)
	    {
        cont = false;
        // 		for(int k=0; k<deviationCount; k++)
        // 		    frontPoints.pop_back();
        if (deviationCount)
          frontPoints.erase(frontPoints.end() - (deviationCount-1),
                            frontPoints.end());

	    }
	    //if exceeded threshold, then remove the latest additions
      // 	    else if (deviationCount>deviationThreshold)
      // 	    {
      // 		cont = false;
      // 		frontPoints.erase(frontPoints.end() - deviationThreshold,
      // 				  frontPoints.end());
      // 	    }
	    else
	    {
        //push back
        frontPoints.push_back(peak);
        //numFront++;
	    }
    }
    //line got outside, so stop
    else
	    cont = false;

    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
	    //draw original point, localized point, and line endpoints
	    cv::line(*imageClr, cv::Point(line.startPoint),
             cv::Point(line.endPoint), CV_RGB(0, 0, 1));
	    //output points
	    cv::circle(*imageClr, cv::Point(peak), 1, CV_RGB(0, 1, 0), -1);
	    //input points
	    cv::circle(*imageClr, cv::Point(nextPoint), 1, CV_RGB(1, 0, 0), -1);
	    //show image
	    SHOW_IMAGE(imageClr, str, 10);
    }//#endif
  }

  numFront = frontPoints.size();
  numBack = backPoints.size();
  //now that we have extended the points in both directions, we need to put them
  //back into the return matrix
  cv::Mat *extendedPoints = new cv::Mat(inPoints->rows + numBack + numFront,
                                      2, CV_32FC1);
  //first put back points in reverse order
  vector<cv::Point2f>::iterator pointi;
  int i = 0;
  for (i=0, pointi=backPoints.end(); i<numBack; i++)
  {
	  pointi--;
    extendedPoints->at<float>(i, 0) = (*pointi).x;
    extendedPoints->at<float>(i, 1) = (*pointi).y;
  }

  //then put the original points
//  i = numBack;
//  memcpy(extendedPoints->ptr(i, 0), ((float*)inPoints->data),
//         sizeof(float)*2*inPoints->rows);
  inPoints->copyTo(extendedPoints->rowRange(numBack, numBack + inPoints->rows));

  //then put the front points in normal order
  for (i = numBack+inPoints->rows, pointi=frontPoints.begin();
       i<extendedPoints->rows; pointi++, i++)
  {
    extendedPoints->at<float>(i, 0) = (*pointi).x;
    extendedPoints->at<float>(i, 1) = (*pointi).y;
  }

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    SHOW_IMAGE(imageClr, str, 10);
    //clear
    delete imageClr;
  }//#endif

  //clear
  backPoints.clear();
  frontPoints.clear();
  //return
  return extendedPoints;
}



/** This functions extends a point along the tangent and gets the normal line
 * at the new point
 *
 * \param curPoint the current point to extend
 * \param tangent the tangent at this point (not necessarily normalized)
 * \param linePixelsTangent the number of pixels to go in tangent direction
 * \param linePixelsNormal the number of pixels to go in normal direction
 * \param nextPoint the next point on the extended line
 * \return the normal line at new point
 */
Line mcvGetExtendedNormalLine(cv::Point2f &curPoint, cv::Point2f &tangent,
                              int linePixelsTangent, int linePixelsNormal,
                              cv::Point2f &nextPoint)
{
  //normalize tangent
  float ssq = 1./cv::sqrt(tangent.x*tangent.x + tangent.y*tangent.y);
  tangent.x *= ssq;
  tangent.y *= ssq;

  //get next point along the way
  nextPoint.x = curPoint.x + linePixelsTangent * tangent.x;
  nextPoint.y = curPoint.y + linePixelsTangent * tangent.y;

  //get normal direction
  cv::Point2f normal = cv::Point(-tangent.y, tangent.x);

  //get two points along the normal line
  Line line;
  line.startPoint = cv::Point(nextPoint.x + linePixelsNormal*normal.x,
                                 nextPoint.y + linePixelsNormal*normal.y);
  line.endPoint = cv::Point(nextPoint.x - linePixelsNormal*normal.x,
                               nextPoint.y - linePixelsNormal*normal.y);

  //return
  return line;
}



/** This functions gets the point on the input line that matches the
 * peak in the input image, where the peak is the middle of a bright
 * line on dark background in the image
 *
 * \param im the input image
 * \param line input line
 * \param peaks a vector of peak outputs
 * \param peakVals the values for each of these peaks
 * \param positivePeak whether we are looking for positive or
 *   negative peak(default true)
 * \param smoothPeaks whether to smooth pixels for calculating peaks
 *  or not
 *
 */
float mcvGetLinePeak(const cv::Mat *im, const Line &line,
                     vector<cv::Point2f> &peaks,
                     vector<float> &peakVals, bool positivePeak,
                     bool smoothPeaks)
{
  //create step to convolve with
  FLOAT_MAT_ELEM_TYPE stepp[] = //{-1, 0, 1};//{-1, -1, -1, 1, 1, 1};
  //  {-0.3000, -0.2, -0.1, 0, 0, 0.1, 0.2, 0.3, 0.4};
  // {-0.6, -0.4, -0.2, 0.2, 0.4, 0.6};
  //latest->    {-0.2, -0.4, -0.2, 0, 0, 0.2, 0.4, 0.2}; //{-.75, -.5, .5, .75};
    { 0.000003726653172, 0.000040065297393, 0.000335462627903,
      0.002187491118183, 0.011108996538242, 0.043936933623407,
      0.135335283236613, 0.324652467358350, 0.606530659712633,
      0.882496902584595, 1.000000000000000, 0.882496902584595,
      0.606530659712633, 0.324652467358350, 0.135335283236613,
      0.043936933623407, 0.011108996538242, 0.002187491118183,
      0.000335462627903, 0.000040065297393, 0.000003726653172};
  int stepsize = 21;
  cv::Mat step = cv::Mat(1, stepsize, CV_32FC1, stepp);

  //take negative to work for opposite polarity
  if (!positivePeak)
//    cv::scale(&step, &step, -1);
    step *= -1;
  //     //get the gaussian kernel to convolve with
  //     int cols = 5;
  //     float step = .5;
  //     cv::Mat *step = new cv::Mat(1, (int)(2*cols/step+1), CV_32FC1);
  //     int j; float i;
  //     for (i=-w, j=0; i<=w; i+=step, ++j)
  //         step->at<FLOAT_MAT_ELEM_TYPE>(0, j) =
  //            (float) exp(-(.5*i*i));


  //then get the pixel coordinates of the line in the image
  cv::Mat *pixels;
  pixels = mcvGetLinePixels(line);
  //get pixel values
  cv::Mat *pix = new cv::Mat(1, pixels->rows, CV_32FC1);
  for(int j=0; j<pixels->rows; j++)
  {
    pix->at<float>(0, j) =
        im->at<float>(MIN(MAX(pixels->at<int>(j, 1),0),im->rows-1), MIN(MAX(pixels->at<int>(j, 0),0),im->cols-1));
  }
  //clear
  delete pixels;

  //remove the mean
  cv::Scalar mean = cv::mean(*pix);
  *pix = *pix - mean;

  //convolve with step
  cv::Mat *pixStep = new cv::Mat(pix->rows, pix->cols, CV_32FC1);
  if (smoothPeaks)
    cv::filter2D(*pix, *pixStep, -1, step);
  else
    pix->copyTo(*pixStep);
  //     SHOW_MAT(pixStep, "pixStep");
  //     SHOW_MAT(pix, "pixels");

  //get local maxima
  double topVal;
  float top;
  vector<double> maxima;
  vector<int> maximaLoc;
  cv::Point2f peak;
  //get top point
  mcvGetVectorLocalMax(pixStep, maxima, maximaLoc);
  if(maximaLoc.size()>0)
  {
    //get max
    topVal = maxima.front();
    //loop and get peaks
    for (unsigned int i=0; i<maximaLoc.size(); ++i)
    {
	    //get subpixel accuracy
	    double val1 = pixStep->at<float>(0, MAX(maximaLoc[i]-1, 0));
	    double val3 = pixStep->at<float>(0, MIN(maximaLoc[i]+1,
                                                        pixStep->cols-1));
	    top = (float)mcvGetLocalMaxSubPixel(val1, maxima[i], val3);
	    top += maximaLoc[i];
	    //fprintf(stderr, "val1=%f, val2=%f, val3=%f\n", val1, maxima[i], val3);
	    //fprintf(stderr, "top=%d, subpixel=%f\n", maximaLoc[i], top);
	    top /= pix->cols;
	    //get loc
// 	    top = maximaLoc[i]/(float)(pix->cols);
	    //get peak
	    peak.x = line.startPoint.x*(1-top) + top * line.endPoint.x;
	    peak.y = line.startPoint.y*(1-top) + top * line.endPoint.y;
	    //push back
	    peaks.push_back(peak);
	    peakVals.push_back(maxima[i]);
    }
  } // if
  else
  {
    top = (pix->cols-2)/2./(pix->cols);
    topVal = -1;
    //push back
    peak.x = line.startPoint.x*(1-top) + top * line.endPoint.x;
    peak.y = line.startPoint.y*(1-top) + top * line.endPoint.y;
    //push back
    peaks.push_back(peak);
    peakVals.push_back(topVal);

  }
  maxima.clear();
  maximaLoc.clear();

//     //get new point
//     top /= (pix->cols);
//     peak.x = line.startPoint.x*(1-top) + top * line.endPoint.x;
//     peak.y = line.startPoint.y*(1-top) + top * line.endPoint.y;

  //clear
  delete pix;
  delete pixStep;

  //return mean of rising and falling val
  return  topVal;//MIN(risingVal, fallingVal);//no minus //(risingVal+fallingVal)/2;
}

/** This functions normalizes the given vector
 *
 * \param vector the input vector to normalize
 */
cv::Point2f mcvNormalizeVector(const cv::Point2f &v)
{
  //return vector
  cv::Point2f ret = v;

  //normalize vector
  float ssq = 1./cv::sqrt(ret.x*ret.x + ret.y*ret.y);
  ret.x *= ssq;
  ret.y *= ssq;

  //return
  return ret;
}


/** This functions normalizes the given vector
 *
 * \param vector the input vector to normalize
 */
cv::Point2f mcvNormalizeVector(const cv::Point &v)
{
  //return vector
  return mcvNormalizeVector(cv::Point(v));

}

/** This functions normalizes the given vector
 *
 * \param x the x component
 * \param y the y component
 */
cv::Point2f mcvNormalizeVector(float x, float y)
{
  //return vector
  return mcvNormalizeVector(cv::Point(x, y));
}


/** This functions adds two vectors and returns the result
 *
 * \param v1 the first vector
 * \param v2 the second vector
 * \return the sum
 */
cv::Point2f mcvAddVector(cv::Point2f v1, cv::Point2f v2)
{
  //get sum
  cv::Point2f sum = cv::Point(v1.x + v2.x, v1.y + v2.y);
  //return vector
  return sum;
}


/** This functions multiplies a vector by a scalar
 *
 * \param v the vector
 * \param s the scalar
 * \return the sum
 */
cv::Point2f mcvMultiplyVector(cv::Point2f v, float s)
{
  //get sum
  cv::Point2f prod;
  prod.x = v.x * s;
  prod.y = v.y * s;
  //return vector
  return prod;
}

/** This functions computes the score of the given spline from the
 * input image
 *
 * \param image the input image
 * \param spline the input spline
 * \param h spline resolution
 * \param jitterVal the amounts to count scores around the spline in x & y
 *          directions
 * \param lengthRatio the ratio to add to score from the spline length
 * \param angleRatio the ratio to add to score from spline curvature measure
 *
 * \return the score
 */
float mcvGetSplineScore(const cv::Mat* image, Spline& spline, float h,
                        int  jitterVal, float lengthRatio, float angleRatio)
{

  //check that all control points for spline are inside the image
  cv::Size size = cv::Size(image->cols-1, image->rows-1);
  //     SHOW_SPLINE(spline, "spline");
  for (int i=0; i<=spline.degree; i++)
    if (!mcvIsPointInside(spline.points[i], size))
	    return -100.f;

  //get the pixels that belong to the spline
  cv::Mat *pixels = mcvGetBezierSplinePixels(spline, h, size, false);
  if(!pixels)
    return -100.f;

  //get jitter vector
  vector<int>jitter = mcvGetJitterVector(jitterVal); //2);

  //compute its score by summing up pixel values belonging to it
  //int jitter[] = {0, 1, -1, 2, -2}, jitterLength = 5;
  //SHOW_MAT(pixels, "pixels");
  float score = 0.f;
  for (unsigned int j=0; j<jitter.size(); j++)
    for (int i=0; i<pixels->rows; i++)
    {
	    //jitter in x
      // 	    int k = MIN(MAX(pixels->at<int>(i, 0)+
      // 			    jitter[j], 0), image->cols-1);
      // 	    fprintf(stderr, "col=%d\n & row=%d", k, pixels->at<int>(i, 1));
	    score += image->at<float>(pixels->at<int>(i, 1),
                           MIN(MAX(pixels->at<int>(i, 0) +
                           jitter[j], 0), image->cols-1));
      // 	    //jitter the y
      // 	    score += cv::getReal2D(image,
      // 				 MIN(MAX(pixels->at<int>(i, 1)+
      // 					 jitter[j], 0), image->rows-1),
      // 				 pixels->at<int>(i, 0));
    } // for i

  //length: min 0 and max of 1 (normalized according to max of cols and rows
  //of image)
  //float length = ((float)pixels->rows) / MAX(image->cols, image->rows);
  float length = 0.f;
  //     for (int i=0; i<pixels->rows-1; i++)
  //     {
  // 	//get the vector between every two consecutive points
  // 	cv::Point2f v =
  // 	    mcvSubtractVector(cv::Point(pixels->at<int>(i+1, 0),
  // 					   pixels->at<int>(i+1, 1)),
  // 			      cv::Point(pixels->at<int>(i, 0),
  // 					   pixels->at<int>(i, 1)));
  // 	//add to length
  // 	length += cv::sqrt(v.x * v.x + v.y * v.y);
  //     }
  //get length between first and last control point
  cv::Point2f v = mcvSubtractVector(spline.points[0], spline.points[spline.degree]);
  length = cv::sqrt(v.x * v.x + v.y * v.y);
  //normalize
  length /= image->rows; //MAX(image->cols, image->rows);

  //add measure of spline straightness: angle between vectors from points 1&2 and
  //points 2&3: clsoer to 1 the better (straight)
  //add 1 to value to make it range from 0->2 (2 better)
  float angle = 0;
  for (int i=0; i<spline.degree-1; i++)
  {
    //get first vector
    cv::Point2f t1 = mcvNormalizeVector (mcvSubtractVector(spline.points[i+1],
                                                            spline.points[i]));

    //get second vector
    cv::Point2f t2 = mcvNormalizeVector (mcvSubtractVector(spline.points[i+2],
                                                            spline.points[i+1]));
    //get angle
    angle += t1.x*t2.x + t1.y*t2.y;
  } // for i
  //get mean
  angle /= (spline.degree-1);
  //normalize 0->1 (with 1 best)
  angle += 1;
  angle /= 2;

  //add ratio of spline length
  //score = .8*score + .4*pixels->rows; //.8 & .3

  // 	printf("angle = %f\n", angle);
  //score = 0.6*score + 0.4*(angle*score); //.6 .4

  //make 0 best and -1 worse
  angle -= 1;
  length -= 1;

  //     printf("angle=%.2f, length=%.2f, score=%.2f", angle, length, score);
      //add angle and length ratios
  //     angle = score*angle; //(1-angleRatio)*score + angleRatio*angle;  //angle*score
  //     length = lengthRatio*length*score;//(1-lengthRatio)*score + lengthRatio*length*score;
  //     score = angle + length;
  // score = score + angleRatio*angle*score + lengthRatio*length*score;

  if (DEBUG_LINES)
  	fprintf(stderr, "raw score=%.2f, angle=%.2f, length=%.2f, final=%.2f\n",
            score, angle, length, score *
            (1 + (angleRatio*angle + lengthRatio*length)/2));
  score = score * (1 + (angleRatio*angle + lengthRatio*length)/2);
  //     printf(" final score=%.2f\n", score);

  //clear pixels
  delete pixels;
  jitter.clear();

  //return
  return score;
}



/** This functions returns a vector of jitter from the input maxJitter value
 * This is used for computing spline scores for example, to get scores
 * around the rasterization of the spline
 *
 * \param maxJitter the max value to look around
 *
 * \return the required vector of jitter values
 */
vector<int> mcvGetJitterVector(int maxJitter)
{
  vector<int> jitter(2*maxJitter+1);

  //fill in
  jitter.push_back(0);
  for(int i=1; i<=maxJitter; ++i)
  {
    jitter.push_back(i);
    jitter.push_back(-i);
  }

  //return
  return jitter;
}


/** This functions gets the average direction of the set of points
 * by computing the mean vector between points
 *
 * \param points the input points [Nx2] matrix
 * \param forward go forward or backward in computation (default true)
 * \return the mean direction
 *
 */
cv::Point2f  mcvGetPointsMeanVector(const cv::Mat *points, bool forward)
{
  cv::Point2f mean, v;

  //init
  mean = cv::Point(0,0);

  //go forward direction
  for (int i=1; i<points->rows; ++i)
  {
    //get the vector joining the two points
    v = cv::Point(points->at<float>(i, 0) -
                     points->at<float>(i-1, 0),
                     points->at<float>(i, 1) -
                     points->at<float>(i-1, 1));
    //normalize
    v = mcvNormalizeVector(v);
    //get mean
    mean.x = (mean.x * (i-1) + v.x) / i;
    mean.y = (mean.y * (i-1) + v.y) / i;
    //renormlaize
    mean = mcvNormalizeVector(mean);
  }

  //check if to return forward or backward
  if (!forward)
    mean = cv::Point(-mean.x, -mean.y);

  return mean;
}


/** This functions checks if to merge two splines or not
 *
 * \param sp1 the first spline
 * \param sp2 the second spline
 * \param thetaThreshold Angle threshold for merging splines (radians)
 * \param rThreshold R threshold (distance from origin) for merginn splines
 * \param MeanhetaThreshold Mean angle threshold for merging splines (radians)
 * \param MeanRThreshold Mean r threshold (distance from origin) for merginn
 *          splines
 * \param centroidThreshold Distance threshold between spline cetroids for
 *          merging
 *
 * \return true if to merge, false otherwise
 *
 */
bool mcvCheckMergeSplines(const Spline& sp1, const Spline& sp2,
                          float thetaThreshold, float rThreshold,
                          float meanThetaThreshold, float meanRThreshold,
                          float centroidThreshold)
{
  //get spline stats
  cv::Point2f centroid1, centroid2;
  float theta1, theta2, length1, length2, r1, r2;
  float meanTheta1, meanTheta2, meanR1, meanR2;
  mcvGetSplineFeatures(sp1, &centroid1, &theta1, &r1,
                       &length1, &meanTheta1, &meanR1);
  mcvGetSplineFeatures(sp2, &centroid2, &theta2, &r2,
                       &length2, &meanTheta2, &meanR2);

  //threshold for difference in orientation
  //float thetaThreshold = 30*CV_PI/180.;
  //threshold for difference in centroid (squared)
  //float centroidThreshold = 50;
  //threshold for meanR
  //float rThreshold = 15;
  float meanThetaDist = fabs(meanTheta1 - meanTheta2);//fabs(theta1-theta2);
  float meanRDist = fabs(meanR1 - meanR2);
  float thetaDist = fabs(theta1 - theta2);//fabs(theta1-theta2);
  float rDist = fabs(r1 - r2);
  float centroidDist = fabs(mcvGetVectorNorm(mcvSubtractVector(centroid1, centroid2)));

  //correct theta diff
  //     thetaDist = thetaDist>CV_PI ? thetaDist-CV_PI : thetaDist;
  //     meanThetaDist = meanThetaDist>CV_PI ? meanThetaDist-CV_PI :
  // 	meanThetaDist;

  bool meanThetaOk = meanThetaDist <= meanThetaThreshold;
  bool meanROk = meanRDist <= meanRThreshold;
  bool thetaOk = thetaDist <= thetaThreshold;
  bool rOk = rDist <= rThreshold;
  bool centroidOk = centroidDist <= centroidThreshold;

  bool centroidNotOk = centroidDist >= 200;
  bool rNotOk = rDist >= 100;
  bool thetaNotOk = thetaDist >= .8;

  bool merge = false;
  //((thetaOk || meanThetaOk) && centroidOk) ||
  if ((thetaOk || meanThetaOk) &&	(rOk || meanROk || centroidOk) &&
    !rNotOk && !centroidNotOk && !thetaNotOk)
    merge = true;


  //debug
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

    //show splines
    // 	SHOW_SPLINE(sp1, "S1");
    // 	SHOW_SPLINE(sp2, "S2");
    fprintf(stderr, "%s: thetaDist=%.2f, meanThetaDist=%.2f, "
            "rDist=%.2f, meanRDist=%.2f, centroidDist=%.2f\n",
            merge? "Merged    " : "Not merged",
            thetaDist, meanThetaDist, rDist, meanRDist, centroidDist);

    fprintf(stderr, "\ttheta1=%.2f, theta2=%.2f\n", theta1, theta2);

    cv::Mat* im = new cv::Mat(480, 640, CV_8UC3);
    im->setTo(0.);
    //draw splines
    mcvDrawSpline(im, sp1, CV_RGB(255, 0, 0), 1);
    mcvDrawSpline(im, sp2, CV_RGB(0, 255, 0), 1);
    SHOW_IMAGE(im, "Check Merge Splines", 10);
    //clear
    delete im;

  }//#endif

  //return
  return merge;
}

/** This functions computes some features for a set of points
 *
 * \param points the input points
 * \param centroid the computed centroid of the points
 * \param theta the major orientation of the points (angle of line joining
 *   first and last points, angle as in Hough Transform lines)
 * \param r distance from origin for line from first to last point
 * \param length the length of the line from first to last point
 * \param meanTheta the average orientation of the points (by computing
 *   mean theta for line segments form the points)
 * \param meanR the average distance from the origin of the points (the
 *    same computations as for meanTheta)
 * \param curveness computes the angle between vectors of points,
 *    which gives an indication of the curveness of the spline
 *    -1-->1 with 1 best and -1 worst
 *
 */
void mcvGetPointsFeatures(const cv::Mat* points, cv::Point2f* centroid,
                          float* theta, float* r, float* length,
                          float* meanTheta, float* meanR, float* curveness)
{

  //get start and end point
  cv::Point2f start = cv::Point(points->at<float>(0, 0),
                                    points->at<float>(0, 1));
  cv::Point2f end = cv::Point(points->at<float>(points->rows-1, 0),
                                  points->at<float>(points->rows-1, 1));
  //compute centroid
  if (centroid)
  {
    //get sum of control points
    *centroid = cv::Point(0, 0);
    for (int i=0; i<=points->rows; ++i)
	    *centroid = mcvAddVector(*centroid,
                               cv::Point(points->at<float>(i, 0),
                                            points->at<float>(i, 1)));
    //take mean
    *centroid = cv::Point(centroid->x / (points->rows),
                             centroid->y / (points->rows));
    }

  //compute theta
  if (theta && r)
  {
    //get line from first and last control points
    Line line;
    line.startPoint = start;
    line.endPoint = end;
    //get theta
    //float r;
    mcvLineXY2RTheta(line, *r, *theta);
    //add pi if negative
    if (*theta<0)
      *theta  += CV_PI;
  }

  //mean theta
  if (meanTheta && meanR)
  {
    *meanTheta = 0;
    *meanR = 0;

    //loop and get theta
    for (int i=0; i<points->rows-1; i++)
    {
      //get the line
      Line line;
      line.startPoint = cv::Point(points->at<float>(i, 0),
                                      points->at<float>(i, 1));
      line.endPoint = cv::Point(points->at<float>(i+1, 0),
                                    points->at<float>(i+1, 1));
      //get theta and r
      float r, t;
      mcvLineXY2RTheta(line, r, t);
      //add pi if neg
      if (t<0) t += CV_PI;
      //add
      *meanTheta += t;
      *meanR += r;
    }

    //normalize
    *meanTheta /= points->rows - 1;
    *meanR /= points->rows - 1;
  }

  //compute length of spline: length of vector between first and last point
  if (length)
  {
    //get the vector
    cv::Point2f v = mcvSubtractVector(start, end);

    //compute length
    *length = cv::sqrt(v.x * v.x + v.y * v.y);
  }

  //compute curveness
  if (curveness)
  {
    *curveness = 0;
    if (points->rows>2)
    {
      //initialize
      cv::Point2f p0;
      cv::Point2f p1 = start;
      cv::Point2f p2 = cv::Point(points->at<float>(1, 0),
                                      points->at<float>(1, 1));

      for (int i=0; i<points->rows-2; i++)
      {
        //go next
        p0 = p1;
        p1 = p2;
        p2 = cv::Point(points->at<float>(i+2, 0),
                          points->at<float>(i+2, 1));
        //get first vector
        cv::Point2f t1 = mcvNormalizeVector(mcvSubtractVector(p1, p0));

        //get second vector
        cv::Point2f t2 = mcvNormalizeVector (mcvSubtractVector(p2, p1));
        //get angle
        *curveness += t1.x*t2.x + t1.y*t2.y;
      }
    //get mean
    *curveness /= points->rows-2;
    }
  }
}


/** This functions computes some features for the spline
 *
 * \param spline the input spline
 * \param centroid the computed centroid of spline (mean of control points)
 * \param theta the major orientation of the spline (angle of line joining
 *   first and last control points, angle as in Hough Transform lines)
 * \param r distance from origin for line from first to last control point
 * \param length the length of the line from first to last control point
 * \param meanTheta the average orientation of the spline (by computing
 *   mean theta for line segments form the spline)
 * \param meanR the average distance from the origin of the spline (the
 *    same computations as for meanTheta)
 * \param curveness computes the angle between vectors of control points,
 *    which gives an indication of the curveness of the spline
 *    -1-->1 with 1 best and -1 worst
 *
 */
void mcvGetSplineFeatures(const Spline& spline, cv::Point2f* centroid,
                          float* theta, float* r, float* length,
                          float* meanTheta, float* meanR, float* curveness)
{
  //compute centroid
  if (centroid)
  {
    //get sum of control points
    *centroid = cv::Point(0, 0);
    for (int i=0; i<=spline.degree; ++i)
      *centroid = mcvAddVector(*centroid, spline.points[i]);
    //take mean
    *centroid = cv::Point(centroid->x / (spline.degree+1),
                             centroid->y / (spline.degree+1));
  }

  //compute theta
  if (theta && r)
  {
    //get line from first and last control points
    Line line;
    line.startPoint = spline.points[0];
    line.endPoint = spline.points[spline.degree];
    //get theta
    //float r;
    mcvLineXY2RTheta(line, *r, *theta);
    //add pi if negative
    //if (*theta<0) *theta  += CV_PI;

    //compute theta as angle to the horizontal x-axis
    *theta = mcvGetLineAngle(line);
  }

  //mean theta
  if (meanTheta && meanR)
  {
    *meanTheta = 0;
    *meanR = 0;
    //get points on the spline
    cv::Mat* points = mcvEvalBezierSpline(spline, .1);
    //loop and get theta
    for (int i=0; i<points->rows-1; i++)
    {
	    //get the line
	    Line line;
	    line.startPoint = cv::Point(points->at<float>(i, 0),
                                     points->at<float>(i, 1));
	    line.endPoint = cv::Point(points->at<float>(i+1, 0),
                                   points->at<float>(i+1, 1));
	    //get theta and r
	    float r, t;
	    mcvLineXY2RTheta(line, r, t);
	    //add pi if neg
      //#warning "add pi to theta calculations for spline feature"
	    //if (t<0) t += CV_PI;
	    //add
	    t = mcvGetLineAngle(line);
	    *meanTheta += t;
	    *meanR += r;
    }

    //normalize
    *meanTheta /= points->rows - 1;
    *meanR /= points->rows - 1;

    //clear
    delete points;
  }

  //compute length of spline: length of vector between first and last point
  if (length)
  {
    //get the vector
    cv::Point2f v = cv::Point(spline.points[0].x -
                                  spline.points[spline.degree].x,
                                  spline.points[0].y -
                                  spline.points[spline.degree].y);
    //compute length
    *length = cv::sqrt(v.x * v.x + v.y * v.y);
  }

  //compute curveness
  if (curveness)
  {
    *curveness = 0;
    for (int i=0; i<spline.degree-1; i++)
    {
	    //get first vector
	    cv::Point2f t1 =
        mcvNormalizeVector(mcvSubtractVector(spline.points[i+1],
                                             spline.points[i]));

	    //get second vector
	    cv::Point2f t2 =
        mcvNormalizeVector(mcvSubtractVector(spline.points[i+2],
                                             spline.points[i+1]));
	    //get angle
	    *curveness += t1.x*t2.x + t1.y*t2.y;
    }
    //get mean
    *curveness /= (spline.degree-1);
  }
}


/** This functions computes difference between two vectors
 *
 * \param v1 first vector
 * \param v2 second vector
 * \return difference vector v1 - v2
 *
 */
cv::Point2f  mcvSubtractVector(const cv::Point2f& v1, const cv::Point2f& v2)
{
  return cv::Point(v1.x - v2.x, v1.y - v2.y);
}

/** This functions computes the vector norm
 *
 * \param v input vector
 * \return norm of the vector
 *
 */
float  mcvGetVectorNorm(const cv::Point2f& v)
{

  return cv::sqrt(v.x * v.x + v.y * v.y);
}


/** This functions checks if to merge two splines or not
 *
 * \param line1 the first line
 * \param line2 the second line
 * \param thetaThreshold Angle threshold for merging splines (radians)
 * \param rThreshold R threshold (distance from origin) for merginn splines
 *
 * \return true if to merge, false otherwise
 *
 */
bool mcvCheckMergeLines(const Line& line1, const Line& line2,
                        float thetaThreshold, float rThreshold)

{
  //convert lines to r theta
  float r1, r2, theta1, theta2;
  mcvLineXY2RTheta(line1, r1, theta1);
  mcvLineXY2RTheta(line2, r2, theta2);

  //adjust the thetas
  if (theta1<0) theta1 += CV_PI;
  if (theta2<0) theta2 += CV_PI;

  //check
  float rDist = fabs(r1 - r2);
  float thetaDist = fabs(theta1 - theta2);

  bool rOk = rDist <= rThreshold;
  bool thetaOk = thetaDist <= thetaThreshold;

  bool merge = false;
  if (rOk && thetaOk) merge = true;

  //debug
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

    //show splines
    fprintf(stderr, "%s: thetaDist=%.2f, rDist=%.2f\n",
            merge? "Merged" : "Not   ", thetaDist, rDist);


    cv::Mat* im = new cv::Mat(480, 640, CV_8UC3);
    im->setTo(0.);
    //draw lines
    mcvDrawLine(im, line1, CV_RGB(255, 0, 0), 1);
    mcvDrawLine(im, line2, CV_RGB(0, 255, 0), 1);
    SHOW_IMAGE(im, "Check Merge Lines", 10);
    //clear
    delete im;

  }//#endif

  return merge;
}


/** This functions converts a line to a spline
 *
 * \param line the line
 * \param degree the spline degree
 *
 * \return the returned spline
 *
 */
Spline mcvLineXY2Spline(const Line& line, int degree)
{
  //the spline to return
  Spline spline;
  spline.degree = degree;

  //put two end points
  spline.points[0] = line.startPoint;
  spline.points[degree] = line.endPoint;

  //get direction of line
  cv::Point2f dir = mcvSubtractVector(line.endPoint, line.startPoint);
  //get intermediate points
  for (int j=1; j<degree; ++j)
  {
    //get point
    cv::Point2f point;
    float t = j / (float)degree;
    point.x = line.startPoint.x + t * dir.x;
    point.y = line.startPoint.y + t * dir.y;
    //put it
    spline.points[j] = point;
  }

  //return
  return spline;
}


/** This functions gets the angle of the line with the horizontal
 *
 * \param line the line
 *
 * \return the required angle (radians)
 *
 */
float mcvGetLineAngle(const Line& line)
{
  //get vector from start to end
  cv::Point2f v = mcvNormalizeVector(mcvSubtractVector(line.startPoint,
                                                        line.endPoint));
  //angle which is acos(v.x) as we multiply by (1,0) which
  //cancels the second component
  return acos(fabs(v.x));
}

/** This functions classifies the passed points according to their
 * color to be either white, yellow, or neither
 *
 * \param im the input color image
 * \param points the array of points
 * \param window the window to use
 * \param numYellowMin min percentage of yellow points
 * \param rgMin
 * \param rgMax
 * \param gbMin
 * \param rbMin
 *
 * \return the line color
 *
 */
LineColor mcvGetPointsColor(const cv::Mat* im, const cv::Mat* points,
                            int window, float numYellowMin,
                            float rgMin, float rgMax,
                            float gbMin, float rbMin,
                            bool rbf, float rbfThreshold)
{

  //check if color image
  if (im->type() != CV_8UC3)
    return LINE_COLOR_WHITE;

  //    //half the cols of the window
  //     int window = 3;

  //     //thresholds
  //     float rgMin = 1, rgMax = 40;
  //     float gbMin = 15;
  //     float rbMin = 25;

  //     float numYellowMin = .5;

  //number of bins to use
  int numBins = 16;
  int histLen = 3*numBins + 3;
  float binWidth = 255. / numBins;

  //allocate the histogram
  cv::Mat* hist = new cv::Mat(1, histLen, CV_32FC1);

  //rbf centroids
  int rbfNumCentroids = 10; //10; //10;
  float rbfSigma = 400; //625; //400;
  float rbfCentroids[] = //{0.000000,0.000000,0.000000,0.000000,0.000000,0.178571,0.107143,0.142857,2.071429,19.928571,11.178571,3.250000,2.357143,2.071429,2.821429,4.892857,0.000000,0.000000,0.000000,0.000000,0.000000,0.178571,0.178571,0.178571,2.285714,21.642857,9.964286,3.107143,2.071429,2.428571,2.571429,4.392857,0.000000,0.000000,0.000000,0.000000,0.035714,0.214286,0.142857,0.285714,7.821429,21.821429,6.392857,2.785714,2.178571,2.107143,1.857143,3.357143,2.250729,7.100583,8.849854,0.000000,0.000000,0.000000,0.000000,0.000000,1.904762,14.857143,4.666667,4.285714,6.476190,12.857143,3.428571,0.238095,0.285714,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,5.142857,12.857143,3.571429,3.761905,11.428571,10.809524,0.857143,0.380952,0.190476,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000, -0.000000,6.809524,12.904762,4.380952,7.095238,15.190476,2.095238,0.380952,0.142857,0.000000,0.000000,0.000000,4.434402,8.619048,11.597668,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.652174,17.173913,9.869565,4.130435,6.652174,3.260870,2.434783,1.869565,0.130435,1.826087,0.000000,0.000000,0.000000,0.000000,0.000000,0.869565,10.652174,19.000000,4.956522,2.521739,2.565217,2.695652,2.739130,1.478261,0.173913,1.347826,0.000000,0.000000,0.000000,0.000000,0.000000,0.043478,7.565217,23.217391,6.739130,2.608696,2.652174,2.739130,2.521739,0.304348,0.043478,0.565217,13.452529,9.215617,15.786158,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.520000,0.240000,0.120000,1.720000,16.240000,8.720000,4.320000,3.960000,4.720000,8.440000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.440000,0.240000,0.200000,1.600000,17.520000,9.880000,6.120000,4.400000,4.400000,4.200000,0.000000,0.000000,0.000000,0.000000,0.000000,0.040000,0.520000,0.280000,0.240000,6.400000,24.200000,10.920000,3.040000,1.240000,0.960000,1.160000,6.879184,16.165714,22.386939,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.666667,6.083333,3.000000,0.916667,1.583333,1.250000,1.916667,2.250000,29.333333,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.250000,6.500000,2.166667,1.583333,1.250000,1.666667,1.583333,2.000000,30.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.166667,2.666667,6.000000,2.416667,1.416667,1.500000,1.250000,2.000000,3.000000,28.583333,3.119048,3.772109,5.993197,0.000000,0.000000,0.000000,0.000000,0.000000,6.611111,12.000000,13.000000,15.611111,1.777778,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,17.944444,29.500000,1.500000,0.055556,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.166667,32.444444,15.888889,0.500000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,34.328798,8.948979,41.862812,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.450000,24.150000,4.750000,3.300000,2.450000,2.850000,3.150000,2.050000,4.850000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.200000,23.850000,6.100000,2.450000,2.300000,3.200000,3.700000,1.450000,4.750000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,7.100000,21.900000,4.200000,2.350000,2.400000,3.450000,1.800000,1.500000,4.300000,2.412245,5.574490,6.370408,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.533333,2.200000,0.666667,0.533333,1.333333,11.733333,16.800000,7.866667,4.066667,3.266667,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.800000,1.800000,0.666667,0.666667,1.066667,12.266667,18.266667,8.533333,3.133333,1.800000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.333333,1.800000,0.533333,0.800000,2.466667,20.000000,14.266667,4.066667,2.066667,1.666667,3.639456,7.100680,8.832653,0.000000,0.000000,0.000000,0.000000,0.700000,3.100000,1.200000,1.100000,0.700000,2.900000,1.500000,2.300000,1.300000,1.600000,2.300000,30.300000,0.000000,0.000000,0.000000,0.000000,2.000000,2.700000,1.800000,0.900000,1.800000,3.300000,2.800000,2.500000,2.200000,2.900000,6.900000,19.200000,0.000000,0.000000,0.000000,0.000000,3.500000,4.600000,3.900000,21.900000,13.400000,1.300000,0.200000,0.200000,0.000000,0.000000,0.000000,0.000000,18.167347,85.375510,103.542857,0.000000,0.000000,0.000000,0.000000,0.250000,0.093750,0.218750,2.156250,5.718750,5.953125,3.515625,2.531250,3.359375,3.156250,4.859375,17.187500,0.000000,0.000000,0.000000,0.000000,0.281250,0.140625,0.296875,3.468750,7.625000,6.125000,5.046875,6.703125,8.609375,5.656250,3.125000,1.921875,0.000000,0.000000,0.000000,0.000000,0.343750,0.171875,0.843750,7.203125,13.390625,14.203125,8.984375,2.937500,0.750000,0.140625,0.031250,0.000000,25.309630,30.238520,55.548150};
    {0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.473684,9.236842,15.421053,3.921053,3.736842,3.500000,4.026316,3.105263,1.026316,3.552632,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.921053,9.157895,14.342105,5.026316,3.157895,3.710526,4.421053,3.078947,0.868421,3.315789,0.000000,0.000000,0.000000,0.000000,0.000000,0.052632,3.394737,12.473684,12.947368,4.078947,3.289474,4.500000,3.710526,1.131579,0.815789,2.605263,2.346402,7.178840,8.077336,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.043478,0.695652,7.478261,9.565217,4.826087,3.173913,4.000000,4.000000,5.652174,9.565217,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.043478,1.304348,12.130435,9.434783,6.000000,6.043478,7.956522,2.521739,2.739130,0.826087,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.043478,3.173913,21.434783,12.739130,9.782609,1.826087,0.000000,0.000000,0.000000,0.000000,20.372671,23.477374,43.850044,0.000000,0.000000,0.000000,0.000000,0.000000,1.046512,6.395349,1.906977,2.930233,14.697674,12.651163,3.418605,1.279070,1.139535,1.465116,2.069767,0.000000,0.000000,0.000000,0.000000,0.000000,1.906977,5.674419,1.651163,3.116279,18.000000,10.906977,2.023256,1.023256,1.255814,1.395349,2.046512,0.000000,0.000000,0.000000,0.000000,0.023256,3.441860,5.046512,1.930233,7.953488,19.883721,4.232558,1.534884,1.069767,1.279070,1.046512,1.558140,2.570005,7.212625,9.312767,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.615385,4.538462,8.538462,7.115385,4.807692,4.884615,5.230769,13.269231,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.576923,5.230769,10.423077,9.000000,7.000000,5.269231,5.730769,5.769231,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.923077,11.576923,15.576923,15.076923,4.000000,0.730769,0.115385,0.000000,10.828885,24.554160,35.383046,0.000000,0.000000,0.000000,0.000000,3.285714,4.857143,1.857143,1.857143,0.857143,3.142857,1.571429,3.142857,1.857143,2.000000,2.571429,22.000000,0.000000,0.000000,0.000000,0.000000,5.142857,4.428571,2.571429,1.428571,2.142857,3.571429,3.000000,3.285714,2.428571,3.714286,7.714286,9.571429,0.000000,0.000000,0.000000,0.000000,7.714286,6.714286,5.571429,21.142857,5.428571,1.857143,0.285714,0.285714,0.000000,0.000000,0.000000,0.000000,20.274053,66.647230,86.921283,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.323529,3.500000,6.264706,3.294118,2.352941,2.029412,2.676471,2.264706,4.705882,21.588235,0.000000,0.000000,0.000000,0.000000,0.000000,0.117647,0.470588,5.588235,6.794118,3.382353,4.205882,7.411765,9.911765,8.000000,2.294118,0.823529,0.000000,0.000000,0.000000,0.000000,0.000000,0.147059,1.352941,11.117647,10.941176,14.794118,8.441176,1.764706,0.441176,0.000000,0.000000,0.000000,31.429772,33.329532,64.759304,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.461538,5.615385,4.076923,1.153846,1.692308,1.230769,2.076923,2.153846,28.538462,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.076923,6.076923,3.384615,1.692308,1.307692,1.769231,1.615385,2.076923,29.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.153846,2.461538,5.769231,3.461538,1.538462,1.615385,1.384615,1.923077,3.000000,27.692308,3.149137,3.576138,5.896389,0.000000,0.000000,0.000000,0.000000,0.000000,0.750000,0.250000,0.250000,0.250000,2.250000,1.000000,0.500000,0.500000,0.750000,2.000000,40.500000,0.000000,0.000000,0.000000,0.000000,0.500000,0.250000,0.500000,0.250000,1.250000,2.500000,1.750000,1.000000,1.750000,1.500000,4.250000,33.500000,0.000000,0.000000,0.000000,0.000000,0.750000,1.250000,1.750000,21.250000,24.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,12.535714,109.326531,121.862245,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,4.750000,23.250000,11.750000,5.250000,4.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,3.625000,24.875000,13.375000,3.875000,3.250000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.125000,11.875000,24.000000,7.000000,2.875000,3.125000,2.778061,5.803571,6.367347,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.461538,0.230769,1.307692,20.000000,13.230769,3.923077,2.076923,3.000000,3.769231,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.846154,0.461538,0.384615,1.076923,19.000000,14.384615,4.076923,2.307692,3.307692,3.153846,0.000000,0.000000,0.000000,0.000000,0.000000,0.076923,1.000000,0.538462,0.384615,2.384615,24.615385,9.923077,2.153846,2.230769,2.692308,3.000000,1.921507,5.174254,5.394034};
  float rbfWeights[] = //{2.129407, -4.759265, -5.866391, -3.805425,4.032571, -7.081644,6.375648, -4.098478, -5.564061,5.215164,5.347754};
    {-0.014830,-1.586532,0.842004,-1.478210,1.868582,1.531812,0.848317,-1.189907,2.452640,-1.003280,-1.286481};
  //add bias term
  float f = rbfWeights[0];

  int numYellow=0;
  //loop on points
  for (int i=0; i<points->rows; ++i)
  {
    //clear histogram
    hist->setTo(0);

    //get the window indices
    int xmin = MAX(cvRound(points->at<float>(i, 0)-window), 0);
    int xmax = MIN(cvRound(points->at<float>(i, 0)+window),
                   im->cols);
    int ymin = MAX(cvRound(points->at<float>(i, 1)-window), 0);
    int ymax = MIN(cvRound(points->at<float>(i, 1)+window),
                   im->rows);

    //get mean for every channel
    float r=0.f, g=0.f, b=0.f, rr, gg, bb;
    int bin;
    for (int x=xmin; x<=xmax; x++)
	    for (int y=ymin; y<=ymax; y++)
	    {
        //get colors
        rr = (im->data + im->step*y)[x*3];
        gg = (im->data + im->step*y)[x*3+1];
        bb = (im->data + im->step*y)[x*3+2];
        //add to totals
        r += rr;
        g += gg;
        b += bb;

        if (rbf)
        {
          //compute histogram
          bin = MIN((int)(rr / binWidth), numBins);
          ((float*)hist->data)[bin] ++;
          bin = MIN((int)(gg / binWidth), numBins);
          ((float*)hist->data)[bin + numBins] ++;
          bin = MIN((int)(bb / binWidth), numBins);
          ((float*)hist->data)[bin + 2*numBins] ++;
        }
      }

    //normalize
    int num = (xmax-xmin+1) * (ymax-ymin+1);
    r /= num;
    g /= num;
    b /= num;

    //now compute differences
    float rg = r - g;
    float gb = g - b;
    float rb = r - b;

    //add differences to histogram
    if (rbf)
    {
	    ((float*)hist->data)[hist->cols-2] = fabs(rg);
	    ((float*)hist->data)[hist->cols-1] = fabs(gb);
	    ((float*)hist->data)[hist->cols] = fabs(rb);

	    //compute output of RBF model
	    //
	    //add rest of terms
	    for (int j=0; j<rbfNumCentroids; j++)
	    {
        //compute squared distance to centroid
        float d = 0., t;
        for (int k=0; k<histLen; k++)
        {
            t = ((float*)hist->data)[k] -
          rbfCentroids[j*histLen + k];
            d += t*t;
        }

        //compute product with weight
        f += rbfWeights[j+1] * exp(-.5 * d /rbfSigma);
	    }
    }

    //classify
    bool yellow;
    if (rbf)
	    yellow = f > rbfThreshold;
    else
	    yellow = rg>rgMin && rg<rgMax  && gb>gbMin && rb>rbMin;
    if (yellow)
	    numYellow++;

    if (DEBUG_LINES)
	    fprintf(stderr, "%s: f=%f, rg=%.2f, gb=%.2f, rb=%.2f\n",
              yellow? "YES" : "NO ", f, rg, gb, rb);
    //  	fprintf(stderr, "Point: %d is %s\n", i, yellow ? "YELLOW" : "WHITE");
    //  	fprintf(stderr, "r=%f, g=%f, b=%f\n", r, g, b);
    //  	fprintf(stderr, "rg=%f, gb=%f, rb=%f\n\n", rg, gb, rb);
  }

  //classify line
  LineColor clr = LINE_COLOR_WHITE;
  if (numYellow > numYellowMin*points->rows)
    clr = LINE_COLOR_YELLOW;

  //release
  delete hist;

  return clr;
}


/** \brief This function extracts bounding boxes from splines
 *
 * \param splines vector of splines
 * \param type the type of lines (LINE_HORIZONTAL or LINE_VERTICAL)
 * \param size the size of image containing the lines
 * \param boxes a vector of output bounding boxes
 */
void mcvGetSplinesBoundingBoxes(const vector<Spline> &splines, LineType type,
                                cv::Size size, vector<cv::Rect> &boxes)
{
  //copy lines to boxes
  int start, end;
  //clear
  boxes.clear();
  switch(type)
  {
    case LINE_VERTICAL:
      for(unsigned int i=0; i<splines.size(); ++i)
      {
        //get min and max x and add the bounding box covering the whole rows
        start = (int)fmin(splines[i].points[0].x,
                          splines[i].points[splines[i].degree].x);
        end = (int)fmax(splines[i].points[0].x,
                        splines[i].points[splines[i].degree].x);
        boxes.push_back(cv::Rect(start, 0, end-start+1, size.height-1));
      }
      break;

    case LINE_HORIZONTAL:
      for(unsigned int i=0; i<splines.size(); ++i)
      {
        //get min and max y and add the bounding box covering the whole cols
        start = (int)fmin(splines[i].points[0].y,
                          splines[i].points[splines[i].degree].y);
        end = (int)fmax(splines[i].points[0].y,
                        splines[i].points[splines[i].degree].y);
        boxes.push_back(cv::Rect(0, start, size.width-1, end-start+1));
      }
      break;
  }
}

/** \brief This function extracts bounding boxes from lines
 *
 * \param lines vector of lines
 * \param type the type of lines (LINE_HORIZONTAL or LINE_VERTICAL)
 * \param size the size of image containing the lines
 * \param boxes a vector of output bounding boxes
 */
void mcvGetLinesBoundingBoxes(const vector<Line> &lines, LineType type,
                              cv::Size size, vector<cv::Rect> &boxes)
{
  //copy lines to boxes
  int start, end;
  //clear
  boxes.clear();
  switch(type)
  {
    case LINE_VERTICAL:
      for(unsigned int i=0; i<lines.size(); ++i)
      {
        //get min and max x and add the bounding box covering the whole rows
        start = (int)fmin(lines[i].startPoint.x, lines[i].endPoint.x);
        end = (int)fmax(lines[i].startPoint.x, lines[i].endPoint.x);
        boxes.push_back(cv::Rect(start, 0, end-start+1, size.height-1));
      }
      break;

    case LINE_HORIZONTAL:
      for(unsigned int i=0; i<lines.size(); ++i)
      {
        //get min and max y and add the bounding box covering the whole cols
  	    start = (int)fmin(lines[i].startPoint.y, lines[i].endPoint.y);
        end = (int)fmax(lines[i].startPoint.y, lines[i].endPoint.y);
        boxes.push_back(cv::Rect(0, start, size.width-1, end-start+1));
      }
      break;
    }
}


/** \brief This function takes a bunch of lines, and check which
 * 2 lines can make a lane
 *
 * \param lines vector of lines
 * \param scores vector of line scores
 * \param wMu expected lane cols
 * \param wSigma std deviation of lane cols
 */
void mcvCheckLaneWidth(vector<Line> &lines, vector<float> &scores,
                       float wMu, float wSigma)
{
  //check if we have only 1, then exit
  if (lines.size() <2)
    return;

  //get distance between the lines assuming they are vertical
  int numInLines = lines.size();
  vector<float> rs;
  for (int i=0; i<numInLines; i++)
    rs.push_back( (lines[i].startPoint.x + lines[i].endPoint.x) / 2.);

  //now make a loop and check all possible pairs
  vector<float>::iterator ir, jr;
  int i, j, maxi, maxj;
  double score = 0., maxScore = -5.;
  wSigma *= wSigma;
  for (i=0, ir=rs.begin(); ir!=rs.end(); ir++, i++)
    for (j=i+1, jr=ir+1; jr!=rs.end(); jr++, j++)
    {
	    //get that score
	    score = fabs(*ir - *jr) - wMu;
	    score = exp(-.5 * score * score / wSigma);

	    //check max
	    if (score >= maxScore)
	    {
        maxScore = score;
        maxi = i;
        maxj = j;

        fprintf(stderr, "diff=%.5f score=%.10f i=%d j=%d\n",
                *ir - *jr, score, maxi, maxj);
	    }
    } // for j

  //now return the max and check threshold
  vector<Line> newLines;
  vector<float> newScores;

  if (maxScore<.4) //.25
  {
    maxi = scores[maxi] > scores[maxj] ? maxi : maxj;
    newLines.push_back(lines[maxi]);
    newScores.push_back(scores[maxi]);
  }
  //add both
  else
  {
    newLines.push_back(lines[maxi]);
    newLines.push_back(lines[maxj]);
    newScores.push_back(scores[maxi]);
    newScores.push_back(scores[maxj]);
  }
  lines = newLines;
  scores = newScores;

  //clear
  newLines.clear();
  newScores.clear();
  rs.clear();
}


void dummy()
{

}

} // namespace LaneDetector

#pragma GCC diagnostic pop
