/**
 * \file LaneDetector.cc
 * \author Mohamed Aly <malaa@caltech.edu>
 * \date Thu 26 Jul, 2007
 *
 */

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

#include <cv.h>
#include <highgui.h>

namespace LaneDetector
{
  // used for debugging
  int DEBUG_LINES = 0;

/**
 * This function filters the input image looking for horizontal
 * or vertical lines with specific width or height.
 *
 * \param inImage the input image
 * \param outImage the output image in IPM
 * \param wx width of kernel window in x direction = 2*wx+1
 * (default 2)
 * \param wy width of kernel window in y direction = 2*wy+1
 * (default 2)
 * \param sigmax std deviation of kernel in x (default 1)
 * \param sigmay std deviation of kernel in y (default 1)
 * \param lineType type of the line
 *      LINE_HORIZONTAL (default)
 *      LINE_VERTICAL
 */
 void mcvFilterLines(const CvMat *inImage, CvMat *outImage,
                     unsigned char wx, unsigned char wy, FLOAT sigmax,
                     FLOAT sigmay, LineType lineType)
{
    //define the two kernels
    //this is for 7-pixels wide
//     FLOAT_MAT_ELEM_TYPE derivp[] = {-2.328306e-10, -6.984919e-09, -1.008157e-07, -9.313226e-07, -6.178394e-06, -3.129616e-05, -1.255888e-04, -4.085824e-04, -1.092623e-03, -2.416329e-03, -4.408169e-03, -6.530620e-03, -7.510213e-03, -5.777087e-03, -5.777087e-04, 6.932504e-03, 1.372058e-02, 1.646470e-02, 1.372058e-02, 6.932504e-03, -5.777087e-04, -5.777087e-03, -7.510213e-03, -6.530620e-03, -4.408169e-03, -2.416329e-03, -1.092623e-03, -4.085824e-04, -1.255888e-04, -3.129616e-05, -6.178394e-06, -9.313226e-07, -1.008157e-07, -6.984919e-09, -2.328306e-10};
//     int derivLen = 35;
//     FLOAT_MAT_ELEM_TYPE smoothp[] = {2.384186e-07, 5.245209e-06, 5.507469e-05, 3.671646e-04, 1.744032e-03, 6.278515e-03, 1.778913e-02, 4.066086e-02, 7.623911e-02, 1.185942e-01, 1.541724e-01, 1.681881e-01, 1.541724e-01, 1.185942e-01, 7.623911e-02, 4.066086e-02, 1.778913e-02, 6.278515e-03, 1.744032e-03, 3.671646e-04, 5.507469e-05, 5.245209e-06, 2.384186e-07};
//     int smoothLen = 23;
  CvMat fx;
  CvMat fy;
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
      fx = cvMat(1, smoothLen, FLOAT_MAT_TYPE, smoothp);
      fy = cvMat(derivLen, 1, FLOAT_MAT_TYPE, derivp);
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
      fy = cvMat(1, smoothLen, FLOAT_MAT_TYPE, smoothp);
      fx = cvMat(derivLen, 1, FLOAT_MAT_TYPE, derivp);
    }
    break;
  }

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
  //SHOW_MAT(kernel, "Kernel:");
  }//#endif

#warning "still check subtracting mean from image"
  //subtract mean
  CvScalar mean = cvAvg(inImage);
  cvSubS(inImage, mean, outImage);


  //do the filtering
  cvFilter2D(outImage, outImage, &fx); //inImage outImage
  cvFilter2D(outImage, outImage, &fy);


//     CvMat *deriv = cvCreateMat
//     //define x
//     CvMat *x = cvCreateMat(2*wx+1, 1, FLOAT_MAT_TYPE);
//     //define y
//     CvMat *y = cvCreateMat(2*wy+1, 1, FLOAT_MAT_TYPE);

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
//     CvMat *kernel = cvCreateMat(2*wy+1, 2*wx+1, FLOAT_MAT_TYPE);
//     cvGEMM(y, x, 1, 0, 1, kernel, CV_GEMM_B_T);

//     //subtract the mean
//     CvScalar mean = cvAvg(kernel);
//     cvSubS(kernel, mean, kernel);

//     #ifdef DEBUG_GET_STOP_LINES
//     //SHOW_MAT(kernel, "Kernel:");
//     #endif

//     //do the filtering
//     cvFilter2D(inImage, outImage, kernel);

//     //clean
//     cvReleaseMat(&x);
//     cvReleaseMat(&y);
//     cvReleaseMat(&kernel);
}

/**
 * This function gets a 1-D gaussian filter with specified
 * std deviation and range
 *
 * \param kernel input mat to hold the kernel (2*w+1x1)
 *      column vector (already allocated)
 * \param w width of kernel is 2*w+1
 * \param sigma std deviation
 */
void mcvGetGaussianKernel(CvMat *kernel, unsigned char w, FLOAT sigma)
{
  //get variance
  sigma *= sigma;

  //get the kernel
  for (double i=-w; i<=w; i++)
      CV_MAT_ELEM(*kernel, FLOAT_MAT_ELEM_TYPE, int(i+w), 0) =
          (FLOAT_MAT_ELEM_TYPE) exp(-(.5/sigma)*(i*i));
}

/**
 * This function gets a 1-D second derivative gaussian filter
 * with specified std deviation and range
 *
 * \param kernel input mat to hold the kernel (2*w+1x1)
 *      column vector (already allocated)
 * \param w width of kernel is 2*w+1
 * \param sigma std deviation
 */
void mcvGet2DerivativeGaussianKernel(CvMat *kernel,
                                     unsigned char w, FLOAT sigma)
{
  //get variance
  sigma *= sigma;

  //get the kernel
  for (double i=-w; i<=w; i++)
      CV_MAT_ELEM(*kernel, FLOAT_MAT_ELEM_TYPE, int(i+w), 0) =
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
 * \param linePixelWidth width (or height) of lines to detect
 * \param localMaxima whether to detect local maxima or just get
 *      the maximum
 * \param detectionThreshold threshold for detection
 * \param smoothScores whether to smooth scores detected or not
 */
void mcvGetHVLines(const CvMat *inImage, vector <Line> *lines,
                   vector <FLOAT> *lineScores, LineType lineType,
                   FLOAT linePixelWidth, bool binarize, bool localMaxima,
                   FLOAT detectionThreshold, bool smoothScores)
{
  CvMat * image = cvCloneMat(inImage);
  //binarize input image if to binarize
  if (binarize)
  {
    //mcvBinarizeImage(image);
    image = cvCreateMat(inImage->rows, inImage->cols, INT_MAT_TYPE);
    cvThreshold(inImage, image, 0, 1, CV_THRESH_BINARY); //0.05
  }

  //get sum of lines through horizontal or vertical
  //sumLines is a column vector
  CvMat sumLines, *sumLinesp;
  int maxLineLoc = 0;
  switch (lineType)
  {
    case LINE_HORIZONTAL:
      sumLinesp = cvCreateMat(image->height, 1, FLOAT_MAT_TYPE);
      cvReduce(image, sumLinesp, 1, CV_REDUCE_SUM); //_AVG
      cvReshape(sumLinesp, &sumLines, 0, 0);
      //max location for a detected line
      maxLineLoc = image->height-1;
      break;
    case LINE_VERTICAL:
      sumLinesp = cvCreateMat(1, image->width, FLOAT_MAT_TYPE);
      cvReduce(image, sumLinesp, 0, CV_REDUCE_SUM); //_AVG
      cvReshape(sumLinesp, &sumLines, 0, image->width);
      //max location for a detected line
      maxLineLoc = image->width-1;
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
  CvMat smooth = cvMat(1, smoothWidth, CV_32FC1, smoothp);
  if (smoothScores)
    cvFilter2D(&sumLines, &sumLines, &smooth);
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
	    FLOAT val = CV_MAT_ELEM(sumLines, FLOAT_MAT_ELEM_TYPE, i, 0);
	    //check if local maximum
	    if( (val > CV_MAT_ELEM(sumLines, FLOAT_MAT_ELEM_TYPE, i-1, 0))
        && (val > CV_MAT_ELEM(sumLines, FLOAT_MAT_ELEM_TYPE, i+1, 0))
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
//     gnuplot_ctrl *h =  mcvPlotMat1D(NULL, &sumLines, "Line Scores");
//     CvMat *y = mcvVector2Mat(sumLinesMax);
//     CvMat *x =  mcvVector2Mat(sumLinesMaxLoc);
//     mcvPlotMat2D(h, x, y);
//     //gnuplot_plot_xy(h, (double*)&sumLinesMaxLoc,(double*)&sumLinesMax, sumLinesMax.size(),"");
//     cin.get();
//     gnuplot_close(h);
//     cvReleaseMat(&x);
//     cvReleaseMat(&y);
//}//#endif
  //process the found maxima
  for (int i=0; i<(int)sumLinesMax.size(); i++)
  {
    //get subpixel accuracy
    double maxLocAcc = mcvGetLocalMaxSubPixel(
      CV_MAT_ELEM(sumLines, FLOAT_MAT_ELEM_TYPE, MAX(sumLinesMaxLoc[i]-1,0), 0),
      CV_MAT_ELEM(sumLines, FLOAT_MAT_ELEM_TYPE, sumLinesMaxLoc[i], 0),
      CV_MAT_ELEM(sumLines, FLOAT_MAT_ELEM_TYPE,
                  MIN(sumLinesMaxLoc[i]+1,maxLineLoc), 0) );
    maxLocAcc += sumLinesMaxLoc[i];
    maxLocAcc = MIN(MAX(0, maxLocAcc), maxLineLoc);


	//TODO: get line extent

	//put the extracted line
    Line line;
    switch (lineType)
    {
      case LINE_HORIZONTAL:
        line.startPoint.x = 0.5;
        line.startPoint.y = (FLOAT)maxLocAcc + .5;//sumLinesMaxLoc[i]+.5;
        line.endPoint.x = inImage->width-.5;
        line.endPoint.y = line.startPoint.y;
        break;
      case LINE_VERTICAL:
        line.startPoint.x = (FLOAT)maxLocAcc + .5;//sumLinesMaxLoc[i]+.5;
        line.startPoint.y = .5;
        line.endPoint.x = line.startPoint.x;
        line.endPoint.y = inImage->height-.5;
        break;
    }
    (*lines).push_back(line);
    if (lineScores)
        (*lineScores).push_back(sumLinesMax[i]);
  }//for

  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    CvMat *im, *im2 = cvCloneMat(image);
    if (binarize)
      cvConvertScale(im2, im2, 255, 0);

    if (binarize)
	    im = cvCreateMat(image->rows, image->cols, CV_8UC3);
    else
	    im = cvCreateMat(image->rows, image->cols, CV_32FC3);
    mcvScaleMat(im2, im2);
    cvCvtColor(im2, im, CV_GRAY2RGB);
    for (unsigned int i=0; i<lines->size(); i++)
    {
      Line line = (*lines)[i];
      mcvIntersectLineWithBB(&line, cvSize(image->cols, image->rows), &line);
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
    cvReleaseMat(&im);
    cvReleaseMat(&im2);
  }

  //clean
  cvReleaseMat(&sumLinesp);
  //cvReleaseMat(&smooth);
  sumLinesMax.clear();
  sumLinesMaxLoc.clear();
  cvReleaseMat(&image);
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

void mcvGetHoughTransformLines(const CvMat *inImage, vector <Line> *lines,
                               vector <FLOAT> *lineScores,
                               FLOAT rMin, FLOAT rMax, FLOAT rStep,
                               FLOAT thetaMin, FLOAT thetaMax,
                               FLOAT thetaStep, bool binarize, bool localMaxima,
                               FLOAT detectionThreshold, bool smoothScores,
                               bool group, FLOAT groupThreshold)
{
  CvMat *image;

  //binarize input image if to binarize
  if (!binarize)
  {
    image = cvCloneMat(inImage); assert(image!=0);
    //         mcvBinarizeImage(image);
  }
  //binarize input image
  else
  {
    image = cvCreateMat(inImage->rows, inImage->cols, INT_MAT_TYPE);
    cvThreshold(inImage, image, 0, 1, CV_THRESH_BINARY); //0.05
    //get max of image
    //double maxval, minval;
    //cvMinMaxLoc(inImage, &minval, &maxval);
    //cout << "Max = " << maxval << "& Min=" << minval << "\n";
    //CvScalar mean = cvAvg(inImage);
    //cout << "Mean=" << mean.val[0] << "\n";
  }

  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    SHOW_IMAGE(image, "Hough thresholded image", 10);
  }//#endif

  //define the accumulator array: rows correspond to r and columns to theta
  int rBins = int((rMax-rMin)/rStep);
  int thetaBins = int((thetaMax-thetaMin)/thetaStep);
  CvMat *houghSpace = cvCreateMat(rBins, thetaBins, CV_MAT_TYPE(image->type)); //FLOAT_MAT_TYPE);
  assert(houghSpace!=0);
  //init to zero
  cvSet(houghSpace, cvRealScalar(0));

  //init values of r and theta
  FLOAT *rs = new FLOAT[rBins];
  FLOAT *thetas = new FLOAT[thetaBins];
  FLOAT r, theta;
  int ri, thetai;
  for (r=rMin+rStep/2,  ri=0 ; ri<rBins; ri++,r+=rStep)
    rs[ri] = r;
  for (theta=thetaMin, thetai=0 ; thetai<thetaBins; thetai++,
    theta+=thetaStep)
    thetas[thetai] = theta;

  //get non-zero points in the image
  int nzCount = cvCountNonZero(image);
  CvMat *nzPoints = cvCreateMat(nzCount, 2, CV_32SC1);
  int idx = 0;
  for (int i=0; i<image->width; i++)
    for (int j=0; j<image->height; j++)
      if ( cvGetReal2D(image, j, i) )
      {
        CV_MAT_ELEM(*nzPoints, int, idx, 0) = i;
        CV_MAT_ELEM(*nzPoints, int, idx, 1) = j;
        idx++;
      }

    //calculate r values for all theta and all points
    //CvMat *rPoints = cvCreateMat(image->width*image->height, thetaBins, CV_32SC1);//FLOAT_MAT_TYPE)
    //CvMat *rPoints = cvCreateMat(nzCount, thetaBins, CV_32SC1);//FLOAT_MAT_TYPE);
    //cvSet(rPoints, cvRealScalar(-1));
    //loop on x
    //float x=0.5, y=0.5;
    int i, k; //j
    for (i=0; i<nzCount; i++)
      for (k=0; k<thetaBins; k++)
      {
        //compute the r value for that point and that theta
        theta = thetas[k];
        float rval = CV_MAT_ELEM(*nzPoints, int, i, 0) * cos(theta) +
        CV_MAT_ELEM(*nzPoints, int, i, 1) * sin(theta); //x y
        int r = (int)( ( rval - rMin) / rStep);
        //	    CV_MAT_ELEM(*rPoints, int, i, k) =
        //(int)( ( rval - rMin) / rStep);

        //accumulate in the hough space if a valid value
        if (r>=0 && r<rBins)
          if(binarize)
            CV_MAT_ELEM(*houghSpace, INT_MAT_ELEM_TYPE, r, k)++;
          //CV_MAT_ELEM(*image, INT_MAT_ELEM_TYPE, j, i);
        else
          CV_MAT_ELEM(*houghSpace, FLOAT_MAT_ELEM_TYPE, r, k)+=
          CV_MAT_ELEM(*image, FLOAT_MAT_ELEM_TYPE,
                      CV_MAT_ELEM(*nzPoints, int, i, 1),
                      CV_MAT_ELEM(*nzPoints, int, i, 0));
      }

      //clear
      cvReleaseMat(&nzPoints);

//     bool inside;
//     for (i=0; i<image->width; i++) //x=0; x++
// 	//loop on y
// 	for (j=0; j<image->height; j++) //y=0 y++
// 	    //loop on theta
// 	    for (k=0; k<thetaBins; k++)
// 	    {
// 		//compute the r value and then subtract rMin and div by rStep
// 		//to get the r bin index to which it belongs (0->rBins-1)
// 		if (lineConf->binarize && CV_MAT_ELEM(*image, INT_MAT_ELEM_TYPE, j, i) !=0)
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
// 				i*image->height + j, k) =
// 			(int)( ( rval - lineConf->rMin) / lineConf->rStep);
// 		}

// 	    }

//      SHOW_MAT(rPoints, "rPoints");
//      cin.get();

    //now we should accumulate the values into the approprate bins in the houghSpace
//     for (ri=0; ri<rBins; ri++)
// 	for (thetai=0; thetai<thetaBins; thetai++)
// 	    for (i=0; i<image->width; i++)
// 		for (j=0; j<image->height; j++)
// 		{
// 		    //check if this cell belongs to that bin or not
// 		    if (CV_MAT_ELEM(*rPoints, int,
// 				    i*image->height + j , thetai)==ri)
// 		    {
// 			if(lineConf->binarize)
// 			    CV_MAT_ELEM(*houghSpace, INT_MAT_ELEM_TYPE, ri, thetai)++;
// 			//CV_MAT_ELEM(*image, INT_MAT_ELEM_TYPE, j, i);
// 			else
// 			    CV_MAT_ELEM(*houghSpace, FLOAT_MAT_ELEM_TYPE, ri, thetai)+=
// 				CV_MAT_ELEM(*image, FLOAT_MAT_ELEM_TYPE, j, i);
// 		    }
// 		}


  //smooth hough transform
  if (smoothScores)
    cvSmooth(houghSpace, houghSpace, CV_GAUSSIAN, 3);

  //get local maxima
  vector <double> maxLineScores;
  vector <CvPoint> maxLineLocs;
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
  CvPoint maxLineLoc;
  cvMinMaxLoc(houghSpace, 0, &maxLineScore, 0, &maxLineLoc);
  if (maxLineScores.size()==0 && maxLineScore>=detectionThreshold)
  {
    maxLineScores.push_back(maxLineScore);
    maxLineLocs.push_back(maxLineLoc);
  }


  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    // 	cout << "Local maxima = " << maxLineScores.size() << "\n";

    {
      CvMat *im, *im2 = cvCloneMat(image);
      if (binarize)
        cvConvertScale(im2, im2, 255, 0);

      if (binarize)
        im = cvCreateMat(image->rows, image->cols, CV_8UC3);//cvCloneMat(image);
      else
        im = cvCreateMat(image->rows, image->cols, CV_32FC3);
      cvCvtColor(im2, im, CV_GRAY2RGB);
      for (int i=0; i<(int)maxLineScores.size(); i++)
      {
        Line line;
        assert(maxLineLocs[i].x>=0 && maxLineLocs[i].x<thetaBins);
        assert(maxLineLocs[i].y>=0 && maxLineLocs[i].y<rBins);
        mcvIntersectLineRThetaWithBB(rs[maxLineLocs[i].y], thetas[maxLineLocs[i].x],
                                    cvSize(image->cols, image->rows), &line);
                                    if (binarize)
                                      mcvDrawLine(im, line, CV_RGB(255,0,0), 1);
                                    else
                                      mcvDrawLine(im, line, CV_RGB(1,0,0), 1);
      }
      SHOW_IMAGE(im, "Hough before grouping", 10);
      cvReleaseMat(&im);
      cvReleaseMat(&im2);

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
      vector<CvPoint>::iterator iloc, jloc,
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
            // 		    CvPoint tloc;
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
    CvMat *im, *im2 = cvCloneMat(image);
    if (binarize)
      cvConvertScale(im2, im2, 255, 0);
    if (binarize)
      im = cvCreateMat(image->rows, image->cols, CV_8UC3);//cvCloneMat(image);
    else
      im = cvCreateMat(image->rows, image->cols, CV_32FC3);
    cvCvtColor(im2, im, CV_GRAY2RGB);
    for (int i=0; i<(int)maxLineScores.size(); i++)
    {
      Line line;
      assert(maxLineLocs[i].x>=0 && maxLineLocs[i].x<thetaBins);
      assert(maxLineLocs[i].y>=0 && maxLineLocs[i].y<rBins);
      mcvIntersectLineRThetaWithBB(rs[maxLineLocs[i].y],
                                   thetas[maxLineLocs[i].x],
                                   cvSize(image->cols, image->rows), &line);
      if (binarize)
        mcvDrawLine(im, line, CV_RGB(255,0,0), 1);
      else
        mcvDrawLine(im, line, CV_RGB(1,0,0), 1);
    }
    SHOW_IMAGE(im, "Hough after grouping", 10);
    cvReleaseMat(&im);
    cvReleaseMat(&im2);

    //put local maxima in image
    CvMat *houghSpaceClr;
    if(binarize)
      houghSpaceClr = cvCreateMat(houghSpace->height, houghSpace->width,
                                  CV_8UC3);
    else
      houghSpaceClr = cvCreateMat(houghSpace->height, houghSpace->width,
                                  CV_32FC3);
    mcvScaleMat(houghSpace, houghSpace);
    cvCvtColor(houghSpace, houghSpaceClr, CV_GRAY2RGB);
    for (int i=0; i<(int)maxLineLocs.size(); i++)
      cvCircle(houghSpaceClr, cvPoint(maxLineLocs[i].x, maxLineLocs[i].y),
              1, CV_RGB(1, 0, 0), -1);
              // 	    if(lineConf->binarize)
              // 		CV_MAT_ELEM(*houghSpace, unsigned char, maxLineLocs[i].y,
              // 			    maxLineLocs[i].x) = 255;
              // 	    else
              // 		CV_MAT_ELEM(*houghSpace, float, maxLineLocs[i].y, maxLineLocs[i].x) = 1.f;
              //show the hough space
    SHOW_IMAGE(houghSpaceClr, "Hough Space", 10);
    cvReleaseMat(&houghSpaceClr);
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
                                   cvSize(image->cols, image->rows), &line);
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
  cvReleaseMat(&image);
  cvReleaseMat(&houghSpace);
  //cvReleaseMat(&rPoints);
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
void mcvBinarizeImage(CvMat *inImage)
{

  if (CV_MAT_TYPE(inImage->type)==FLOAT_MAT_TYPE)
  {
    for (int i=0; i<inImage->height; i++)
      for (int j=0; j<inImage->width; j++)
        if (CV_MAT_ELEM(*inImage, FLOAT_MAT_ELEM_TYPE, i, j) != 0.f)
          CV_MAT_ELEM(*inImage, FLOAT_MAT_ELEM_TYPE, i, j)=1;
  }
  else if (CV_MAT_TYPE(inImage->type)==INT_MAT_TYPE)
  {
    for (int i=0; i<inImage->height; i++)
      for (int j=0; j<inImage->width; j++)
        if (CV_MAT_ELEM(*inImage, INT_MAT_ELEM_TYPE, i, j) != 0)
          CV_MAT_ELEM(*inImage, INT_MAT_ELEM_TYPE, i, j)=1;
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
    if (inVector->height==1) \
    { \
        /*initial value*/ \
        tmax = (double) CV_MAT_ELEM(*inVector, type, 0, inVector->width-1); \
        tmaxLoc = inVector->width-1; \
        /*loop*/ \
        for (int i=inVector->width-1-ignore; i>=0+ignore; i--) \
        { \
            if (tmax<CV_MAT_ELEM(*inVector, type, 0, i)) \
            { \
                tmax = CV_MAT_ELEM(*inVector, type, 0, i); \
                tmaxLoc = i; \
            } \
        } \
    } \
    /*column vector */ \
    else \
    { \
        /*initial value*/ \
        tmax = (double) CV_MAT_ELEM(*inVector, type, inVector->height-1, 0); \
        tmaxLoc = inVector->height-1; \
        /*loop*/ \
        for (int i=inVector->height-1-ignore; i>=0+ignore; i--) \
        { \
            if (tmax<CV_MAT_ELEM(*inVector, type, i, 0)) \
            { \
                tmax = (double) CV_MAT_ELEM(*inVector, type, i, 0); \
                tmaxLoc = i; \
            } \
        } \
    } \

void mcvGetVectorMax(const CvMat *inVector, double *max, int *maxLoc, int ignore)
{
  double tmax;
  int tmaxLoc;

  if (CV_MAT_TYPE(inVector->type)==FLOAT_MAT_TYPE)
  {
    MCV_VECTOR_MAX(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inVector->type)==INT_MAT_TYPE)
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
 *       where each location is cvPoint(x=col, y=row) zero-based
 *
 */

void mcvGetMatLocalMax(const CvMat *inMat, vector<double> &localMaxima,
		     vector<CvPoint> &localMaximaLoc, double threshold)
{

  double val;

#define MCV_MAT_LOCAL_MAX(type)  \
    /*loop on the matrix and get points that are larger than their*/ \
    /*neighboring 8 pixels*/ \
    for(int i=1; i<inMat->rows-1; i++) \
	for (int j=1; j<inMat->cols-1; j++) \
	{ \
	    /*get the current value*/ \
	    val = CV_MAT_ELEM(*inMat, type, i, j); \
	    /*check if it's larger than all its neighbors*/ \
	    if( val > CV_MAT_ELEM(*inMat, type, i-1, j-1) && \
		val > CV_MAT_ELEM(*inMat, type, i-1, j) && \
		val > CV_MAT_ELEM(*inMat, type, i-1, j+1) && \
		val > CV_MAT_ELEM(*inMat, type, i, j-1) && \
		val > CV_MAT_ELEM(*inMat, type, i, j+1) && \
		val > CV_MAT_ELEM(*inMat, type, i+1, j-1) && \
		val > CV_MAT_ELEM(*inMat, type, i+1, j) && \
		val > CV_MAT_ELEM(*inMat, type, i+1, j+1) && \
                val >= threshold) \
	    { \
		/*found a local maxima, put it in the return vector*/ \
		/*in decending order*/ \
		/*iterators for the two vectors*/ \
		vector<double>::iterator k; \
		vector<CvPoint>::iterator l; \
		/*loop till we find the place to put it in descendingly*/ \
		for(k=localMaxima.begin(), l=localMaximaLoc.begin(); \
		    k != localMaxima.end()  && val<= *k; k++,l++); \
		/*add its index*/ \
		localMaxima.insert(k, val); \
		localMaximaLoc.insert(l, cvPoint(j, i)); \
	    } \
	}

  //check type
  if (CV_MAT_TYPE(inMat->type)==FLOAT_MAT_TYPE)
  {
    MCV_MAT_LOCAL_MAX(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type)==INT_MAT_TYPE)
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
 *       where each location is cvPoint(x=col, y=row) zero-based
 *
 */

void mcvGetMatMax(const CvMat *inMat, vector<double> &maxima,
                  vector<CvPoint> &maximaLoc, double threshold)
{

  double val;

#define MCV_MAT_MAX(type)  \
    /*loop on the matrix and get points that are larger than their*/ \
    /*neighboring 8 pixels*/ \
    for(int i=1; i<inMat->rows-1; i++) \
	for (int j=1; j<inMat->cols-1; j++) \
	{ \
	    /*get the current value*/ \
	    val = CV_MAT_ELEM(*inMat, type, i, j); \
	    /*check if it's larger than threshold*/ \
	    if (val >= threshold) \
	    { \
		/*found a maxima, put it in the return vector*/ \
		/*in decending order*/ \
		/*iterators for the two vectors*/ \
		vector<double>::iterator k; \
		vector<CvPoint>::iterator l; \
		/*loop till we find the place to put it in descendingly*/ \
		for(k=maxima.begin(), l=maximaLoc.begin(); \
		    k != maxima.end()  && val<= *k; k++,l++); \
		/*add its index*/ \
		maxima.insert(k, val); \
		maximaLoc.insert(l, cvPoint(j, i)); \
	    } \
	}

  //check type
  if (CV_MAT_TYPE(inMat->type)==FLOAT_MAT_TYPE)
  {
    MCV_MAT_MAX(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type)==INT_MAT_TYPE)
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
    if(inVec->height == 1)							\
    {										\
	for(int i=1; i<inVec->width-1; i++)					\
	{									\
	    /*get the current value*/						\
	    val = CV_MAT_ELEM(*inVec, type, 0, i);				\
	    /*check if it's larger than all its neighbors*/			\
	    if( val > CV_MAT_ELEM(*inVec, type, 0, i-1) &&			\
		val > CV_MAT_ELEM(*inVec, type, 0, i+1) )			\
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
	for(int i=1; i<inVec->height-1; i++)					\
	{									\
	    /*get the current value*/						\
	    val = CV_MAT_ELEM(*inVec, type, i, 0);				\
	    /*check if it's larger than all its neighbors*/			\
	    if( val > CV_MAT_ELEM(*inVec, type, i-1, 0) &&			\
		val > CV_MAT_ELEM(*inVec, type, i+1, 0) )			\
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

void mcvGetVectorLocalMax(const CvMat *inVec, vector<double> &localMaxima,
                          vector<int> &localMaximaLoc)
{

  double val;

  //check type
  if (CV_MAT_TYPE(inVec->type)==FLOAT_MAT_TYPE)
  {
    MCV_VECTOR_LOCAL_MAX(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inVec->type)==INT_MAT_TYPE)
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
FLOAT mcvGetQuantile(const CvMat *mat, FLOAT qtile)
{
  //make it a row vector
  CvMat rowMat;
  cvReshape(mat, &rowMat, 0, 1);

  //get the quantile
  FLOAT qval;
  qval = quantile((FLOAT*) rowMat.data.ptr, rowMat.width, qtile);

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
void mcvThresholdLower(const CvMat *inMat, CvMat *outMat, FLOAT threshold)
{

#define MCV_THRESHOLD_LOWER(type) \
     for (int i=0; i<inMat->height; i++) \
        for (int j=0; j<inMat->width; j++) \
            if ( CV_MAT_ELEM(*inMat, type, i, j)<threshold) \
                CV_MAT_ELEM(*outMat, type, i, j)=(type) 0; /*check it, was: threshold*/\

  //check if to copy into outMat or not
  if (inMat != outMat)
    cvCopy(inMat, outMat);

  //check type
  if (CV_MAT_TYPE(inMat->type)==FLOAT_MAT_TYPE)
  {
    MCV_THRESHOLD_LOWER(FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type)==INT_MAT_TYPE)
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
void mcvGetStopLines(const CvMat *inImage, vector<Line> *stopLines,
		     vector<FLOAT> *lineScores, const CameraInfo *cameraInfo,
		      LaneDetectorConf *stopLineConf)

{
  //input size
  CvSize inSize = cvSize(inImage->width, inImage->height);

  //TODO: smooth image
  CvMat *image = cvCloneMat(inImage);
  //cvSmooth(image, image, CV_GAUSSIAN, 5, 5, 1, 1);

  IPMInfo ipmInfo;

//     //get the IPM size such that we have height of the stop line
//     //is 3 pixels
//     double ipmWidth, ipmHeight;
//     mcvGetIPMExtent(cameraInfo, &ipmInfo);
//     ipmHeight = 3*(ipmInfo.yLimits[1]-ipmInfo.yLimits[0]) / (stopLineConf->lineHeight/3.);
//     ipmWidth = ipmHeight * 4/3;
//     //put into the conf
//     stopLineConf->ipmWidth = int(ipmWidth);
//     stopLineConf->ipmHeight = int(ipmHeight);

//         if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
//     cout << "IPM width:" << stopLineConf->ipmWidth << " IPM height:"
// 	 << stopLineConf->ipmHeight << "\n";
//     }//#endif


  //Get IPM
  CvSize ipmSize = cvSize((int)stopLineConf->ipmWidth,
                          (int)stopLineConf->ipmHeight);
  CvMat * ipm;
  ipm = cvCreateMat(ipmSize.height, ipmSize.width, inImage->type);
  //mcvGetIPM(inImage, ipm, &ipmInfo, cameraInfo);
  ipmInfo.vpPortion = stopLineConf->ipmVpPortion;
  ipmInfo.ipmLeft = stopLineConf->ipmLeft;
  ipmInfo.ipmRight = stopLineConf->ipmRight;
  ipmInfo.ipmTop = stopLineConf->ipmTop;
  ipmInfo.ipmBottom = stopLineConf->ipmBottom;
  ipmInfo.ipmInterpolation = stopLineConf->ipmInterpolation;
  list<CvPoint> outPixels;
  list<CvPoint>::iterator outPixelsi;
  mcvGetIPM(image, ipm, &ipmInfo, cameraInfo, &outPixels);

  //smooth the IPM
  //cvSmooth(ipm, ipm, CV_GAUSSIAN, 5, 5, 1, 1);

  //debugging
  CvMat *dbIpmImage;
  if(DEBUG_LINES) {//    #ifdef DEBUG_GET_STOP_LINES
      dbIpmImage = cvCreateMat(ipm->height, ipm->width, ipm->type);
      cvCopy(ipm, dbIpmImage);
  }//#endif


  //compute stop line width: 2000 mm
  FLOAT stopLinePixelWidth = stopLineConf->lineWidth * ipmInfo.xScale;
  //stop line pixel height: 12 inches = 12*25.4 mm
  FLOAT stopLinePixelHeight = stopLineConf->lineHeight *  ipmInfo.yScale;
  //kernel dimensions
  //unsigned char wx = 2;
  //unsigned char wy = 2;
  FLOAT sigmax = stopLinePixelWidth;
  FLOAT sigmay = stopLinePixelHeight;

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
  //cout << "Line width:" << stopLinePixelWidth << "Line height:"
  //	 << stopLinePixelHeight << "\n";
  }//#endif

  //filter the IPM image
  mcvFilterLines(ipm, ipm, stopLineConf->kernelWidth,
                 stopLineConf->kernelHeight, sigmax, sigmay,
                 LINE_HORIZONTAL);

    //zero out points outside the image in IPM view
  for(outPixelsi=outPixels.begin(); outPixelsi!=outPixels.end(); outPixelsi++)
    CV_MAT_ELEM(*ipm, float, (*outPixelsi).y, (*outPixelsi).x) = 0;
  outPixels.clear();

  //zero out negative values
  mcvThresholdLower(ipm, ipm, 0);

  //compute quantile: .985
  FLOAT qtileThreshold = mcvGetQuantile(ipm, stopLineConf->lowerQuantile);
  mcvThresholdLower(ipm, ipm, qtileThreshold);

  //debugging
  CvMat *dbIpmImageThresholded;
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    dbIpmImageThresholded = cvCreateMat(ipm->height, ipm->width, ipm->type);
    cvCopy(ipm, dbIpmImageThresholded);
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
    //FLOAT rMin = 0.05*ipm->height, rMax = 0.4*ipm->height, rStep = 1;
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
    //put a dummy line at the beginning till we check that cvDiv bug
    Line dummy = {{1.,1.},{2.,2.}};
    stopLines->insert(stopLines->begin(), dummy);
    //convert to mat and get in image coordinates
    CvMat *mat = cvCreateMat(2, 2*stopLines->size(), FLOAT_MAT_TYPE);
    mcvLines2Mat(stopLines, mat);
    stopLines->clear();
    mcvTransformGround2Image(mat, mat, cameraInfo);
    //get back to vector
    mcvMat2Lines(mat, stopLines);
    //remove the dummy line at the beginning
    stopLines->erase(stopLines->begin());
    //clear
    cvReleaseMat(&mat);

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
    //CvMat *image = cvCreateMat(inImage->height, inImage->width, CV_32FC3);
    //cvCvtColor(inImage, image, CV_GRAY2RGB);
    //CvMat *image = cvCloneMat(inImage);
    //for (int i=0; i<(int)stopLines->size(); i++)
    for (int i=0; i<1 && stopLines->size()>0; i++)
    {
      //SHOW_POINT((*stopLines)[i].startPoint, "start");
      //SHOW_POINT((*stopLines)[i].endPoint, "end");
      mcvDrawLine(image, (*stopLines)[i], CV_RGB(255,0,0), 3);
    }
    SHOW_IMAGE(image, "Detected Stoplines", 10);
    //cvReleaseMat(&image);
    cvReleaseMat(&dbIpmImage);
    cvReleaseMat(&dbIpmImageThresholded);
    dbIpmStopLines.clear();
  }//#endif //DEBUG_GET_STOP_LINES

  //clear
  cvReleaseMat(&ipm);
  cvReleaseMat(&image);
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
 *   initialize the current detection (NULL to ignore)
 *
 *
 */
void mcvGetLanes(const CvMat *inImage, const CvMat* clrImage,
                 vector<Line> *lanes, vector<FLOAT> *lineScores,
                 vector<Spline> *splines, vector<float> *splineScores,
                 CameraInfo *cameraInfo, LaneDetectorConf *stopLineConf,
                 LineState* state)
{
  //input size
  CvSize inSize = cvSize(inImage->width, inImage->height);

  //TODO: smooth image
  CvMat *image = cvCloneMat(inImage);
  //cvSmooth(image, image, CV_GAUSSIAN, 5, 5, 1, 1);

  //SHOW_IMAGE(image, "Input image", 10);

  IPMInfo ipmInfo;

  //state: create a new structure, and put pointer to it if it's null
  LineState newState;
  if(!state) state = &newState;

//     //get the IPM size such that we have height of the stop line
//     //is 3 pixels
//     double ipmWidth, ipmHeight;
//     mcvGetIPMExtent(cameraInfo, &ipmInfo);
//     ipmHeight = 3*(ipmInfo.yLimits[1]-ipmInfo.yLimits[0]) / (stopLineConf->lineHeight/3.);
//     ipmWidth = ipmHeight * 4/3;
//     //put into the conf
//     stopLineConf->ipmWidth = int(ipmWidth);
//     stopLineConf->ipmHeight = int(ipmHeight);

//     #ifdef DEBUG_GET_STOP_LINES
//     cout << "IPM width:" << stopLineConf->ipmWidth << " IPM height:"
// 	 << stopLineConf->ipmHeight << "\n";
//     #endif


  //Get IPM
  CvSize ipmSize = cvSize((int)stopLineConf->ipmWidth,
      (int)stopLineConf->ipmHeight);
  CvMat * ipm;
  ipm = cvCreateMat(ipmSize.height, ipmSize.width, inImage->type);
  //mcvGetIPM(inImage, ipm, &ipmInfo, cameraInfo);
  ipmInfo.vpPortion = stopLineConf->ipmVpPortion;
  ipmInfo.ipmLeft = stopLineConf->ipmLeft;
  ipmInfo.ipmRight = stopLineConf->ipmRight;
  ipmInfo.ipmTop = stopLineConf->ipmTop;
  ipmInfo.ipmBottom = stopLineConf->ipmBottom;
  ipmInfo.ipmInterpolation = stopLineConf->ipmInterpolation;
  list<CvPoint> outPixels;
  list<CvPoint>::iterator outPixelsi;
  mcvGetIPM(image, ipm, &ipmInfo, cameraInfo, &outPixels);

  //smooth the IPM image with 5x5 gaussian filter
#warning "Check: Smoothing IPM image"
  //cvSmooth(ipm, ipm, CV_GAUSSIAN, 3);
  //      SHOW_MAT(ipm, "ipm");

  //     //subtract mean
  //     CvScalar mean = cvAvg(ipm);
  //     cvSubS(ipm, mean, ipm);

  //keep copy
  CvMat* rawipm = cvCloneMat(ipm);

  //smooth the IPM
  //cvSmooth(ipm, ipm, CV_GAUSSIAN, 5, 5, 1, 1);

  //debugging
  CvMat *dbIpmImage;
  if(DEBUG_LINES)
  {//#ifdef DEBUG_GET_STOP_LINES
    dbIpmImage = cvCreateMat(ipm->height, ipm->width, ipm->type);
    cvCopy(ipm, dbIpmImage);
    //show the IPM image
    SHOW_IMAGE(dbIpmImage, "IPM image", 10);
  }//#endif

  //compute stop line width: 2000 mm
  FLOAT stopLinePixelWidth = stopLineConf->lineWidth *
      ipmInfo.xScale;
  //stop line pixel height: 12 inches = 12*25.4 mm
  FLOAT stopLinePixelHeight = stopLineConf->lineHeight  *
      ipmInfo.yScale;
  //kernel dimensions
  //unsigned char wx = 2;
  //unsigned char wy = 2;
  FLOAT sigmax = stopLinePixelWidth;
  FLOAT sigmay = stopLinePixelHeight;

//     //filter in the horizontal direction
//     CvMat * ipmt = cvCreateMat(ipm->width, ipm->height, ipm->type);
//     cvTranspose(ipm, ipmt);
//     mcvFilterLines(ipmt, ipmt, stopLineConf->kernelWidth,
// 		   stopLineConf->kernelHeight, sigmax, sigmay,
// 		   LINE_VERTICAL);
//     //retranspose
//     CvMat *ipm2 = cvCreateMat(ipm->height, ipm->width, ipm->type);
//     cvTranspose(ipmt, ipm2);
//     cvReleaseMat(&ipmt);

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
    CV_MAT_ELEM(*ipm, float, (*outPixelsi).y, (*outPixelsi).x) = 0;
  // 	CV_MAT_ELEM(*ipm2, float, (*outPixelsi).y, (*outPixelsi).x) = 0;
  }
  outPixels.clear();

#warning "Check this clearing of IPM image for 2 lanes"
  if (stopLineConf->ipmWindowClear)
  {
    //check to blank out other periferi of the image
    //blank from 60->100 (width 40)
    CvRect mask = cvRect(stopLineConf->ipmWindowLeft, 0,
                         stopLineConf->ipmWindowRight -
                         stopLineConf->ipmWindowLeft + 1,
                         ipm->height);
    mcvSetMat(ipm, mask, 0);
  }

  //show filtered image
  if (DEBUG_LINES) {
    SHOW_IMAGE(ipm, "Lane unthresholded filtered", 10);
  }

  //take the negative to get double yellow lines
  //cvScale(ipm, ipm, -1);

  CvMat *fipm = cvCloneMat(ipm);

    //zero out negative values
//     SHOW_MAT(fipm, "fipm");
#warning "clean negative parts in filtered image"
  mcvThresholdLower(ipm, ipm, 0);
//     mcvThresholdLower(ipm2, ipm2, 0);

//     //add the two images
//     cvAdd(ipm, ipm2, ipm);

//     //clear the horizontal filtered image
//     cvReleaseMat(&ipm2);

  //fipm was here
  //make copy of filteed ipm image

  vector <Line> dbIpmStopLines;
  vector<Spline> dbIpmSplines;

  //int numStrips = 2;
  int stripHeight = ipm->height / stopLineConf->numStrips;
  for (int i=0; i<stopLineConf->numStrips; i++) //lines
  {
    //get the mask
    CvRect mask;
    mask = cvRect(0, i*stripHeight, ipm->width,
            stripHeight);
  // 	SHOW_RECT(mask, "Mask");

    //get the subimage to work on
    CvMat *subimage = cvCloneMat(ipm);
    //clear all but the mask
    mcvSetMat(subimage, mask, 0);

    //compute quantile: .985
    FLOAT qtileThreshold = mcvGetQuantile(subimage, stopLineConf->lowerQuantile);
    mcvThresholdLower(subimage, subimage, qtileThreshold);
  // 	FLOAT qtileThreshold = mcvGetQuantile(ipm, stopLineConf->lowerQuantile);
  // 	mcvThresholdLower(ipm, ipm, qtileThreshold);

  //     qtileThreshold = mcvGetQuantile(ipm2, stopLineConf->lowerQuantile);
  //     mcvThresholdLower(ipm2, ipm2, qtileThreshold);

      //and fipm was here last
  //     //make copy of filtered ipm image
  //     CvMat *fipm = cvCloneMat(ipm);
    vector<Line> subimageLines;
    vector<Spline> subimageSplines;
    vector<float> subimageLineScores, subimageSplineScores;

	//check to blank out other periferi of the image
// 	mask = cvRect(40, 0, 80, subimage->height);
// 	mcvSetMat(subimage, mask, 0);
    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
	    CvMat *dbIpmImageThresholded;
	    dbIpmImageThresholded = cvCreateMat(ipm->height, ipm->width, ipm->type);
	    cvCopy(subimage, dbIpmImageThresholded);    //ipm
	    char str[256];
	    sprintf(str, "Lanes #%d thresholded IPM", i);
	    //thresholded ipm
	    SHOW_IMAGE(dbIpmImageThresholded, str, 10);
	    cvReleaseMat(&dbIpmImageThresholded);
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

	    CvMat *dbIpmImageThresholded;
	    dbIpmImageThresholded = cvCreateMat(ipm->height, ipm->width, ipm->type);
	    cvCopy(subimage, dbIpmImageThresholded);    //ipm
	    char str[256];
	    sprintf(str, "Lanes #%d thresholded IPM", i);
	    //thresholded ipm
	    SHOW_IMAGE(dbIpmImageThresholded, str, 10);
	    cvReleaseMat(&dbIpmImageThresholded);

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
    cvReleaseMat(&subimage);
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
    CvMat *imageClr = cvCreateMat(inImage->height, inImage->width, CV_32FC3);
    cvCvtColor(image, imageClr, CV_GRAY2RGB);
    //CvMat *image = cvCloneMat(inImage);
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
              cvPointFrom32f((*splines)[i].points[(*splines)[i].degree]),
              .5, CV_RGB(0, 0, 255));
      }

    SHOW_IMAGE(imageClr, "Detected lanes", 0);
    //cvReleaseMat(&image);
    cvReleaseMat(&dbIpmImage);
    //cvReleaseMat(&dbIpmImageThresholded);
    cvReleaseMat(&imageClr);
    dbIpmStopLines.clear();
    dbIpmSplines.clear();
  }//#endif //DEBUG_GET_STOP_LINES

  //clear
  cvReleaseMat(&ipm);
  cvReleaseMat(&image);
  cvReleaseMat(&fipm);
  cvReleaseMat(&rawipm);
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
void mcvPostprocessLines(const CvMat* image, const CvMat* clrImage,
                         const CvMat* rawipm, const CvMat* fipm,
                         vector<Line> &lines, vector<float> &lineScores,
                         vector<Spline> &splines, vector<float> &splineScores,
                         LaneDetectorConf *lineConf, LineState *state,
                         IPMInfo &ipmInfo, CameraInfo &cameraInfo)
{
  CvSize inSize = cvSize(image->width-1, image->height-1);

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
        CvMat *points = mcvEvalBezierSpline(splines[i], .1); //.05
        //mcvLocalizePoints(ipm, points, points); //inImage
        //extend spline
        CvMat* p = mcvExtendPoints(rawipm, points,
                lineConf->extendIPMAngleThreshold,
                lineConf->extendIPMMeanDirAngleThreshold,
                lineConf->extendIPMLinePixelsTangent,
                lineConf->extendIPMLinePixelsNormal,
                lineConf->extendIPMContThreshold,
                lineConf->extendIPMDeviationThreshold,
                cvRect(0, lineConf->extendIPMRectTop,
                  rawipm->width-1,
                  lineConf->extendIPMRectBottom-lineConf->extendIPMRectTop),
                false);
        //refit spline
        Spline spline = mcvFitBezierSpline(p, lineConf->ransacSplineDegree);

		//save
#warning "Check this later: extension in IPM. Check threshold value"
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
        cvReleaseMat(&points);
        cvReleaseMat(&p);
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
        CvMat *points = mcvEvalBezierSpline(splines[i], .05);
        // 	    CvMat *points = mcvGetBezierSplinePixels((*splines)[i], .05,
        // 						     cvSize(inImage->width-1,
        // 							    inImage->height-1),
        // 						     true);
        // 	    CvMat *p = cvCreateMat(points->height, points->width, CV_32FC1);
        // 	    cvConvert(points, p);
        mcvLocalizePoints(image, points, points, lineConf->localizeNumLinePixels,
              lineConf->localizeAngleThreshold); //inImage

        //get color
        CvMat* clrPoints = points;

        //extend spline
        CvMat* p = mcvExtendPoints(image, points,
                                   lineConf->extendAngleThreshold,
                                   lineConf->extendMeanDirAngleThreshold,
                                   lineConf->extendLinePixelsTangent,
                                   lineConf->extendLinePixelsNormal,
                                   lineConf->extendContThreshold,
                                   lineConf->extendDeviationThreshold,
                    cvRect(0, lineConf->extendRectTop,
                           image->width,
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
        cvReleaseMat(&points);
        cvReleaseMat(&p);

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
void mcvGetLines(const CvMat* image, LineType lineType,
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
	//FLOAT rMin = 0.05*ipm->height, rMax = 0.4*ipm->height, rStep = 1;
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
// 		      cvSize(image->width, image->height));
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
                               cvSize(image->width, image->height),
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

    CvMat* im = cvCreateMat(480, 640, CV_8UC3);
    cvSet(im, cvRealScalar(0.));
    //draw splines
    mcvDrawSpline(im, spline, CV_RGB(255, 0, 0), 1);
    SHOW_IMAGE(im, "Check Splines", 10);
    //clear
    cvReleaseMat(&im);
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
int mcvCheckPoints(const CvMat* points)
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

    CvMat* im = cvCreateMat(480, 640, CV_8UC3);
    cvSet(im, cvRealScalar(0.));
    //draw splines
    for (int i=0; i<points->height-1; i++)
    {
	    Line line;
	    line.startPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, i, 0),
                                     CV_MAT_ELEM(*points, float, i, 1));
	    line.endPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, i+1, 0),
                                   CV_MAT_ELEM(*points, float, i+1, 1));
	    mcvDrawLine(im, line, CV_RGB(255, 0, 0), 1);
    }
    SHOW_IMAGE(im, "Check Points", 0);
    //clear
    cvReleaseMat(&im);
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
void mcvLines2Mat(const vector<Line> *lines, CvMat *mat)
{
  //allocate the matrix
  //*mat = cvCreateMat(2, size*2, FLOAT_MAT_TYPE);

  //loop and put values
  int j;
  for (int i=0; i<(int)lines->size(); i++)
  {
    j = 2*i;
    CV_MAT_ELEM(*mat, FLOAT_MAT_ELEM_TYPE, 0, j) = (*lines)[i].startPoint.x;
    CV_MAT_ELEM(*mat, FLOAT_MAT_ELEM_TYPE, 1, j) = (*lines)[i].startPoint.y;
    CV_MAT_ELEM(*mat, FLOAT_MAT_ELEM_TYPE, 0, j+1) = (*lines)[i].endPoint.x;
    CV_MAT_ELEM(*mat, FLOAT_MAT_ELEM_TYPE, 1, j+1) = (*lines)[i].endPoint.y;
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
void mcvMat2Lines(const CvMat *mat, vector<Line> *lines)
{

  Line line;
  //loop and put values
  for (int i=0; i<int(mat->width/2); i++)
  {
    int j = 2*i;
    //get the line
    line.startPoint.x = CV_MAT_ELEM(*mat, FLOAT_MAT_ELEM_TYPE, 0, j);
    line.startPoint.y =  CV_MAT_ELEM(*mat, FLOAT_MAT_ELEM_TYPE, 1, j);
    line.endPoint.x = CV_MAT_ELEM(*mat, FLOAT_MAT_ELEM_TYPE, 0, j+1);
    line.endPoint.y = CV_MAT_ELEM(*mat, FLOAT_MAT_ELEM_TYPE, 1, j+1);
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
void mcvIntersectLineWithBB(const Line *inLine, const CvSize bbox,
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
    FLOAT deltax, deltay;
    deltax = inLine->endPoint.x - inLine->startPoint.x;
    deltay = inLine->endPoint.y - inLine->startPoint.y;
    //hold parameters
    FLOAT t[4]={2,2,2,2};
    FLOAT xup, xdown, yleft, yright;

    //intersect with top and bottom borders: y=0 and y=bbox.height-1
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
    FLOAT_POINT2D pts[4] = {{xup, 0},{xdown,bbox.height},
      {0, yleft},{bbox.width, yright}};

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
void mcvIntersectLineRThetaWithBB(FLOAT r, FLOAT theta, const CvSize bbox,
                                  Line *outLine)
{
  //hold parameters
  double xup, xdown, yleft, yright;

  //intersect with top and bottom borders: y=0 and y=bbox.height-1
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
  FLOAT_POINT2D pts[4] = {{xup, 0},{xdown,bbox.height},
        {0, yleft},{bbox.width, yright}};

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
bool mcvIsPointInside(FLOAT_POINT2D point, CvSize bbox)
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
void mcvIntersectLineRThetaWithRect(FLOAT r, FLOAT theta, const Line &rect,
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
  FLOAT_POINT2D pts[4] = {{xup, rect.startPoint.y},{xdown,rect.endPoint.y},
        {rect.startPoint.x, yleft},{rect.endPoint.x, yright}};

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
bool mcvIsPointInside(FLOAT_POINT2D &point, const CvRect &rect)
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
void mcvMatInt2Float(const CvMat *inMat, CvMat *outMat)
{
  for (int i=0; i<inMat->height; i++)
    for (int j=0; j<inMat->width; j++)
      CV_MAT_ELEM(*outMat, FLOAT_MAT_ELEM_TYPE, i, j) =
            (FLOAT_MAT_ELEM_TYPE) CV_MAT_ELEM(*inMat, INT_MAT_ELEM_TYPE,
                                              i, j)/255;
}


/** This function draws a line onto the passed image
 *
 * \param image the input iamge
 * \param line input line
 * \param line color
 * \param width line width
 *
 */
void mcvDrawLine(CvMat *image, Line line, CvScalar color, int width)
{
  cvLine(image, cvPoint((int)line.startPoint.x,(int)line.startPoint.y),
          cvPoint((int)line.endPoint.x,(int)line.endPoint.y),
          color, width);
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
  CvMat X = cvMat(3, 3, CV_64FC1, Xp);

  //array to hold the y values
  double yp[] = {val1, val2, val3};
  CvMat y = cvMat(3, 1, CV_64FC1, yp);

  //solve to get the coefficients
  double Ap[3];
  CvMat A = cvMat(3, 1, CV_64FC1, Ap);
  cvSolve(&X, &y, &A, CV_SVD);

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
CvMat * mcvGetLinePixels(const Line &line)
{
  //get two end points
  CvPoint start;
  start.x  = int(line.startPoint.x); start.y = int(line.startPoint.y);
  CvPoint end;
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
    CvPoint t = start;
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
  CvMat *pixels = cvCreateMat(end.x-start.x+1, 2, CV_32SC1);

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
    k = pixels->height-1;
    kupdate = -1;
  }

  for (i=start.x; i<=end.x; i++, k+=kupdate)
  {
    //put the new point
    if(steep)
    {
      CV_MAT_ELEM(*pixels, int, k, 0) = j;
      CV_MAT_ELEM(*pixels, int, k, 1) = i;
      // 	    x.push_back(j);
      // 	    y.push_back(i);
    }
    else
    {
      CV_MAT_ELEM(*pixels, int, k, 0) = i;
      CV_MAT_ELEM(*pixels, int, k, 1) = j;
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
void mcvGetLineExtent(const CvMat *im, const Line &inLine, Line &outLine)
{
  //first clip the input line to the image coordinates
  Line line = inLine;
  mcvIntersectLineWithBB(&inLine, cvSize(im->width-1, im->height-1), &line);

  //then get the pixel values of the line in the image
  CvMat *pixels; //vector<int> x, y;
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
  CvMat *pix = cvCreateMat(1, im->width, FLOAT_MAT_TYPE);
  CvMat *rstep = cvCreateMat(pix->height, pix->width, FLOAT_MAT_TYPE);
  CvMat *fstep = cvCreateMat(pix->height, pix->width, FLOAT_MAT_TYPE);
  for (int c=0; c<numChanges; c++)
  {
    //get the pixels
    //for(int i=0; i<(int)x.size(); i++)
    for(int i=0; i<pixels->height; i++)
    {
      CV_MAT_ELEM(*pix, FLOAT_MAT_ELEM_TYPE, 0, i) =
      CV_MAT_ELEM(*im, FLOAT_MAT_ELEM_TYPE,
                  changey ?
                  min(max(CV_MAT_ELEM(*pixels, int, i, 1)+
                  changes[c],0),im->height-1) :
                  CV_MAT_ELEM(*pixels, int, i, 1),
                  changey ? CV_MAT_ELEM(*pixels, int, i, 0) :
                  min(max(CV_MAT_ELEM(*pixels, int, i, 0)+
                  changes[c],0),im->width-1));
                  // 			    changey ? min(max(y[i]+changes[c],0),im->height-1) : y[i],
                  // 			    changey ? x[i] : min(max(x[i]+changes[c],0),im->width-1));
    }
    //remove the mean
    CvScalar mean = cvAvg(pix);
    cvSubS(pix, mean, pix);

    //now convolve with rising step to get start point
    FLOAT_MAT_ELEM_TYPE stepp[] = {-0.3000, -0.2, -0.1, 0, 0, 0.1, 0.2, 0.3, 0.4};
    // {-0.6, -0.4, -0.2, 0.2, 0.4, 0.6};
    int stepsize = 9;
    //{-0.2, -0.4, -0.2, 0, 0, 0.2, 0.4, 0.2}; //{-.75, -.5, .5, .75};
    CvMat step = cvMat(1, stepsize, FLOAT_MAT_TYPE, stepp);
    //	  SHOW_MAT(&step,"step");
    //smooth
    //	  FLOAT_MAT_ELEM_TYPE smoothp[] = {.25, .5, .25};
    //CvMat smooth = cvMat(1, 3, FLOAT_MAT_TYPE, smoothp);
    //cvFilter2D(&step, &step, &smooth);
    //SHOW_MAT(&step,"smoothed step");
    //convolve
    cvFilter2D(pix, rstep, &step);
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
    //cvFlip(&step, NULL, 1);
    //convolve
    //cvFilter2D(pix, fstep, &step);
    //get local max
    //     localMax.clear();
    //     localMaxLoc.clear();
    //     mcvGetVectorLocalMax(fstep, localMax, localMaxLoc);
    //     int endLoc = localMaxLoc[0];
    //take the negative
    cvConvertScale(rstep, fstep, -1);
    mcvGetVectorMax(fstep, &max, &endLoc, 0);
    //check if zero
    if(max==0)
      endLoc = endLocs[c-1];
    if(endLoc<=startLoc)
      endLoc = im->width-1;

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
  outLine.startPoint.x = CV_MAT_ELEM(*pixels, int, startLoc, 0);
  outLine.startPoint.y = CV_MAT_ELEM(*pixels, int, startLoc, 1);
  outLine.endPoint.x = CV_MAT_ELEM(*pixels, int, endLoc, 0);
  outLine.endPoint.y = CV_MAT_ELEM(*pixels, int, endLoc, 1);
  //     outLine.startPoint.x = x[startLoc]; outLine.startPoint.y = y[startLoc];
  //     outLine.endPoint.x = x[endLoc]; outLine.endPoint.y = y[endLoc];

  //clear
  cvReleaseMat(&pix);
  cvReleaseMat(&rstep);
  cvReleaseMat(&fstep);
  cvReleaseMat(&pixels);
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
void mcvFitRobustLine(const CvMat *points, float *lineRTheta,
                      float *lineAbc)
{
  // check number of points
  if (points->cols < 2) 
  {
    return;
  }
    
  //clone the points
  CvMat *cpoints = cvCloneMat(points);
  //get mean of the points and subtract from the original points
  float meanX=0, meanY=0;
  CvScalar mean;
  CvMat row1, row2;
  //get first row, compute avg and store
  cvGetRow(cpoints, &row1, 0);
  mean = cvAvg(&row1);
  meanX = (float) mean.val[0];
  cvSubS(&row1, mean, &row1);
  //same for second row
  cvGetRow(cpoints, &row2, 1);
  mean = cvAvg(&row2);
  meanY = (float) mean.val[0];
  cvSubS(&row2, mean, &row2);

  //compute the SVD for the centered points array
  CvMat *W = cvCreateMat(2, 1, CV_32FC1);
  CvMat *V = cvCreateMat(2, 2, CV_32FC1);
  //    CvMat *V = cvCreateMat(2, 2, CV_32fC1);
  CvMat *cpointst = cvCreateMat(cpoints->cols, cpoints->rows, CV_32FC1);

  cvTranspose(cpoints, cpointst);
  cvSVD(cpointst, W, 0, V, CV_SVD_V_T);
  cvTranspose(V, V);
  cvReleaseMat(&cpointst);

  //get the [a,b] which is the second column corresponding to
  //smaller singular value
  float a, b, c;
  a = CV_MAT_ELEM(*V, float, 0, 1);
  b = CV_MAT_ELEM(*V, float, 1, 1);

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
  cvReleaseMat(&cpoints);
  cvReleaseMat(&W);
  cvReleaseMat(&V);
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
void mcvFitRansacLine(const CvMat *image, int numSamples, int numIterations,
                      float threshold, float scoreThreshold, int numGoodFit,
                      bool getEndPoints, LineType lineType,
                      Line *lineXY, float *lineRTheta, float *lineScore)
{

  //get the points with non-zero pixels
  CvMat *points;
  points = mcvGetNonZeroPoints(image,true);
  if (!points)
    return;
  //check numSamples
  if (numSamples>points->cols)
    numSamples = points->cols;
  //subtract half
  cvAddS(points, cvRealScalar(0.5), points);

  //normalize pixels values to get weights of each non-zero point
  //get third row of points containing the pixel values
  CvMat w;
  cvGetRow(points, &w, 2);
  //normalize it
  CvMat *weights = cvCloneMat(&w);
  cvNormalize(weights, weights, 1, 0, CV_L1);
  //get cumulative    sum
  mcvCumSum(weights, weights);

  //random number generator
  CvRNG rng = cvRNG(0xffffffff);
  //matrix to hold random sample
  CvMat *randInd = cvCreateMat(numSamples, 1, CV_32SC1);
  CvMat *samplePoints = cvCreateMat(2, numSamples, CV_32FC1);
  //flag for points currently included in the set
  CvMat *pointIn = cvCreateMat(1, points->cols, CV_8SC1);
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
  CvPoint2D32f minp={-1., -1.}, maxp={-1., -1.};
  //outer loop
  for (int i=0; i<numIterations; i++)
  {
    //set flag to zero
    //cvSet(pointIn, cvRealScalar(0));
    cvSetZero(pointIn);
    //get random sample from the points
    #warning "Using weighted sampling for Ransac Line"
    // 	cvRandArr(&rng, randInd, CV_RAND_UNI, cvRealScalar(0), cvRealScalar(points->cols));
    mcvSampleWeighted(weights, numSamples, randInd, &rng);

    for (int j=0; j<numSamples; j++)
    {
      //flag it as included
      CV_MAT_ELEM(*pointIn, char, 0, CV_MAT_ELEM(*randInd, int, j, 0)) = 1;
      //put point
      CV_MAT_ELEM(*samplePoints, float, 0, j) =
      CV_MAT_ELEM(*points, float, 0, CV_MAT_ELEM(*randInd, int, j, 0));
      CV_MAT_ELEM(*samplePoints, float, 1, j) =
      CV_MAT_ELEM(*points, float, 1, CV_MAT_ELEM(*randInd, int, j, 0));
    }

    //fit the line
    mcvFitRobustLine(samplePoints, curLineRTheta, curLineAbc);

    //get end points from points in the samplePoints
    minc = 1e5; mind = 1e5; maxc = -1e5; maxd = -1e5;
    for (int j=0; getEndPoints && j<numSamples; ++j)
    {
      //get x & y
      x = CV_MAT_ELEM(*samplePoints, float, 0, j);
      y = CV_MAT_ELEM(*samplePoints, float, 1, j);

      //get the coordinate to work on
      if (lineType == LINE_HORIZONTAL)
        c = x;
      else if (lineType == LINE_VERTICAL)
        c = y;
      //compare
      if (c>maxc)
      {
        maxc = c;
        maxp = cvPoint2D32f(x, y);
      }
      if (c<minc)
      {
        minc = c;
        minp = cvPoint2D32f(x, y);
      }
    } //for

    // 	fprintf(stderr, "\nminx=%f, miny=%f\n", minp.x, minp.y);
    // 	fprintf(stderr, "maxp=%f, maxy=%f\n", maxp.x, maxp.y);

    //loop on other points and compute distance to the line
    score=0;
    for (int j=0; j<points->cols; j++)
    {
      // 	    //if not already inside
      // 	    if (!CV_MAT_ELEM(*pointIn, char, 0, j))
      // 	    {
        //compute distance to line
        dist = fabs(CV_MAT_ELEM(*points, float, 0, j) * curLineAbc[0] +
        CV_MAT_ELEM(*points, float, 1, j) * curLineAbc[1] + curLineAbc[2]);
        //check distance
        if (dist<=threshold)
        {
          //add this point
          CV_MAT_ELEM(*pointIn, char, 0, j) = 1;
          //update score
          score += cvGetReal2D(image, (int)(CV_MAT_ELEM(*points, float, 1, j)-.5),
                               (int)(CV_MAT_ELEM(*points, float, 0, j)-.5));
        }
        // 	    }
    }

    //check the number of close points and whether to consider this a good fit
    int numClose = cvCountNonZero(pointIn);
    //cout << "numClose=" << numClose << "\n";
    if (numClose >= numGoodFit)
    {
        //get the points included to fit this line
        CvMat *fitPoints = cvCreateMat(2, numClose, CV_32FC1);
        int k=0;
        //loop on points and copy points included
        for (int j=0; j<points->cols; j++)
      if(CV_MAT_ELEM(*pointIn, char, 0, j))
      {
          CV_MAT_ELEM(*fitPoints, float, 0, k) =
        CV_MAT_ELEM(*points, float, 0, j);
          CV_MAT_ELEM(*fitPoints, float, 1, k) =
        CV_MAT_ELEM(*points, float, 1, j);
          k++;

      }

      //fit the line
      mcvFitRobustLine(fitPoints, curLineRTheta, curLineAbc);

      //compute distances to new line
      dist = 0.;
      for (int j=0; j<fitPoints->cols; j++)
      {
        //compute distance to line
        x = CV_MAT_ELEM(*fitPoints, float, 0, j);
        y = CV_MAT_ELEM(*fitPoints, float, 1, j);
        float d = fabs( x * curLineAbc[0] +
        y * curLineAbc[1] +
        curLineAbc[2])
        * cvGetReal2D(image, (int)(y-.5), (int)(x-.5));
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
            // 			maxp = cvPoint2D32f(x, y);
            // 		    }
            // 		    if (c<minc)
            // 		    {
              // 			minc = c;
              // 			mind = d;
              // 			minp = cvPoint2D32f(x, y);
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
      cvReleaseMat(&fitPoints);

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
      CvMat* im = cvCloneMat(image);
      mcvScaleMat(image, im);
      CvMat *imageClr = cvCreateMat(image->rows, image->cols, CV_32FC3);
      cvCvtColor(im, imageClr, CV_GRAY2RGB);

      Line line;
      //draw current line if there
      if (curLineRTheta[0]>0)
      {
        mcvIntersectLineRThetaWithBB(curLineRTheta[0], curLineRTheta[1],
                                    cvSize(image->cols, image->rows), &line);
        mcvDrawLine(imageClr, line, CV_RGB(1,0,0), 1);
        if (getEndPoints)
          mcvDrawLine(imageClr, curEndPointLine, CV_RGB(0,1,0), 1);
      }

      //draw best line
      if (bestLineRTheta[0]>0)
      {
        mcvIntersectLineRThetaWithBB(bestLineRTheta[0], bestLineRTheta[1],
                                    cvSize(image->cols, image->rows), &line);
        mcvDrawLine(imageClr, line, CV_RGB(0,0,1), 1);
        if (getEndPoints)
          mcvDrawLine(imageClr, bestEndPointLine, CV_RGB(1,1,0), 1);
      }
      sprintf(str, "scor=%.2f, best=%.2f", score, bestScore);
      mcvDrawText(imageClr, str, cvPoint(30, 30), .25, CV_RGB(255,255,255));

      SHOW_IMAGE(imageClr, "Fit Ransac Line", 10);

      //clear
      cvReleaseMat(&im);
      cvReleaseMat(&imageClr);
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
                                   cvSize(image->cols-1, image->rows-1),
                                   lineXY);
  }
  if (lineScore)
    *lineScore = bestScore;

  //clear
  cvReleaseMat(&points);
  cvReleaseMat(&samplePoints);
  cvReleaseMat(&randInd);
  cvReleaseMat(&pointIn);
}




/** This function gets the indices of the non-zero values in a matrix
 *
 * \param inMat the input matrix
 * \param outMat the output matrix, with 2xN containing the x and y in
 *    each column and the pixels value [xs; ys; pixel values]
 * \param floatMat whether to return floating points or integers for
 *    the outMat
 */
CvMat* mcvGetNonZeroPoints(const CvMat *inMat, bool floatMat)
{


#define MCV_GET_NZ_POINTS(inMatType, outMatType) \
     /*loop and allocate the points*/ \
     for (int i=0; i<inMat->rows; i++) \
 	for (int j=0; j<inMat->cols; j++) \
 	    if (CV_MAT_ELEM(*inMat, inMatType, i, j)) \
 	    { \
 		CV_MAT_ELEM(*outMat, outMatType, 0, k) = j; \
 		CV_MAT_ELEM(*outMat, outMatType, 1, k) = i; \
                CV_MAT_ELEM(*outMat, outMatType, 2, k) = \
                  (outMatType) CV_MAT_ELEM(*inMat, inMatType, i, j); \
                k++; \
 	    } \

  int k=0;

  //get number of non-zero points
  int numnz = cvCountNonZero(inMat);

  //allocate the point array and get the points
  CvMat* outMat;
  if (numnz)
  {
    if (floatMat)
      outMat = cvCreateMat(3, numnz, CV_32FC1);
    else
      outMat = cvCreateMat(3, numnz, CV_32SC1);
  }
  else
    return NULL;

  //check type
  if (CV_MAT_TYPE(inMat->type)==FLOAT_MAT_TYPE &&
    CV_MAT_TYPE(outMat->type)==FLOAT_MAT_TYPE)
  {
    MCV_GET_NZ_POINTS(FLOAT_MAT_ELEM_TYPE, FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type)==FLOAT_MAT_TYPE &&
    CV_MAT_TYPE(outMat->type)==INT_MAT_TYPE)
  {
    MCV_GET_NZ_POINTS(FLOAT_MAT_ELEM_TYPE, INT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type)==INT_MAT_TYPE &&
    CV_MAT_TYPE(outMat->type)==FLOAT_MAT_TYPE)
  {
    MCV_GET_NZ_POINTS(INT_MAT_ELEM_TYPE, FLOAT_MAT_ELEM_TYPE)
  }
  else if (CV_MAT_TYPE(inMat->type)==INT_MAT_TYPE &&
    CV_MAT_TYPE(outMat->type)==INT_MAT_TYPE)
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
                   float groupThreshold, CvSize bbox)
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

    CvMat* im = cvCreateMat(240, 320, CV_8UC3);
    cvSet(im, cvRealScalar(0.));
    //draw splines
    for (unsigned int i=0; i<splines.size(); i++)
      mcvDrawSpline(im, splines[i], CV_RGB(255, 0, 0), 1);

    SHOW_IMAGE(im, "Splines Before grouping", 10);
    //clear
    cvReleaseMat(&im);

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

    CvMat* im = cvCreateMat(240, 320, CV_8UC3);
    cvSet(im, cvRealScalar(0.));
    //draw splines
    for (unsigned int i=0; i<splines.size(); i++)
      mcvDrawSpline(im, splines[i], CV_RGB(255, 0, 0), 1);

    SHOW_IMAGE(im, "Splines After grouping", 10);
    //clear
    cvReleaseMat(&im);

  }//#endif

}

/** \brief This function groups together bounding boxes
 *
 * \param size the size of image containing the lines
 * \param boxes a vector of output grouped bounding boxes
 * \param type the type of lines (LINE_HORIZONTAL or LINE_VERTICAL)
 * \param groupThreshold the threshold used for grouping (ratio of overlap)
 */
void mcvGroupBoundingBoxes(vector<CvRect> &boxes, LineType type,
                           float groupThreshold)
{
  bool cont = true;

  //Todo: check if to intersect with bounding box or not

  //save boxes
  //vector<CvRect> tboxes = boxes;

  //loop to get the largest overlap (according to type) and check
  //the overlap ratio
  float overlap, maxOverlap;
  while(cont)
  {
    maxOverlap =  overlap = -1e5;
    //loop on lines and get max overlap
    vector<CvRect>::iterator i, j, maxI, maxJ;
    for(i = boxes.begin(); i != boxes.end(); i++)
    {
      for(j = i+1; j != boxes.end(); j++)
      {
        switch(type)
        {
          case LINE_VERTICAL:
            //get one with smallest x, and compute the x2 - x1 / width of smallest
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
      *maxI  = cvRect(min((*maxI).x, (*maxJ).x),
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
void mcvGetRansacLines(const CvMat *im, vector<Line> &lines,
                       vector<float> &lineScores, LaneDetectorConf *lineConf,
                       LineType lineType)
{
  //check if to binarize image
  CvMat *image = cvCloneMat(im);
  if (lineConf->ransacLineBinarize)
    mcvBinarizeImage(image);

  int width = image->width-1;
  int height = image->height-1;
  //try grouping the lines into regions
  //float groupThreshold = 15;
  mcvGroupLines(lines, lineScores, lineConf->groupThreshold,
                cvSize(width, height));

  //group bounding boxes of lines
  float overlapThreshold = lineConf->overlapThreshold; //0.5; //.8;
  vector<CvRect> boxes;
  mcvGetLinesBoundingBoxes(lines, lineType, cvSize(width, height),
                           boxes);
  mcvGroupBoundingBoxes(boxes, lineType, overlapThreshold);
  //     mcvGroupLinesBoundingBoxes(lines, lineType, overlapThreshold,
  // 			       cvSize(width, height), boxes);

  //     //check if there're no lines, then check the whole image
  //     if (boxes.size()<1)
  // 	boxes.push_back(cvRect(0, 0, width-1, height-1));

  int window = lineConf->ransacLineWindow; //15;
  vector<Line> newLines;
  vector<float> newScores;
  for (int i=0; i<(int)boxes.size(); i++) //lines
  {
    // 	fprintf(stderr, "i=%d\n", i);
    //Line line = lines[i];
    CvRect mask, box;
    //get box
    box = boxes[i];
    switch (lineType)
    {
      case LINE_HORIZONTAL:
      {
        //get extent
        //int ystart = (int)fmax(fmin(line.startPoint.y, line.endPoint.y)-window, 0);
        //int yend = (int)fmin(fmax(line.startPoint.y, line.endPoint.y)+window, height-1);
        int ystart = (int)fmax(box.y - window, 0);
        int yend = (int)fmin(box.y + box.height + window, height-1);
        //get the mask
        mask = cvRect(0, ystart, width, yend-ystart+1);
      }
      break;

      case LINE_VERTICAL:
      {
        //get extent of window to search in
        //int xstart = (int)fmax(fmin(line.startPoint.x, line.endPoint.x)-window, 0);
        //int xend = (int)fmin(fmax(line.startPoint.x, line.endPoint.x)+window, width-1);
        int xstart = (int)fmax(box.x - window, 0);
        int xend = (int)fmin(box.x + box.width + window, width-1);
        //get the mask
        mask = cvRect(xstart, 0, xend-xstart+1, height);
      }
      break;
    }
    //get the subimage to work on
    CvMat *subimage = cvCloneMat(image);
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
    #warning "check this screening in ransacLines"
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
      CvMat *subimageClr = cvCreateMat(subimage->rows, subimage->cols,
                                       CV_32FC3);
      cvCvtColor(subimage, subimageClr, CV_GRAY2RGB);
      //draw rectangle
      //      	    mcvDrawRectangle(subimageClr, box,
                  // 			     CV_RGB(255, 255, 0), 1);
      mcvDrawRectangle(subimageClr, mask, CV_RGB(255, 255, 255), 1);

      //draw line
      if (lineRTheta[0]>0)
        mcvDrawLine(subimageClr, line, CV_RGB(1,0,0), 1);
      SHOW_IMAGE(subimageClr, str, 10);
      //clear
      cvReleaseMat(&subimageClr);
    }//#endif

    //clear
    cvReleaseMat(&subimage);
  } // for i

  //group lines
  vector<Line> oldLines;
  if (DEBUG_LINES)
    oldLines = lines;
  lines.clear();
  lineScores.clear();
  #warning "not grouping at end of getRansacLines"
  //mcvGroupLines(newLines, newScores, lineConf->groupThreshold, cvSize(width, height));
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
    CvMat* im2 = cvCloneMat(im);
    mcvScaleMat(im2, im2);
    CvMat *imClr = cvCreateMat(im->rows, im->cols, CV_32FC3);
    cvCvtColor(im2, imClr, CV_GRAY2RGB);
    CvMat* imClr2 = cvCloneMat(imClr);
    cvReleaseMat(&im2);

    //draw spline
    for (unsigned int j=0; j<lines.size(); j++)
      mcvDrawLine(imClr, lines[j], CV_RGB(0,1,0), 1);
    SHOW_IMAGE(imClr, title, 10);

    //draw spline
    for (unsigned int j=0; j<oldLines.size(); j++)
      mcvDrawLine(imClr2, oldLines[j], CV_RGB(1,0,0), 1);
    SHOW_IMAGE(imClr2, "Input Lines", 10);

    //clear
    cvReleaseMat(&imClr);
    cvReleaseMat(&imClr2);
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
  cvReleaseMat(&image);
}

/** This function sets the matrix to a value except for the mask window passed in
 *
 * \param inMat input matrix
 * \param mask the rectangle defining the mask: (xleft, ytop, width, height)
 * \param val the value to put
 */
void  mcvSetMat(CvMat *inMat, CvRect mask, double val)
{

  //get x-end points of region to work on, and work on the whole image height
  //(int)fmax(fmin(line.startPoint.x, line.endPoint.x)-xwindow, 0);
  int xstart = mask.x, xend = mask.x + mask.width-1;
  //xend = (int)fmin(fmax(line.startPoint.x, line.endPoint.x), width-1);
  int ystart = mask.y, yend = mask.y + mask.height-1;

  //set other two windows to zero
  CvMat maskMat;
  CvRect rect;
  //part to the left of required region
  rect = cvRect(0, 0, xstart-1, inMat->height);
  if (rect.x<inMat->width && rect.y<inMat->height &&
    rect.x>=0 && rect.y>=0 && rect.width>0 && rect.height>0)
  {
    cvGetSubRect(inMat, &maskMat, rect);
    cvSet(&maskMat, cvRealScalar(val));
  }
  //part to the right of required region
  rect = cvRect(xend+1, 0, inMat->width-xend-1, inMat->height);
  if (rect.x<inMat->width && rect.y<inMat->height &&
    rect.x>=0 && rect.y>=0 && rect.width>0 && rect.height>0)
  {
    cvGetSubRect(inMat, &maskMat, rect);
    cvSet(&maskMat, cvRealScalar(val));
  }

  //part to the top
  rect = cvRect(xstart, 0, mask.width, ystart-1);
  if (rect.x<inMat->width && rect.y<inMat->height &&
    rect.x>=0 && rect.y>=0 && rect.width>0 && rect.height>0)
  {
    cvGetSubRect(inMat, &maskMat, rect);
    cvSet(&maskMat, cvRealScalar(val));
  }

  //part to the bottom
  rect = cvRect(xstart, yend+1, mask.width, inMat->height-yend-1);
  if (rect.x<inMat->width && rect.y<inMat->height &&
    rect.x>=0 && rect.y>=0 && rect.width>0 && rect.height>0)
  {
    cvGetSubRect(inMat, &maskMat, rect);
    cvSet(&maskMat, cvRealScalar(val));
  }
}


/** This function sorts a set of points
 *
 * \param inPOints Nx2 matrix of points [x,y]
 * \param outPOints Nx2 matrix of points [x,y]
 * \param dim the dimension to sort on (0: x, 1:y)
 * \param dir direction of sorting (0: ascending, 1:descending)
 */
void mcvSortPoints(const CvMat *inPoints, CvMat *outPoints,
                   int dim, int dir)
{
  //make a copy of the input
  CvMat *pts = cvCloneMat(inPoints);

  //clear the output
  //cvSetZero(outPoints);

  //make the list of sorted indices
  list<int> sorted;
  list<int>::iterator sortedi;
  int i, j;

  //loop on elements and adjust its index
  for (i=0; i<pts->height; i++)
  {
    //if ascending
    if (dir==0)
      for (sortedi = sorted.begin();
           sortedi != sorted.end() &&
           (CV_MAT_ELEM(*pts, float, i, dim) >=
           CV_MAT_ELEM(*outPoints, float, *sortedi, dim));
           sortedi++);
    //descending
    else
      for (sortedi = sorted.begin();
           sortedi != sorted.end() &&
           (CV_MAT_ELEM(*pts, float, i, dim) <=
           CV_MAT_ELEM(*outPoints, float, *sortedi, dim));
           sortedi++);

    //found the position, so put it into sorted
    sorted.insert(sortedi, i);
  }

  //sorted the array, so put back
  for (i=0, sortedi=sorted.begin(); sortedi != sorted.end(); sortedi++, i++)
    for(j=0; j<outPoints->width; j++)
      CV_MAT_ELEM(*outPoints, float, i, j) = CV_MAT_ELEM(*pts, float,
                                                         *sortedi, j);

  //clear
  cvReleaseMat(&pts);
  sorted.clear();
}

/** This function fits a Bezier spline to the passed input points
 *
 * \param points the input points
 * \param degree the required spline degree
 * \return spline the returned spline
 */
Spline mcvFitBezierSpline(CvMat *points, int degree)
{

  //set the degree
  Spline spline;
  spline.degree = degree;

  //get number of points
  int n = points->height;
  //float step = 1./(n-1);

  //sort the pointa
  mcvSortPoints(points, points, 1, 0);
  //     SHOW_MAT(points, "Points after sorting:");

  //get first point and distance between points
  CvPoint2D32f  p0 = cvPoint2D32f(CV_MAT_ELEM(*points, float, 0, 0),
                                  CV_MAT_ELEM(*points, float, 0, 1));

  float diff = 0.f;
  float *us = new float[points->height];
  us[0] = 0;
  for (int i=1; i<points->height; ++i)
  {
    float dx = CV_MAT_ELEM(*points, float, i, 0) -
      CV_MAT_ELEM(*points, float, i-1, 0);
    float dy = CV_MAT_ELEM(*points, float, i, 1) -
      CV_MAT_ELEM(*points, float, i-1, 1);
    us[i] = cvSqrt(dx*dx + dy*dy) + us[i-1];
    // 	diff += us[i];;
  }
  diff = us[points->height-1];

  //float y0 = CV_MAT_ELEM(*points, float, 0, 1);
  //float ydiff = CV_MAT_ELEM(*points, float, points->height-1, 1) - y0;

  //M matrices: M2 for quadratic (degree 2) and M3 for cubic
  float M2[] = {1, -2, 1,
                -2, 2, 0,
                1, 0, 0};
  float M3[] = {-1, 3, -3, 1,
                3, -6, 3, 0,
                -3, 3, 0, 0,
                1, 0, 0, 0};

  //M matrix for Bezier
  CvMat M;

  //Basis matrix
  CvMat *B;

  //u value for points to create the basis matrix
  float u = 0.f;

  //switch on the degree
  switch(degree)
  {
    //Quadratic spline
    case 2:
      //M matrix
      M = cvMat(3, 3, CV_32FC1, M2);

      //create the basis matrix
      B = cvCreateMat(n, 3, CV_32FC1);
      for (int i=0; i<B->height; i++) //u+=step
      {
        //get u as ratio of y-coordinate
        // 	    u  = i / ((float)n-1);

        //  	    u = (CV_MAT_ELEM(*points, float, i, 1) - y0) / ydiff;

        // 	    float dx = CV_MAT_ELEM(*points, float, i, 0) - p0.x;
        // 	    float dy = CV_MAT_ELEM(*points, float, i, 1) - p0.y;
        // 	    u = cvSqrt(dx*dx + dy*dy) / diff;
        u = us[i] / diff;

        CV_MAT_ELEM(*B, float, i, 2) = 1;  //1
        CV_MAT_ELEM(*B, float, i, 1) = u;  //u
        CV_MAT_ELEM(*B, float, i, 0) = u*u;  //u^2
      }
      break;

    //Cubic spline
    case 3:
      //M matrix
      M = cvMat(4, 4, CV_32FC1, M3);

      //create the basis matrix
      B = cvCreateMat(n, 4, CV_32FC1);
      for (int i=0; i<B->height; i++) //, u+=step)
      {
        //get u as ratio of y-coordinate
        // 	    u  = i / ((float)n-1);

        //  	    u = (CV_MAT_ELEM(*points, float, i, 1) - y0) / ydiff;

        // 	    float dx = CV_MAT_ELEM(*points, float, i, 0) - p0.x;
        // 	    float dy = CV_MAT_ELEM(*points, float, i, 1) - p0.y;
        // 	    u = cvSqrt(dx*dx + dy*dy) / diff;
        u = us[i] / diff;

        CV_MAT_ELEM(*B, float, i, 3) = 1;  //1
        CV_MAT_ELEM(*B, float, i, 2) = u;  //u
        CV_MAT_ELEM(*B, float, i, 1) = u*u;  //u^2
        CV_MAT_ELEM(*B, float, i, 0) = u*u*u;  //u^2
      }
      break;
  } // switch degree

  //multiply B by M
  cvMatMul(B, &M, B);


  //return the required control points by LS
  CvMat *sp = cvCreateMat(degree+1, 2, CV_32FC1);
  cvSolve(B, points, sp, CV_SVD);

  //     SHOW_MAT(sp, "Spline points:");

  //put back into spline
  memcpy((float *)spline.points, sp->data.fl, sizeof(float)*(spline.degree+1)*2);
  //     if(spline.points[0].x<0)
  // 	SHOW_MAT(points, "INput Points");

  //clear
  cvReleaseMat(&B);
  cvReleaseMat(&sp);
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
CvMat* mcvEvalBezierSpline(const Spline &spline, float h, CvMat *tangents)
{
  //compute number of points to return
  int n = (int)(1./h)+1;

  //allocate the points
  CvMat *points = cvCreateMat(n, 2, CV_32FC1);

  //M matrices
  CvMat M;
  float M2[] = {1, -2, 1,
  -2, 2, 0,
  1, 0, 0};
  float M3[] = {-1, 3, -3, 1,
  3, -6, 3, 0,
  -3, 3, 0, 0,
  1, 0, 0, 0};

  //spline points
  CvMat *sp = cvCreateMat(spline.degree+1, 2, CV_32FC1);
  memcpy(sp->data.fl, (float *)spline.points,
         sizeof(float)*(spline.degree+1)*2);

  //abcd
  CvMat *abcd;

  float P[2], dP[2], ddP[2], dddP[2];
  float h2 = h*h, h3 = h2*h;

  //switch the degree
  switch(spline.degree)
  {
    //Quadratic
    case 2:
      //get M matrix
      M = cvMat(3, 3, CV_32FC1, M2);

      //get abcd where a=row 0, b=row 1, ...
      abcd = cvCreateMat(3, 2, CV_32FC1);
      cvMatMul(&M, sp, abcd);

      //P = c
      P[0] = CV_MAT_ELEM(*abcd, float, 2, 0);
      P[1] = CV_MAT_ELEM(*abcd, float, 2, 1);

      //dP = b*h+a*h^2
      dP[0] = CV_MAT_ELEM(*abcd, float, 1, 0)*h +
      CV_MAT_ELEM(*abcd, float, 0, 0)*h2;
      dP[1] = CV_MAT_ELEM(*abcd, float, 1, 1)*h +
      CV_MAT_ELEM(*abcd, float, 0, 1)*h2;

      //ddP = 2*a*h^2
      ddP[0] = 2 * CV_MAT_ELEM(*abcd, float, 0, 0)*h2;
      ddP[1] = 2 * CV_MAT_ELEM(*abcd, float, 0, 1)*h2;

      //loop and put points
      for (int i=0; i<n; i++)
      {
        //put point
        CV_MAT_ELEM(*points, float, i, 0) = P[0];
        CV_MAT_ELEM(*points, float, i, 1) = P[1];

        //update
        P[0] += dP[0]; P[1] += dP[1];
        dP[0] += ddP[0]; dP[1] += ddP[1];
      }

      //put tangents
      if (tangents)
      {
        //t0 = b
        CV_MAT_ELEM(*tangents, float, 0, 0) =
        CV_MAT_ELEM(*abcd, float, 1, 0);
        CV_MAT_ELEM(*tangents, float, 0, 1) =
        CV_MAT_ELEM(*abcd, float, 1, 1);
        //t1 = 2*a + b
        CV_MAT_ELEM(*tangents, float, 1, 0) = 2 *
        CV_MAT_ELEM(*abcd, float, 0, 0) +
        CV_MAT_ELEM(*abcd, float, 1, 0);
        CV_MAT_ELEM(*tangents, float, 1, 1) = 2 *
        CV_MAT_ELEM(*abcd, float, 0, 1) +
        CV_MAT_ELEM(*abcd, float, 1, 1);
      }
      break;

    /*Cubic*/
    case 3:
      //get M matrix
      M = cvMat(4, 4, CV_32FC1, M3);

      //get abcd where a=row 0, b=row 1, ...
      abcd = cvCreateMat(4, 2, CV_32FC1);
      cvMatMul(&M, sp, abcd);

      //P = d
      P[0] = CV_MAT_ELEM(*abcd, float, 3, 0);
      P[1] = CV_MAT_ELEM(*abcd, float, 3, 1);

      //dP = c*h + b*h^2+a*h^3
      dP[0] = CV_MAT_ELEM(*abcd, float, 2, 0)*h +
      CV_MAT_ELEM(*abcd, float, 1, 0)*h2 +
      CV_MAT_ELEM(*abcd, float, 0, 0)*h3;
      dP[1] = CV_MAT_ELEM(*abcd, float, 2, 1)*h +
      CV_MAT_ELEM(*abcd, float, 1, 1)*h2 +
      CV_MAT_ELEM(*abcd, float, 0, 1)*h3;

      //dddP = 6 * a * h3
      dddP[0] = 6 * CV_MAT_ELEM(*abcd, float, 0, 0) * h3;
      dddP[1] = 6 * CV_MAT_ELEM(*abcd, float, 0, 1) * h3;

      //ddP = 2*b*h2 + 6*a*h3
      ddP[0] = 2 * CV_MAT_ELEM(*abcd, float, 1, 0) * h2 + dddP[0];
      ddP[1] = 2 * CV_MAT_ELEM(*abcd, float, 1, 1) * h2 + dddP[1];

      //loop and put points
      for (int i=0; i<n; i++)
      {
        //put point
        CV_MAT_ELEM(*points, float, i, 0) = P[0];
        CV_MAT_ELEM(*points, float, i, 1) = P[1];

        //update
        P[0] += dP[0]; P[1] += dP[1];
        dP[0] += ddP[0]; dP[1] += ddP[1];
        ddP[0] += dddP[0]; ddP[1] += dddP[1];
      }

      //put tangents
      if (tangents)
      {
        //t0 = c
        CV_MAT_ELEM(*tangents, float, 0, 0) = CV_MAT_ELEM(*abcd, float, 2, 0);
        CV_MAT_ELEM(*tangents, float, 0, 1) = CV_MAT_ELEM(*abcd, float, 2, 1);
        //t1 = 3*a + 2*b + c
        CV_MAT_ELEM(*tangents, float, 1, 0) =
          3 * CV_MAT_ELEM(*abcd, float, 0, 0) +
          2 * CV_MAT_ELEM(*abcd, float, 1, 0) +
          CV_MAT_ELEM(*abcd, float, 2, 0);
        CV_MAT_ELEM(*tangents, float, 1, 1) =
          3 * CV_MAT_ELEM(*abcd, float, 0, 1) +
          2 * CV_MAT_ELEM(*abcd, float, 1, 1) +
          CV_MAT_ELEM(*abcd, float, 2, 1);
      }
      break;
  }

  //clear
  cvReleaseMat(&abcd);
  cvReleaseMat(&sp);

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
 * \return computed points in an array Nx2 [x,y], returns NULL if empty output
 */
CvMat* mcvGetBezierSplinePixels(Spline &spline, float h, CvSize box,
                                bool extendSpline)
{
  //get the points belonging to the spline
  CvMat *tangents = cvCreateMat(2, 2, CV_32FC1);
  CvMat * points = mcvEvalBezierSpline(spline, h, tangents);

  //pixelize the spline
  //CvMat *inpoints = cvCreateMat(points->height, 1, CV_8SC1);
  //cvSet(, CvScalar value, const CvArr* mask=NULL);
  list<int> inpoints;
  list<int>::iterator inpointsi;
  int lastin = -1, numin = 0;
  for (int i=0; i<points->height; i++)
  {
    //round
    CV_MAT_ELEM(*points, float, i, 0) = cvRound(CV_MAT_ELEM(*points, float, i, 0));
    CV_MAT_ELEM(*points, float, i, 1) = cvRound(CV_MAT_ELEM(*points, float, i, 1));

    //check boundaries
    if(CV_MAT_ELEM(*points, float, i, 0) >= 0 &&
      CV_MAT_ELEM(*points, float, i, 0) < box.width &&
      CV_MAT_ELEM(*points, float, i, 1) >= 0 &&
      CV_MAT_ELEM(*points, float, i, 1) < box.height)
    {
      //it's inside, so check if the same as last one
      if(lastin<0 ||
        (lastin>=0 &&
        !(CV_MAT_ELEM(*points, float, lastin, 1)==
        CV_MAT_ELEM(*points, float, i, 1) &&
        CV_MAT_ELEM(*points, float, lastin, 0)==
        CV_MAT_ELEM(*points, float, i, 0) )) )
      {
        //put inside
        //CV_MAT_ELEM(*inpoints, char, i, 0) = 1;
        inpoints.push_back(i);
        lastin = i;
        numin++;
      }
    }
  }

  //check if to extend the spline with lines
  CvMat *pixelst0, *pixelst1;
  if (extendSpline)
  {
    //get first point inside
    int p0 = inpoints.front();
    //extend from the starting point by going backwards along the tangent
    //line from that point to the start of spline
    Line line;
    line.startPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, p0, 0) - 10 *
                                   CV_MAT_ELEM(*tangents, float, 0, 0),
                                   CV_MAT_ELEM(*points, float, p0, 1) - 10 *
                                   CV_MAT_ELEM(*tangents, float, 0, 1));
    line.endPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, p0, 0),
                                 CV_MAT_ELEM(*points, float, p0, 1));
    //intersect the line with the bounding box
    mcvIntersectLineWithBB(&line, cvSize(box.width-1, box.height-1), &line);
    //get line pixels
    pixelst0 = mcvGetLinePixels(line);
    numin += pixelst0->height;

    //get last point inside
    int p1 = inpoints.back();
    //extend from end of spline along tangent
    line.endPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, p1, 0) + 10 *
                                 CV_MAT_ELEM(*tangents, float, 1, 0),
                                 CV_MAT_ELEM(*points, float, p1, 1) + 10 *
                                 CV_MAT_ELEM(*tangents, float, 1, 1));
    line.startPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, p1, 0),
                                   CV_MAT_ELEM(*points, float, p1, 1));
    //intersect the line with the bounding box
    mcvIntersectLineWithBB(&line, cvSize(box.width-1, box.height-1), &line);
    //get line pixels
    pixelst1 = mcvGetLinePixels(line);
    numin += pixelst1->height;
  }

  //put the results in another matrix
  CvMat *rpoints;
  if (numin>0)
    rpoints = cvCreateMat(numin, 2, CV_32SC1);
  else
  {
    return NULL;
  }


  //first put extended line segment if available
  if(extendSpline)
  {
    //copy
    memcpy(cvPtr2D(rpoints, 0, 0), pixelst0->data.fl,
           sizeof(float)*2*pixelst0->height);
  }

  //put spline pixels
  int ri = extendSpline ? pixelst0->height : 0;
  for (inpointsi=inpoints.begin();
  inpointsi!=inpoints.end(); ri++, inpointsi++)
  {
    CV_MAT_ELEM(*rpoints, int, ri, 0) = (int)CV_MAT_ELEM(*points,
                                                         float, *inpointsi, 0);
    CV_MAT_ELEM(*rpoints, int, ri, 1) = (int)CV_MAT_ELEM(*points,
                                                         float, *inpointsi, 1);
  }

  //put second extended piece of spline
  if(extendSpline)
  {
    //copy
    memcpy(cvPtr2D(rpoints, ri, 0), pixelst1->data.fl,
           sizeof(float)*2*pixelst1->height);
    //clear
    cvReleaseMat(&pixelst0);
    cvReleaseMat(&pixelst1);
  }


  //release
  //    cvReleaseMat(&inpoints);
  cvReleaseMat(&points);
  cvReleaseMat(&tangents);
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
void mcvGetRansacSplines(const CvMat *im, vector<Line> &lines,
                         vector<float> &lineScores, LaneDetectorConf *lineConf,
                         LineType lineType, vector<Spline> &splines,
                         vector<float> &splineScores, LineState* state)
{
  //check if to binarize image
  CvMat *image = cvCloneMat(im);
  if (lineConf->ransacSplineBinarize)
    mcvBinarizeImage(image); // ((topmost-intro . 147431))

    int width = image->width;
  int height = image->height;
  //try grouping the lines into regions
  //float groupThreshold = 15;
  #warning "no line grouping in getRansacSplines"
  vector<Line> tlines = lines;
  vector<float> tlineScores = lineScores;
  mcvGroupLines(tlines, tlineScores, lineConf->groupThreshold,
                cvSize(width-1, height-1));

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
  vector<CvRect> boxes;
  CvSize size = cvSize(width, height);
  mcvGetLinesBoundingBoxes(tlines, lineType, size, boxes);
  mcvGroupBoundingBoxes(boxes, lineType, overlapThreshold);
  //     mcvGroupLinesBoundingBoxes(tlines, lineType, overlapThreshold,
  // 			       cvSize(width, height), boxes);
  tlines.clear();
  tlineScores.clear();

  //add bounding boxes from previous frame
  #warning "Turned off adding boxes from previous frame"
  //     boxes.insert(boxes.end(), state->ipmBoxes.begin(),
  // 		 state->ipmBoxes.end());

  //     //check if there're no lines, then check the whole image
  //     if (boxes.size()<1)
  // 	boxes.push_back(cvRect(0, 0, width-1, height-1));

  int window = lineConf->ransacSplineWindow; //15;
  vector<Spline> newSplines;
  vector<float> newSplineScores;
  for (int i=0; i<(int)boxes.size(); i++) //lines
  {
    //Line line = lines[i];

    CvRect mask, box;

    //get box
    box = boxes[i];

    switch (lineType)
    {
      case LINE_HORIZONTAL:
      {
        //get extent
        //int ystart = (int)fmax(fmin(line.startPoint.y, line.endPoint.y)-window, 0);
        //int yend = (int)fmin(fmax(line.startPoint.y, line.endPoint.y)+window, height-1);
        int ystart = (int)fmax(box.y - window, 0);
        int yend = (int)fmin(box.y + box.height + window, height-1);
        //get the mask
        mask = cvRect(0, ystart, width, yend-ystart+1);
      }
      break;

      case LINE_VERTICAL:
      {
        //get extent of window to search in
        //int xstart = (int)fmax(fmin(line.startPoint.x, line.endPoint.x)-window, 0);
        //int xend = (int)fmin(fmax(line.startPoint.x, line.endPoint.x)+window, width-1);
        int xstart = (int)fmax(box.x - window, 0);
        int xend = (int)fmin(box.x + box.width + window, width-1);
        //get the mask
        mask = cvRect(xstart, 0, xend-xstart+1, height);
      }
      break;
    }
    //get the subimage to work on
    CvMat *subimage = cvCloneMat(image);
    //clear all but the mask
    mcvSetMat(subimage, mask, 0);

    //get the RANSAC spline in this part
    //int numSamples = 5, numIterations = 10, numGoodFit = 15;
    //float threshold = 0.5;
    Spline spline;
    float splineScore;
    //resolution to use in pixelizing the spline
    float h = lineConf->ransacSplineStep; // .1; //1. / max(image->width, image->height);
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
      CvMat *subimageClr = cvCreateMat(subimage->rows, subimage->cols,
                                       CV_32FC3);
      cvCvtColor(subimage, subimageClr, CV_GRAY2RGB);

      //draw rectangle
      //mcvDrawRectangle(subimageClr, box,
      //	     CV_RGB(255, 255, 0), 1);
      mcvDrawRectangle(subimageClr, mask, CV_RGB(255, 255, 255), 1);

      //put text
      sprintf(str, "score=%.2f", splineScore);
      // 	    mcvDrawText(subimageClr, str, cvPoint(30, 30),
      // 			.25f, CV_RGB(1,1,1));

      //draw spline
      if (spline.degree > 0)
        mcvDrawSpline(subimageClr, spline, CV_RGB(1,0,0), 1);
      SHOW_IMAGE(subimageClr, title, 10);
      //clear
      cvReleaseMat(&subimageClr);
    }//#endif

    //clear
    cvReleaseMat(&subimage);
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
    CvMat* im2 = cvCloneMat(im);
    mcvScaleMat(im2, im2);
    CvMat *imClr = cvCreateMat(im->rows, im->cols, CV_32FC3);
    cvCvtColor(im2, imClr, CV_GRAY2RGB);
    cvReleaseMat(&im2);

    //draw spline
    for (unsigned int j=0; j<splines.size(); j++)
      mcvDrawSpline(imClr, splines[j], CV_RGB(0,1,0), 1);
    SHOW_IMAGE(imClr, title, 10);
    //clear
    cvReleaseMat(&imClr);
  }//#endif


  //clean
  boxes.clear();
  newSplines.clear();
  newSplineScores.clear();
  cvReleaseMat(&image);
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
 *   pass NULL to ignore this input
 *
 */
void mcvFitRansacSpline(const CvMat *image, int numSamples, int numIterations,
                        float threshold, float scoreThreshold, int numGoodFit,
                        int splineDegree, float h, Spline *spline,
                        float *splineScore, int splineScoreJitter,
                        float splineScoreLengthRatio,
                        float splineScoreAngleRatio, float splineScoreStep,
                        vector<Spline> *prevSplines)
{
  //get the points with non-zero pixels
  CvMat *points = mcvGetNonZeroPoints(image, true);
  if (points==0 || points->cols < numSamples)
  {
    if (spline) spline->degree = -1;
    cvReleaseMat(&points);
    return;
  }
  //     fprintf(stderr, "num points=%d", points->cols);
  //subtract half
  #warning "check adding half to points"
  CvMat p;
  cvGetRows(points, &p, 0, 2);
  cvAddS(&p, cvRealScalar(0.5), &p);

  //normalize pixels values to get weights of each non-zero point
  //get third row of points containing the pixel values
  CvMat w;
  cvGetRow(points, &w, 2);
  //normalize it
  CvMat *weights = cvCloneMat(&w);
  cvNormalize(weights, weights, 1, 0, CV_L1);
  //get cumulative    sum
  mcvCumSum(weights, weights);

  //random number generator
  CvRNG rng = cvRNG(0xffffffff);
  //matrix to hold random sample
  CvMat *randInd = cvCreateMat(numSamples, 1, CV_32SC1);
  CvMat *samplePoints = cvCreateMat(numSamples, 2, CV_32FC1);
  //flag for points currently included in the set
  CvMat *pointIn = cvCreateMat(1, points->cols, CV_8SC1);
  //returned splines
  Spline curSpline, bestSpline;
  bestSpline.degree = 0;//initialize
  float bestScore=0; //, bestDist=1e5;

  //iterator for previous splines
  vector<Spline>::iterator prevSpline;
  bool randSpline = prevSplines==NULL || prevSplines->size()==0;
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
      cvSetZero(pointIn);
      //get random sample from the points
      //cvRandArr(&rng, randInd, CV_RAND_UNI, cvRealScalar(0), cvRealScalar(points->cols));
      mcvSampleWeighted(weights, numSamples, randInd, &rng);
      // 	    SHOW_MAT(randInd, "randInd");
      for (int j=0; j<randInd->rows; j++) //numSamples
      {
        //flag it as included
        int p = CV_MAT_ELEM(*randInd, int, j, 0);
        CV_MAT_ELEM(*pointIn, char, 0, p) = 1;
        //put point
        CV_MAT_ELEM(*samplePoints, float, j, 0) =
        CV_MAT_ELEM(*points, float, 0, p);
        CV_MAT_ELEM(*samplePoints, float, j, 1) =
        CV_MAT_ELEM(*points, float, 1, p);
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
      CvMat *imageClr = cvCreateMat(image->rows, image->cols,
                                    CV_32FC3);
      CvMat *im = cvCloneMat(image);
      mcvScaleMat(image, im);
      cvCvtColor(im, imageClr, CV_GRAY2RGB);
	    //draw spline
	    //previous splines
 	    for (unsigned int k=0; prevSplines && k<prevSplines->size(); ++k)
        mcvDrawSpline(imageClr, (*prevSplines)[k], CV_RGB(0,1,0), 1);
	    if(curSpline.degree>0)
        mcvDrawSpline(imageClr, curSpline, CV_RGB(1,0,0), 1);
	    if(bestSpline.degree>0)
        mcvDrawSpline(imageClr, bestSpline, CV_RGB(0,0,1), 1);

	    //put text
	    CvFont font;
	    cvInitFont(&font, CV_FONT_HERSHEY_TRIPLEX, .25f, .25f);
	    sprintf(str, "score=%.2f bestScre=%.2f", score, bestScore);
	    cvPutText(imageClr, str, cvPoint(30, 30), &font, CV_RGB(1,1,1));

	    sprintf(str, "Spline Fit");
	    SHOW_IMAGE(imageClr, str, 10);
	    //clear
	    cvReleaseMat(&imageClr);
	    cvReleaseMat(&im);
    }//#endif
  } //for

  //return
  if (spline)
    *spline = bestSpline;
  if (splineScore)
    *splineScore = bestScore;


  //clear
  cvReleaseMat(&points);
  cvReleaseMat(&samplePoints);
  cvReleaseMat(&randInd);
  cvReleaseMat(&pointIn);
  cvReleaseMat(&weights);
}

/** This function draws a spline onto the passed image
 *
 * \param image the input iamge
 * \param spline input spline
 * \param spline color
 *
 */
void mcvDrawSpline(CvMat *image, Spline spline, CvScalar color, int width)
{
  //get spline pixels
  CvMat *pixels = mcvGetBezierSplinePixels(spline, .05,
                                           cvSize(image->width, image->height),
                                           false);
  //if no pixels
  if (!pixels)
    return;

  //draw pixels in image with that color
  for (int i=0; i<pixels->height-1; i++)
    // 	cvSet2D(image,
    // 		(int)cvGetReal2D(pixels, i, 1),
    // 		(int)cvGetReal2D(pixels, i, 0),
    // 		color);
    cvLine(image, cvPoint((int)cvGetReal2D(pixels, i, 0),
                          (int)cvGetReal2D(pixels, i, 1)),
           cvPoint((int)cvGetReal2D(pixels, i+1, 0),
                               (int)cvGetReal2D(pixels, i+1, 1)),
           color, width);

  //put the control points with circles
  for (int i=0; i<spline.degree+1; i++)
    cvCircle(image, cvPointFrom32f(spline.points[i]), 3, color, -1);

  //release
  cvReleaseMat(&pixels);
}


/** This function draws a rectangle onto the passed image
 *
 * \param image the input iamge
 * \param rect the input rectangle
 * \param color the rectangle color
 * \param width the rectangle width
 *
 */
void mcvDrawRectangle (CvMat *image, CvRect rect, CvScalar color, int width)
{
  //draw the rectangle
  cvRectangle(image, cvPoint(rect.x, rect.y),
              cvPoint(rect.x + rect.width-1, rect.y + rect.height-1),
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
void mcvDrawText(CvMat *image, char* str, CvPoint point,
		 float size, CvScalar color)
{

  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_TRIPLEX, size, size);
  cvPutText(image, str, point, &font, color);

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
                      CameraInfo &cameraInfo, CvSize imSize)
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
    //put a dummy line at the beginning till we check that cvDiv bug
    Line dummy = {{1.,1.},{2.,2.}};
    lines.insert(lines.begin(), dummy);
    //convert to mat and get in image coordinates
    CvMat *mat = cvCreateMat(2, 2*lines.size(), FLOAT_MAT_TYPE);
    mcvLines2Mat(&lines, mat);
    lines.clear();
    mcvTransformGround2Image(mat, mat, &cameraInfo);
    //get back to vector
    mcvMat2Lines(mat, &lines);
    //remove the dummy line at the beginning
    lines.erase(lines.begin());
    //clear
    cvReleaseMat(&mat);

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
                        CameraInfo &cameraInfo, CvSize imSize)
{
  //loop on splines and convert
  for (int i=0; i<(int)splines.size(); i++)
  {
    //get points for this spline in IPM image
    CvMat *points = mcvEvalBezierSpline(splines[i], .1);

    //transform these points to image coordinates
    CvMat *points2 = cvCreateMat(2, points->height, CV_32FC1);
    cvTranspose(points, points2);
    //mcvPointImIPM2World(CvMat *mat, const IPMInfo *ipmInfo);
    //mcvTransformGround2Image(points2, points2, &cameraInfo);
    mcvTransformImIPM2Im(points2, points2, &ipmInfo, &cameraInfo);
    cvTranspose(points2, points);
    cvReleaseMat(&points2);

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
void mcvSampleWeighted(const CvMat *cumSum, int numSamples, CvMat *randInd,
                       CvRNG *rng)
{
//     //get cumulative sum of the weights
//     //OPTIMIZE:should pass it later instead of recomputing it
//     CvMat *cumSum = cvCloneMat(weights);
//     for (int i=1; i<weights->cols; i++)
// 	CV_MAT_ELEM(*cumSum, float, 0, i) += CV_MAT_ELEM(*cumSum, float, 0, i-1);

  //check if numSamples is equal or more
  int i=0;
  if (numSamples >= cumSum->cols)
  {
    for (; i<numSamples; i++)
      CV_MAT_ELEM(*randInd, int, i, 0) = i;
  }
  else
  {
    //loop
    while(i<numSamples)
    {
      //get random number
      double r = cvRandReal(rng);

      //get the index from cumSum
      int j;
      for (j=0; j<cumSum->cols && r>CV_MAT_ELEM(*cumSum, float, 0, j); j++);

      //make sure this index wasnt chosen before
      bool put = true;
      for (int k=0; k<i; k++)
        if (CV_MAT_ELEM(*randInd, int, k, 0) == j)
          //put it
          put = false;

      if (put)
      {
        //put it in array
        CV_MAT_ELEM(*randInd, int, i, 0) = j;
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
void mcvCumSum(const CvMat *inMat, CvMat *outMat)
{

#define MCV_CUM_SUM(type) 				\
    /*row vector*/ 					\
    if(inMat->rows == 1) 				\
	for (int i=1; i<outMat->cols; i++) 		\
	    CV_MAT_ELEM(*outMat, type, 0, i) += 	\
		CV_MAT_ELEM(*outMat, type, 0, i-1); 	\
    /*column vector*/					\
    else						\
	for (int i=1; i<outMat->rows; i++) 		\
	    CV_MAT_ELEM(*outMat, type, i, 0) += 	\
		CV_MAT_ELEM(*outMat, type, i-1, 0);

  //copy to output if not equal
  if(inMat != outMat)
    cvCopy(inMat, outMat);

  //check type
  if (CV_MAT_TYPE(inMat->type)==CV_32FC1)
  {
    MCV_CUM_SUM(float)
  }
  else if (CV_MAT_TYPE(inMat->type)==CV_32SC1)
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
void mcvLocalizePoints(const CvMat *im, const CvMat *inPoints,
                       CvMat *outPoints, int numLinePixels,
                       float angleThreshold)
{
  //size of inPoints must be at least 3
  if(inPoints->height<3)
  {
    cvCopy(inPoints, outPoints);
    return;
  }

  //number of pixels in line around   each point
  //int numLinePixels = 20;
  //tangent and normal
  CvPoint2D32f tangent, normal;// peakTangent;

  //threshold for accepting new point (if not changing orientation too much)
  //float angleThreshold = .7;//.96;
  CvMat *imageClr;
  char str[256];
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    //get string
    sprintf(str, "Localize Points");

    //convert image to rgb
    imageClr = cvCreateMat(im->rows, im->cols, CV_32FC3);
    cvCvtColor(im, imageClr, CV_GRAY2RGB);
  }//#endif


  //loop on the points
  for (int i=0; i<inPoints->height; i++)
  {

    //get tangent to current point
    if (i==0)
    {
      //first point, then tangent is vector to next point
      tangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, 1, 0) -
      CV_MAT_ELEM(*inPoints, float, 0, 0),
                             CV_MAT_ELEM(*inPoints, float, 1, 1) -
                             CV_MAT_ELEM(*inPoints, float, 0, 1));
    }
    else if (i==1)
      tangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, 1, 0) -
                             CV_MAT_ELEM(*outPoints, float, 0, 0),
                             CV_MAT_ELEM(*inPoints, float, 1, 1) -
                             CV_MAT_ELEM(*outPoints, float, 0, 1));

    else //if (i==inPoints->height-1)
    {
      //last pointm then vector from previous two point
      tangent = cvPoint2D32f(CV_MAT_ELEM(*outPoints, float, i-1, 0) -
                             CV_MAT_ELEM(*outPoints, float, i-2, 0),
                             CV_MAT_ELEM(*outPoints, float, i-1, 1) -
                             CV_MAT_ELEM(*outPoints, float, i-2, 1));
      // 	    tangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, i, 0) -
      // 				   CV_MAT_ELEM(*outPoints, float, i-1, 0),
      // 				   CV_MAT_ELEM(*inPoints, float, i, 1) -
      // 				   CV_MAT_ELEM(*outPoints, float, i-1, 1));
    }
// 	else
// 	{
// 	    //general point, then take next - previous
// 	    tangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, i, 0) - //i+1
// 				   CV_MAT_ELEM(*outPoints, float, i-1, 0),
// 				   CV_MAT_ELEM(*inPoints, float, i, 1) - //i+1
// 				   CV_MAT_ELEM(*outPoints, float, i-1, 1));
// 	}

    //get normal
    float ss = cvInvSqrt(tangent.x * tangent.x + tangent.y * tangent.y);
    tangent.x *= ss; tangent.y *= ss;
    normal.x = tangent.y; normal.y = -tangent.x;

    //get points in normal direction
    Line line;
    line.startPoint = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, i, 0) +
                numLinePixels * normal.x,
                CV_MAT_ELEM(*inPoints, float, i, 1) +
                numLinePixels * normal.y);
    line.endPoint = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, i, 0) -
              numLinePixels * normal.x,
              CV_MAT_ELEM(*inPoints, float, i, 1) -
              numLinePixels * normal.y);


    CvPoint2D32f prevPoint = {0., 0.};
    if (i>0)
      prevPoint = cvPoint2D32f(CV_MAT_ELEM(*outPoints, float, i-1, 0),
                               CV_MAT_ELEM(*outPoints, float, i-1, 1));

    //get line peak i.e. point in middle of bright line on dark background
    CvPoint2D32f peak;
  // 	float val = mcvGetLinePeak(im, line, peak);
    //get line peak
    vector<CvPoint2D32f> peaks;
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
    if (mcvIsPointInside(line.startPoint, cvSize(im->width, im->height)) &&
        mcvIsPointInside(line.endPoint, cvSize(im->width, im->height)) &&
        (//!i ||
        (i>0 &&
          mcvIsValidPeak(peak, tangent, prevPoint,
            angleThreshold))) )
    {
      //put new peak
      CV_MAT_ELEM(*outPoints, float, i, 0) = peak.x;
      CV_MAT_ELEM(*outPoints, float, i, 1) = peak.y;
    }
    else
    {
      //keep original point
      CV_MAT_ELEM(*outPoints, float, i, 0) = CV_MAT_ELEM(*inPoints, float, i, 0);
      CV_MAT_ELEM(*outPoints, float, i, 1) = CV_MAT_ELEM(*inPoints, float, i, 1);
    }

    //debugging
    if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES

      fprintf(stderr, "Localize val=%.3f\n", val);

      //draw original point, localized point, and line endpoints
      cvLine(imageClr, cvPointFrom32f(line.startPoint),
            cvPointFrom32f(line.endPoint), CV_RGB(0, 0, 1));
      //output points
      cvCircle(imageClr, cvPoint((int)CV_MAT_ELEM(*outPoints, float, i, 0),
                                (int)CV_MAT_ELEM(*outPoints, float, i, 1)),
              1, CV_RGB(0, 1, 0), -1);
      //input points
      cvCircle(imageClr, cvPoint((int)(line.startPoint.x+line.endPoint.x)/2,
                                (int)(line.startPoint.y+line.endPoint.y)/2),
              1, CV_RGB(1, 0, 0), -1);
      //show image
      SHOW_IMAGE(imageClr, str, 10);
    }//#endif
  } // for i

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    SHOW_IMAGE(imageClr, str, 10);
    //clear
    cvReleaseMat(&imageClr);
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
bool mcvIsValidPeak(const CvPoint2D32f &peak, const CvPoint2D32f &tangent,
                    const CvPoint2D32f &prevPoint, float angleThreshold)
{
  //compute the tangent line for the peak
  CvPoint2D32f peakTangent;
  peakTangent.x = peak.x - prevPoint.x;
  peakTangent.y = peak.y - prevPoint.y;

  //normalize new tangent
  float ss = cvInvSqrt(peakTangent.x * peakTangent.x + peakTangent.y *
                       peakTangent.y);
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
int mcvChooseBestPeak(const vector<CvPoint2D32f> &peaks,
                      const vector<float> &peakVals,
                      CvPoint2D32f &peak, float &peakVal,
                      float contThreshold, const CvPoint2D32f &tangent,
                      const CvPoint2D32f &prevPoint, float angleThreshold)
{
  int index=-1;
  float maxAngle=0;
  peakVal = -1;

  //loop and check
  for (unsigned int i=0; i<peaks.size(); ++i)
  {
    CvPoint2D32f peak = peaks[i];

    //compute the tangent line for the peak and normalize
    CvPoint2D32f peakTangent;
    peakTangent.x = peak.x - prevPoint.x;
    peakTangent.y = peak.y - prevPoint.y;
    peakTangent = mcvNormalizeVector(peakTangent);

    //compute angle
    float angle = fabs(peakTangent.x*tangent.x + peakTangent.y*tangent.y);

    //check if min angle so far and above both thresholds
    if (DEBUG_LINES)
      fprintf(stderr, "peak#%d/%d (%f, %f): angle=%f, maxAngle=%f\n",
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
CvMat*  mcvExtendPoints(const CvMat *im, const CvMat *inPoints,
                        float angleThreshold, float meanDirAngleThreshold,
                        int linePixelsTangent, int linePixelsNormal,
                        float contThreshold, int deviationThreshold,
                        CvRect bbox, bool smoothPeaks)
{
  //size of inPoints must be at least 3
  if(inPoints->height<4)
  {
    return cvCloneMat(inPoints);
  }


  char str[256];
  CvMat *imageClr;
  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    //get string
    sprintf(str, "Extend Points");

    //convert image to rgb
    imageClr = cvCreateMat(im->rows, im->cols, CV_32FC3);
    CvMat *im2 = cvCloneMat(im);
    mcvScaleMat(im, im2);
    cvCvtColor(im2, imageClr, CV_GRAY2RGB);
    cvReleaseMat(&im2);

    //show original points
    for(int i=0; i<inPoints->height; i++)
        //input points
        cvCircle(imageClr, cvPoint((int)(CV_MAT_ELEM(*inPoints, float, i, 0)),
                                   (int)(CV_MAT_ELEM(*inPoints, float, i, 1))),
                 1, CV_RGB(0, 1, 1), -1);
    //show image
    SHOW_IMAGE(imageClr, str, 10);
  }//#endif

  //tangent and normal
  CvPoint2D32f tangent, curPoint, peak, nextPoint, meanDir;
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
  vector<CvPoint2D32f> backPoints;
  int numBack = 0;
  int deviationCount = 0;
  vector<CvPoint2D32f> peaks;
  vector<float> peakVals;
  //get mean direction of points
  meanDir = mcvGetPointsMeanVector(inPoints, false);
  while(cont)
  {
    int outSize = (int)backPoints.size();
    //get tangent from previous point in input points if no output points yet
    if(outSize==0)
    {
	    curPoint = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, 0, 0),
                              CV_MAT_ELEM(*inPoints, float, 0, 1));
	    tangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, 0, 0) -
                             CV_MAT_ELEM(*inPoints, float, 1, 0),
                             CV_MAT_ELEM(*inPoints, float, 0, 1) -
                             CV_MAT_ELEM(*inPoints, float, 1, 1));
      // 	    prevTangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, 1, 0) -
      // 				       CV_MAT_ELEM(*inPoints, float, 2, 0),
      // 				       CV_MAT_ELEM(*inPoints, float, 1, 1) -
      // 				       CV_MAT_ELEM(*inPoints, float, 2, 1));
      // 	    prevTangent = mcvNormalizeVector(prevTangent);

      // 	    pprevTangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, 2, 0) -
      // 				       CV_MAT_ELEM(*inPoints, float, 3, 0),
      // 				       CV_MAT_ELEM(*inPoints, float, 2, 1) -
      // 				       CV_MAT_ELEM(*inPoints, float, 3, 1));
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
        tangent = cvPoint2D32f(backPoints[outSize-1].x -
                               CV_MAT_ELEM(*inPoints, float, 0, 0),
                               backPoints[outSize-1].y -
                               CV_MAT_ELEM(*inPoints, float, 0, 1));
	    }
	    //more than one
	    else
	    {
        tangent = cvPoint2D32f(backPoints[outSize-1].x -
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
    //if (mcvIsPointInside(nextPoint, cvSize(im->width-1, im->height-1)))
    if (mcvIsPointInside(nextPoint, bbox))
    {
	    //clip line
	    mcvIntersectLineWithBB(&line, cvSize(im->width-1, im->height-1), &line);

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
    cvLine(imageClr, cvPointFrom32f(line.startPoint),
           cvPointFrom32f(line.endPoint),
    CV_RGB(0, 0, 1));
    //output points
    cvCircle(imageClr, cvPointFrom32f(peak), 1, CV_RGB(0, 1, 0), -1);
    //input points
    cvCircle(imageClr, cvPointFrom32f(nextPoint), 1, CV_RGB(1, 0, 0), -1);
    //show image
    SHOW_IMAGE(imageClr, str, 10);
    }//#endif
  }  // while cont

  //do the same for the opposite direction
  cont = true;
  vector<CvPoint2D32f> frontPoints;
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
	    curPoint = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float,
                                          inPoints->height-1, 0),
                              CV_MAT_ELEM(*inPoints, float,
                                          inPoints->height-1, 1));
	    tangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float,
                                         inPoints->height-1, 0) -
                             CV_MAT_ELEM(*inPoints, float,
                                         inPoints->height-2, 0),
                             CV_MAT_ELEM(*inPoints, float,
                                         inPoints->height-1, 1) -
                             CV_MAT_ELEM(*inPoints, float,
                                         inPoints->height-2, 1));

      // 	    prevTangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, inPoints->height-2, 0) -
      // 				       CV_MAT_ELEM(*inPoints, float, inPoints->height-3, 0),
      // 				       CV_MAT_ELEM(*inPoints, float, inPoints->height-2, 1) -
      // 				       CV_MAT_ELEM(*inPoints, float, inPoints->height-3, 1));
      // 	    prevTangent = mcvNormalizeVector(prevTangent);

      // 	    pprevTangent = cvPoint2D32f(CV_MAT_ELEM(*inPoints, float, inPoints->height-3, 0) -
      // 				       CV_MAT_ELEM(*inPoints, float, inPoints->height-4, 0),
      // 				       CV_MAT_ELEM(*inPoints, float, inPoints->height-3, 1) -
      // 				       CV_MAT_ELEM(*inPoints, float, inPoints->height-4, 1));
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
        tangent = cvPoint2D32f(frontPoints[outSize-1].x -
                               CV_MAT_ELEM(*inPoints, float,
                                           inPoints->height-1, 0),
                               frontPoints[outSize-1].y -
                               CV_MAT_ELEM(*inPoints, float,
                                           inPoints->height-1, 1));
	    }
	    //more than one
	    else
	    {
        tangent = cvPoint2D32f(frontPoints[outSize-1].x -
                               frontPoints[outSize-2].x,
                               frontPoints[outSize-1].y -
                               frontPoints[outSize-2].y);
	    }
    }

    Line line;
    line = mcvGetExtendedNormalLine(curPoint, tangent, linePixelsTangent,
                                    linePixelsNormal, nextPoint);

    //check if still inside
  // 	if (mcvIsPointInside(nextPoint, cvSize(im->width-1, im->height-1)))
    if (mcvIsPointInside(nextPoint, bbox))
    {
      //clip line
      mcvIntersectLineWithBB(&line, cvSize(im->width-1, im->height-1), &line);

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
	    cvLine(imageClr, cvPointFrom32f(line.startPoint),
             cvPointFrom32f(line.endPoint), CV_RGB(0, 0, 1));
	    //output points
	    cvCircle(imageClr, cvPointFrom32f(peak), 1, CV_RGB(0, 1, 0), -1);
	    //input points
	    cvCircle(imageClr, cvPointFrom32f(nextPoint), 1, CV_RGB(1, 0, 0), -1);
	    //show image
	    SHOW_IMAGE(imageClr, str, 10);
    }//#endif
  }

  numFront = frontPoints.size();
  numBack = backPoints.size();
  //now that we have extended the points in both directions, we need to put them
  //back into the return matrix
  CvMat *extendedPoints = cvCreateMat(inPoints->height + numBack + numFront,
                                      2, CV_32FC1);
  //first put back points in reverse order
  vector<CvPoint2D32f>::iterator pointi;
  int i = 0;
  for (i=0, pointi=backPoints.end()-1; i<numBack; i++, pointi--)
  {
    CV_MAT_ELEM(*extendedPoints, float, i, 0) = (*pointi).x;
    CV_MAT_ELEM(*extendedPoints, float, i, 1) = (*pointi).y;
  }

  //then put the original points
  i = numBack;
  memcpy(cvPtr2D(extendedPoints, i, 0), inPoints->data.fl,
         sizeof(float)*2*inPoints->height);

  //then put the front points in normal order
  for (i = numBack+inPoints->height, pointi=frontPoints.begin();
       i<extendedPoints->height; pointi++, i++)
  {
    CV_MAT_ELEM(*extendedPoints, float, i, 0) = (*pointi).x;
    CV_MAT_ELEM(*extendedPoints, float, i, 1) = (*pointi).y;
  }

  if(DEBUG_LINES) {//#ifdef DEBUG_GET_STOP_LINES
    SHOW_IMAGE(imageClr, str, 10);
    //clear
    cvReleaseMat(&imageClr);
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
Line mcvGetExtendedNormalLine(CvPoint2D32f &curPoint, CvPoint2D32f &tangent,
                              int linePixelsTangent, int linePixelsNormal,
                              CvPoint2D32f &nextPoint)
{
  //normalize tangent
  float ssq = cvInvSqrt(tangent.x*tangent.x + tangent.y*tangent.y);
  tangent.x *= ssq;
  tangent.y *= ssq;

  //get next point along the way
  nextPoint.x = curPoint.x + linePixelsTangent * tangent.x;
  nextPoint.y = curPoint.y + linePixelsTangent * tangent.y;

  //get normal direction
  CvPoint2D32f normal = cvPoint2D32f(-tangent.y, tangent.x);

  //get two points along the normal line
  Line line;
  line.startPoint = cvPoint2D32f(nextPoint.x + linePixelsNormal*normal.x,
                                 nextPoint.y + linePixelsNormal*normal.y);
  line.endPoint = cvPoint2D32f(nextPoint.x - linePixelsNormal*normal.x,
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
float mcvGetLinePeak(const CvMat *im, const Line &line,
                     vector<CvPoint2D32f> &peaks,
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
  CvMat step = cvMat(1, stepsize, CV_32FC1, stepp);

  //take negative to work for opposite polarity
  if (!positivePeak)
    cvScale(&step, &step, -1);
  //     //get the gaussian kernel to convolve with
  //     int width = 5;
  //     float step = .5;
  //     CvMat *step = cvCreateMat(1, (int)(2*width/step+1), CV_32FC1);
  //     int j; float i;
  //     for (i=-w, j=0; i<=w; i+=step, ++j)
  //         CV_MAT_ELEM(*step, FLOAT_MAT_ELEM_TYPE, 0, j) =
  //            (float) exp(-(.5*i*i));


  //then get the pixel coordinates of the line in the image
  CvMat *pixels;
  pixels = mcvGetLinePixels(line);
  //get pixel values
  CvMat *pix = cvCreateMat(1, pixels->height, CV_32FC1);
  for(int j=0; j<pixels->height; j++)
  {
    CV_MAT_ELEM(*pix, float, 0, j) =
        cvGetReal2D(im,
                    MIN(MAX(CV_MAT_ELEM(*pixels, int, j, 1),0),im->height-1),
                    MIN(MAX(CV_MAT_ELEM(*pixels, int, j, 0),0),im->width-1));
  }
  //clear
  cvReleaseMat(&pixels);

  //remove the mean
  CvScalar mean = cvAvg(pix);
  cvSubS(pix, mean, pix);

  //convolve with step
  CvMat *pixStep = cvCreateMat(pix->height, pix->width, CV_32FC1);
  if (smoothPeaks)
    cvFilter2D(pix, pixStep, &step);
  else
    cvCopy(pix, pixStep);
  //     SHOW_MAT(pixStep, "pixStep");
  //     SHOW_MAT(pix, "pixels");

  //get local maxima
  double topVal;
  float top;
  vector<double> maxima;
  vector<int> maximaLoc;
  CvPoint2D32f peak;
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
	    double val1 = CV_MAT_ELEM(*pixStep, float, 0, MAX(maximaLoc[i]-1, 0));
	    double val3 = CV_MAT_ELEM(*pixStep, float, 0, MIN(maximaLoc[i]+1,
                                                        pixStep->width-1));
	    top = (float)mcvGetLocalMaxSubPixel(val1, maxima[i], val3);
	    top += maximaLoc[i];
	    //fprintf(stderr, "val1=%f, val2=%f, val3=%f\n", val1, maxima[i], val3);
	    //fprintf(stderr, "top=%d, subpixel=%f\n", maximaLoc[i], top);
	    top /= pix->width;
	    //get loc
// 	    top = maximaLoc[i]/(float)(pix->width);
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
    top = (pix->width-2)/2./(pix->width);
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
//     top /= (pix->width);
//     peak.x = line.startPoint.x*(1-top) + top * line.endPoint.x;
//     peak.y = line.startPoint.y*(1-top) + top * line.endPoint.y;

  //clear
  cvReleaseMat(&pix);
  cvReleaseMat(&pixStep);

  //return mean of rising and falling val
  return  topVal;//MIN(risingVal, fallingVal);//no minus //(risingVal+fallingVal)/2;
}

/** This functions normalizes the given vector
 *
 * \param vector the input vector to normalize
 */
CvPoint2D32f mcvNormalizeVector(const CvPoint2D32f &v)
{
  //return vector
  CvPoint2D32f ret = v;

  //normalize vector
  float ssq = cvInvSqrt(ret.x*ret.x + ret.y*ret.y);
  ret.x *= ssq;
  ret.y *= ssq;

  //return
  return ret;
}


/** This functions normalizes the given vector
 *
 * \param vector the input vector to normalize
 */
CvPoint2D32f mcvNormalizeVector(const CvPoint &v)
{
  //return vector
  return mcvNormalizeVector(cvPointTo32f(v));

}

/** This functions normalizes the given vector
 *
 * \param x the x component
 * \param y the y component
 */
CvPoint2D32f mcvNormalizeVector(float x, float y)
{
  //return vector
  return mcvNormalizeVector(cvPoint2D32f(x, y));
}


/** This functions adds two vectors and returns the result
 *
 * \param v1 the first vector
 * \param v2 the second vector
 * \return the sum
 */
CvPoint2D32f mcvAddVector(CvPoint2D32f v1, CvPoint2D32f v2)
{
  //get sum
  CvPoint2D32f sum = cvPoint2D32f(v1.x + v2.x, v1.y + v2.y);
  //return vector
  return sum;
}


/** This functions multiplies a vector by a scalar
 *
 * \param v the vector
 * \param s the scalar
 * \return the sum
 */
CvPoint2D32f mcvMultiplyVector(CvPoint2D32f v, float s)
{
  //get sum
  CvPoint2D32f prod;
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
float mcvGetSplineScore(const CvMat* image, Spline& spline, float h,
                        int  jitterVal, float lengthRatio, float angleRatio)
{

  //check that all control points for spline are inside the image
  CvSize size = cvSize(image->width-1, image->height-1);
  //     SHOW_SPLINE(spline, "spline");
  for (int i=0; i<=spline.degree; i++)
    if (!mcvIsPointInside(spline.points[i], size))
	    return -100.f;

  //get the pixels that belong to the spline
  CvMat *pixels = mcvGetBezierSplinePixels(spline, h, size, false);
  if(!pixels)
    return -100.f;

  //get jitter vector
  vector<int>jitter = mcvGetJitterVector(jitterVal); //2);

  //compute its score by summing up pixel values belonging to it
  //int jitter[] = {0, 1, -1, 2, -2}, jitterLength = 5;
  //SHOW_MAT(pixels, "pixels");
  float score = 0.f;
  for (unsigned int j=0; j<jitter.size(); j++)
    for (int i=0; i<pixels->height; i++)
    {
	    //jitter in x
      // 	    int k = MIN(MAX(CV_MAT_ELEM(*pixels, int, i, 0)+
      // 			    jitter[j], 0), image->width-1);
      // 	    fprintf(stderr, "col=%d\n & row=%d", k, CV_MAT_ELEM(*pixels, int, i, 1));
	    score += cvGetReal2D(image, CV_MAT_ELEM(*pixels, int, i, 1),
                           MIN(MAX(CV_MAT_ELEM(*pixels, int, i, 0) +
                           jitter[j], 0), image->width-1));
      // 	    //jitter the y
      // 	    score += cvGetReal2D(image,
      // 				 MIN(MAX(CV_MAT_ELEM(*pixels, int, i, 1)+
      // 					 jitter[j], 0), image->height-1),
      // 				 CV_MAT_ELEM(*pixels, int, i, 0));
    } // for i

  //length: min 0 and max of 1 (normalized according to max of width and height
  //of image)
  //float length = ((float)pixels->height) / MAX(image->width, image->height);
  float length = 0.f;
  //     for (int i=0; i<pixels->height-1; i++)
  //     {
  // 	//get the vector between every two consecutive points
  // 	CvPoint2D32f v =
  // 	    mcvSubtractVector(cvPoint2D32f(CV_MAT_ELEM(*pixels, int, i+1, 0),
  // 					   CV_MAT_ELEM(*pixels, int, i+1, 1)),
  // 			      cvPoint2D32f(CV_MAT_ELEM(*pixels, int, i, 0),
  // 					   CV_MAT_ELEM(*pixels, int, i, 1)));
  // 	//add to length
  // 	length += cvSqrt(v.x * v.x + v.y * v.y);
  //     }
  //get length between first and last control point
  CvPoint2D32f v = mcvSubtractVector(spline.points[0], spline.points[spline.degree]);
  length = cvSqrt(v.x * v.x + v.y * v.y);
  //normalize
  length /= image->height; //MAX(image->width, image->height);

  //add measure of spline straightness: angle between vectors from points 1&2 and
  //points 2&3: clsoer to 1 the better (straight)
  //add 1 to value to make it range from 0->2 (2 better)
  float angle = 0;
  for (int i=0; i<spline.degree-1; i++)
  {
    //get first vector
    CvPoint2D32f t1 = mcvNormalizeVector (mcvSubtractVector(spline.points[i+1],
                                                            spline.points[i]));

    //get second vector
    CvPoint2D32f t2 = mcvNormalizeVector (mcvSubtractVector(spline.points[i+2],
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
  //score = .8*score + .4*pixels->height; //.8 & .3

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
  cvReleaseMat(&pixels);
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
CvPoint2D32f  mcvGetPointsMeanVector(const CvMat *points, bool forward)
{
  CvPoint2D32f mean, v;

  //init
  mean = cvPoint2D32f(0,0);

  //go forward direction
  for (int i=1; i<points->height; ++i)
  {
    //get the vector joining the two points
    v = cvPoint2D32f(CV_MAT_ELEM(*points, float, i, 0) -
                     CV_MAT_ELEM(*points, float, i-1, 0),
                     CV_MAT_ELEM(*points, float, i, 1) -
                     CV_MAT_ELEM(*points, float, i-1, 1));
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
    mean = cvPoint2D32f(-mean.x, -mean.y);

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
  CvPoint2D32f centroid1, centroid2;
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

    CvMat* im = cvCreateMat(480, 640, CV_8UC3);
    cvSet(im, cvRealScalar(0.));
    //draw splines
    mcvDrawSpline(im, sp1, CV_RGB(255, 0, 0), 1);
    mcvDrawSpline(im, sp2, CV_RGB(0, 255, 0), 1);
    SHOW_IMAGE(im, "Check Merge Splines", 10);
    //clear
    cvReleaseMat(&im);

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
void mcvGetPointsFeatures(const CvMat* points, CvPoint2D32f* centroid,
                          float* theta, float* r, float* length,
                          float* meanTheta, float* meanR, float* curveness)
{

  //get start and end point
  CvPoint2D32f start = cvPoint2D32f(CV_MAT_ELEM(*points, float, 0, 0),
                                    CV_MAT_ELEM(*points, float, 0, 1));
  CvPoint2D32f end = cvPoint2D32f(CV_MAT_ELEM(*points, float,
                                              points->height-1, 0),
                                  CV_MAT_ELEM(*points, float,
                                              points->height-1, 1));
  //compute centroid
  if (centroid)
  {
    //get sum of control points
    *centroid = cvPoint2D32f(0, 0);
    for (int i=0; i<=points->height; ++i)
	    *centroid = mcvAddVector(*centroid,
                               cvPoint2D32f(CV_MAT_ELEM(*points, float, i, 0),
                                            CV_MAT_ELEM(*points, float, i, 1)));
    //take mean
    *centroid = cvPoint2D32f(centroid->x / (points->height),
                             centroid->y / (points->height));
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
    for (int i=0; i<points->height-1; i++)
    {
      //get the line
      Line line;
      line.startPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, i, 0),
                                      CV_MAT_ELEM(*points, float, i, 1));
      line.endPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, i+1, 0),
                                    CV_MAT_ELEM(*points, float, i+1, 1));
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
    *meanTheta /= points->height - 1;
    *meanR /= points->height - 1;
  }

  //compute length of spline: length of vector between first and last point
  if (length)
  {
    //get the vector
    CvPoint2D32f v = mcvSubtractVector(start, end);

    //compute length
    *length = cvSqrt(v.x * v.x + v.y * v.y);
  }

  //compute curveness
  if (curveness)
  {
    *curveness = 0;
    if (points->height>2)
    {
      //initialize
      CvPoint2D32f p0;
      CvPoint2D32f p1 = start;
      CvPoint2D32f p2 = cvPoint2D32f(CV_MAT_ELEM(*points, float, 1, 0),
                                      CV_MAT_ELEM(*points, float, 1, 1));

      for (int i=0; i<points->height-2; i++)
      {
        //go next
        p0 = p1;
        p1 = p2;
        p2 = cvPoint2D32f(CV_MAT_ELEM(*points, float, i+2, 0),
                          CV_MAT_ELEM(*points, float, i+2, 1));
        //get first vector
        CvPoint2D32f t1 = mcvNormalizeVector(mcvSubtractVector(p1, p0));

        //get second vector
        CvPoint2D32f t2 = mcvNormalizeVector (mcvSubtractVector(p2, p1));
        //get angle
        *curveness += t1.x*t2.x + t1.y*t2.y;
      }
    //get mean
    *curveness /= points->height-2;
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
void mcvGetSplineFeatures(const Spline& spline, CvPoint2D32f* centroid,
                          float* theta, float* r, float* length,
                          float* meanTheta, float* meanR, float* curveness)
{
  //compute centroid
  if (centroid)
  {
    //get sum of control points
    *centroid = cvPoint2D32f(0, 0);
    for (int i=0; i<=spline.degree; ++i)
      *centroid = mcvAddVector(*centroid, spline.points[i]);
    //take mean
    *centroid = cvPoint2D32f(centroid->x / (spline.degree+1),
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
    CvMat* points = mcvEvalBezierSpline(spline, .1);
    //loop and get theta
    for (int i=0; i<points->height-1; i++)
    {
	    //get the line
	    Line line;
	    line.startPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, i, 0),
                                     CV_MAT_ELEM(*points, float, i, 1));
	    line.endPoint = cvPoint2D32f(CV_MAT_ELEM(*points, float, i+1, 0),
                                   CV_MAT_ELEM(*points, float, i+1, 1));
	    //get theta and r
	    float r, t;
	    mcvLineXY2RTheta(line, r, t);
	    //add pi if neg
      #warning "add pi to theta calculations for spline feature"
	    //if (t<0) t += CV_PI;
	    //add
	    t = mcvGetLineAngle(line);
	    *meanTheta += t;
	    *meanR += r;
    }

    //normalize
    *meanTheta /= points->height - 1;
    *meanR /= points->height - 1;

    //clear
    cvReleaseMat(&points);
  }

  //compute length of spline: length of vector between first and last point
  if (length)
  {
    //get the vector
    CvPoint2D32f v = cvPoint2D32f(spline.points[0].x -
                                  spline.points[spline.degree].x,
                                  spline.points[0].y -
                                  spline.points[spline.degree].y);
    //compute length
    *length = cvSqrt(v.x * v.x + v.y * v.y);
  }

  //compute curveness
  if (curveness)
  {
    *curveness = 0;
    for (int i=0; i<spline.degree-1; i++)
    {
	    //get first vector
	    CvPoint2D32f t1 =
        mcvNormalizeVector(mcvSubtractVector(spline.points[i+1],
                                             spline.points[i]));

	    //get second vector
	    CvPoint2D32f t2 =
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
CvPoint2D32f  mcvSubtractVector(const CvPoint2D32f& v1, const CvPoint2D32f& v2)
{
  return cvPoint2D32f(v1.x - v2.x, v1.y - v2.y);
}

/** This functions computes the vector norm
 *
 * \param v input vector
 * \return norm of the vector
 *
 */
float  mcvGetVectorNorm(const CvPoint2D32f& v)
{

  return cvSqrt(v.x * v.x + v.y * v.y);
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


    CvMat* im = cvCreateMat(480, 640, CV_8UC3);
    cvSet(im, cvRealScalar(0.));
    //draw lines
    mcvDrawLine(im, line1, CV_RGB(255, 0, 0), 1);
    mcvDrawLine(im, line2, CV_RGB(0, 255, 0), 1);
    SHOW_IMAGE(im, "Check Merge Lines", 10);
    //clear
    cvReleaseMat(&im);

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
  CvPoint2D32f dir = mcvSubtractVector(line.endPoint, line.startPoint);
  //get intermediate points
  for (int j=1; j<degree; ++j)
  {
    //get point
    CvPoint2D32f point;
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
  CvPoint2D32f v = mcvNormalizeVector(mcvSubtractVector(line.startPoint,
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
LineColor mcvGetPointsColor(const CvMat* im, const CvMat* points,
                            int window, float numYellowMin,
                            float rgMin, float rgMax,
                            float gbMin, float rbMin,
                            bool rbf, float rbfThreshold)
{

  //check if color image
  if (cvGetElemType(im) != CV_8UC3)
    return LINE_COLOR_WHITE;

  //    //half the width of the window
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
  CvMat* hist = cvCreateMat(1, histLen, CV_32FC1);

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
    cvSet(hist, cvRealScalar(0));

    //get the window indices
    int xmin = MAX(cvRound(CV_MAT_ELEM(*points, float, i, 0)-window), 0);
    int xmax = MIN(cvRound(CV_MAT_ELEM(*points, float, i, 0)+window),
                   im->cols);
    int ymin = MAX(cvRound(CV_MAT_ELEM(*points, float, i, 1)-window), 0);
    int ymax = MIN(cvRound(CV_MAT_ELEM(*points, float, i, 1)+window),
                   im->rows);

    //get mean for every channel
    float r=0.f, g=0.f, b=0.f, rr, gg, bb;
    int bin;
    for (int x=xmin; x<=xmax; x++)
	    for (int y=ymin; y<=ymax; y++)
	    {
        //get colors
        rr = (im->data.ptr + im->step*y)[x*3];
        gg = (im->data.ptr + im->step*y)[x*3+1];
        bb = (im->data.ptr + im->step*y)[x*3+2];
        //add to totals
        r += rr;
        g += gg;
        b += bb;

        if (rbf)
        {
          //compute histogram
          bin = MIN((int)(rr / binWidth), numBins);
          hist->data.fl[bin] ++;
          bin = MIN((int)(gg / binWidth), numBins);
          hist->data.fl[bin + numBins] ++;
          bin = MIN((int)(bb / binWidth), numBins);
          hist->data.fl[bin + 2*numBins] ++;
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
	    hist->data.fl[hist->width-2] = fabs(rg);
	    hist->data.fl[hist->width-1] = fabs(gb);
	    hist->data.fl[hist->width] = fabs(rb);

	    //compute output of RBF model
	    //
	    //add rest of terms
	    for (int j=0; j<rbfNumCentroids; j++)
	    {
        //compute squared distance to centroid
        float d = 0., t;
        for (int k=0; k<histLen; k++)
        {
            t = hist->data.fl[k] -
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
  cvReleaseMat(&hist);

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
                                CvSize size, vector<CvRect> &boxes)
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
        //get min and max x and add the bounding box covering the whole height
        start = (int)fmin(splines[i].points[0].x,
                          splines[i].points[splines[i].degree].x);
        end = (int)fmax(splines[i].points[0].x,
                        splines[i].points[splines[i].degree].x);
        boxes.push_back(cvRect(start, 0, end-start+1, size.height-1));
      }
      break;

    case LINE_HORIZONTAL:
      for(unsigned int i=0; i<splines.size(); ++i)
      {
        //get min and max y and add the bounding box covering the whole width
        start = (int)fmin(splines[i].points[0].y,
                          splines[i].points[splines[i].degree].y);
        end = (int)fmax(splines[i].points[0].y,
                        splines[i].points[splines[i].degree].y);
        boxes.push_back(cvRect(0, start, size.width-1, end-start+1));
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
                              CvSize size, vector<CvRect> &boxes)
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
        //get min and max x and add the bounding box covering the whole height
        start = (int)fmin(lines[i].startPoint.x, lines[i].endPoint.x);
        end = (int)fmax(lines[i].startPoint.x, lines[i].endPoint.x);
        boxes.push_back(cvRect(start, 0, end-start+1, size.height-1));
      }
      break;

    case LINE_HORIZONTAL:
      for(unsigned int i=0; i<lines.size(); ++i)
      {
        //get min and max y and add the bounding box covering the whole width
  	    start = (int)fmin(lines[i].startPoint.y, lines[i].endPoint.y);
        end = (int)fmax(lines[i].startPoint.y, lines[i].endPoint.y);
        boxes.push_back(cvRect(0, start, size.width-1, end-start+1));
      }
      break;
    }
}


/** \brief This function takes a bunch of lines, and check which
 * 2 lines can make a lane
 *
 * \param lines vector of lines
 * \param scores vector of line scores
 * \param wMu expected lane width
 * \param wSigma std deviation of lane width
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

