/**
 * \file LaneDetector.hh
 * \author Mohamed Aly <malaa@caltech.edu>
 * \date Thu 26 Jul, 2007
 *
 */

#ifndef LANEDETECTOR_HH_
#define LANEDETECTOR_HH_

#include "mcv.hh"
#include "InversePerspectiveMapping.hh"

namespace LaneDetector
{

//Debug global variable
extern int DEBUG_LINES;

///Line type
typedef enum LineType_ {
  LINE_HORIZONTAL = 0,
  LINE_VERTICAL = 1
} LineType;

/// Line color
typedef enum LineColor_ {
  LINE_COLOR_NONE,
  LINE_COLOR_YELLOW,
  LINE_COLOR_WHITE
} LineColor;

/// Line structure with start and end points
typedef struct Line
{
  ///start point
  FLOAT_POINT2D startPoint;
  ///end point
  FLOAT_POINT2D endPoint;
  ///color of line
  LineColor color;
  ///score of line
  float score;
} Line;

/// Spline structure
typedef struct Spline
{
  ///degree of spline
  int degree;
  ///points in spline
  CvPoint2D32f points[4];
  ///color of spline
  LineColor color;
  ///score of spline
  float score;
} Spline;

///Structure to hold state used for initializing the next detection process
///from a previous one
typedef struct LineState_
{
  ///Splines detected in IPM image
  vector<Spline> ipmSplines;

  ///bounding boxes to work on the splines from the previous frames
  vector<CvRect> ipmBoxes;
} LineState;

typedef enum CheckSplineStatus_
{
  ShortSpline = 0x1,
  //spline is curved i.e. form control points
  CurvedSpline = 0x2,
  //spline is curved i.e. overall (thetaDiff)
  CurvedSplineTheta = 0x4,
  HorizontalSpline = 0x8
} CheckSplineStatus;

#define GROUPING_TYPE_HV_LINES 0
#define GROUPING_TYPE_HOUGH_LINES 1

///Structure to hold lane detector settings
typedef struct LaneDetectorConf
{
  ///width of IPM image to use
  FLOAT ipmWidth;
  ///height of IPM image
  FLOAT ipmHeight;
  ///Left point in original image of region to make IPM for
  int ipmLeft;
  ///Right point in original image of region to make IPM for
  int ipmRight;
  ///Top point in original image of region to make IPM for
  int ipmTop;
  ///Bottom point in original image of region to make IPM for
  int ipmBottom;
  ///The method to use for IPM interpolation
  int ipmInterpolation;

  ///width of line we are detecting
  FLOAT lineWidth;
  ///height of line we are detecting
  FLOAT lineHeight;
  ///kernel size to use for filtering
  unsigned char kernelWidth;
  unsigned char kernelHeight;
  ///lower quantile to use for thresholding the filtered image
  FLOAT lowerQuantile;
  ///whether to return local maxima or just the maximum
  bool localMaxima;
  ///the type of grouping to use: 0 for HV lines and 1 for Hough Transform
  unsigned char groupingType;
  ///whether to binarize the thresholded image or use the
  ///raw filtered image
  bool binarize;
  //unsigned char topClip;
  ///threshold for line scores to declare as line
  FLOAT detectionThreshold;
  ///whtehter to smooth the line scores detected or not
  bool smoothScores;
  ///rMin, rMax and rStep for Hough Transform (pixels)
  float rMin, rMax, rStep;
  ///thetaMin, thetaMax, thetaStep for Hough Transform (radians)
  float thetaMin, thetaMax, thetaStep;
  ///portion of image height to add to y-coordinate of vanishing
  ///point when computing the IPM image
  float ipmVpPortion;
  ///get end points or not
  bool getEndPoints;
  ///group nearby lines
  bool group;
  ///threshold for grouping nearby lines
  float groupThreshold;
  ///use RANSAC or not
  bool ransac;
  ///RANSAC Line parameters
  int ransacLineNumSamples;
  int ransacLineNumIterations;
  int ransacLineNumGoodFit;
  float ransacLineThreshold;
  float ransacLineScoreThreshold;
  bool ransacLineBinarize;
  ///half width to use for ransac window
  int ransacLineWindow;
  ///RANSAC Spline parameters
  int ransacSplineNumSamples;
  int ransacSplineNumIterations;
  int ransacSplineNumGoodFit;
  float ransacSplineThreshold;
  float ransacSplineScoreThreshold;
  bool ransacSplineBinarize;
  int ransacSplineWindow;
  ///degree of spline to use
  int ransacSplineDegree;
  ///use a spline or straight line
  bool ransacSpline;
  bool ransacLine;

  ///step used to pixelize spline in ransac
  float ransacSplineStep;

  ///Overlap threshold to use for grouping of bounding boxes
  float overlapThreshold;

  ///Angle threshold used for localization (cosine, 1: most restrictive,
  /// 0: most liberal)
  float localizeAngleThreshold;
  ///Number of pixels to go in normal direction for localization
  int localizeNumLinePixels;

  ///Angle threshold used for extending (cosine, 1: most restrictive,
  /// 0: most liberal)
  float extendAngleThreshold;
  ///Angle threshold from mean direction used for extending (cosine, 1:
  /// most restrictive, 0: most liberal)
  float extendMeanDirAngleThreshold;
  ///Number of pixels to go in tangent direction for extending
  int extendLinePixelsTangent;
  ///Number of pixels to go in normal direction for extending
  int extendLinePixelsNormal;
  ///Trehsold used for stopping the extending process (higher ->
  /// less extending)
  float extendContThreshold;
  ///Stop extending when number of deviating points exceeds this threshold
  int extendDeviationThreshold;
  ///Top point for extension bounding box
  int extendRectTop;
  ///	Bottom point for extension bounding box
  int extendRectBottom;

  ///Angle threshold used for extending (cosine, 1: most restrictive,
  /// 0: most liberal)
  float extendIPMAngleThreshold;
  ///Angle threshold from mean direction used for extending (cosine,
  /// 1: most restrictive, 0: most liberal)
  float extendIPMMeanDirAngleThreshold;
  ///Number of pixels to go in tangent direction for extending
  int extendIPMLinePixelsTangent;
  ///Number of pixels to go in normal direction for extending
  int extendIPMLinePixelsNormal;
  ///Trehsold used for stopping the extending process (higher ->
  /// less extending)
  float extendIPMContThreshold;
  ///Stop extending when number of deviating points exceeds this threshold
  int extendIPMDeviationThreshold;
  ///Top point for extension bounding box
  int extendIPMRectTop;
  ///	Bottom point for extension bounding box
  int extendIPMRectBottom;


  ///Number of pixels to go around the spline to compute score
  int splineScoreJitter;
  ///Ratio of spline length to use
  float splineScoreLengthRatio;
  ///Ratio of spline angle to use
  float splineScoreAngleRatio;
  ///Step to use for spline score computation
  float splineScoreStep;

  ///number of frames the track is allowed to be absent before deleting it
  int splineTrackingNumAbsentFrames;
  ///number of frames before considering the track good
  int splineTrackingNumSeenFrames;

  ///Angle threshold for merging splines (radians)
  float mergeSplineThetaThreshold;
  ///R threshold (distance from origin) for merginn splines
  float mergeSplineRThreshold;
  ///Mean Angle threshold for merging splines (radians)
  float mergeSplineMeanThetaThreshold;
  ///Mean R threshold (distance from origin) for merginn splines
  float mergeSplineMeanRThreshold;
  ///Distance threshold between spline cetroids for merging
  float mergeSplineCentroidThreshold;

  ///number of frames the track is allowed to be absent before deleting it
  int lineTrackingNumAbsentFrames;
  ///number of frames before considering the track good
  int lineTrackingNumSeenFrames;

  ///Angle threshold for merging lines (radians)
  float mergeLineThetaThreshold;
  ///R threshold (distance from origin) for merging lines
  float mergeLineRThreshold;

  ///Number of horizontal strips to divide the image to
  int numStrips;

  ///Whtethet to check splines or not
  bool checkSplines;
  ///Curveness Threshold for checking splines
  float checkSplinesCurvenessThreshold;
  ///Length Threshold for checking splines
  float checkSplinesLengthThreshold;
  ///ThetaDiff Threshold for checking splines
  float checkSplinesThetaDiffThreshold;
  ///ThetaThreshold Threshold for checking splines
  float checkSplinesThetaThreshold;

  ///Whtethet to check IPM splines or not
  bool checkIPMSplines;
  ///Curveness Threshold for checking splines
  float checkIPMSplinesCurvenessThreshold;
  ///Length Threshold for checking splines
  float checkIPMSplinesLengthThreshold;
  ///ThetaDiff Threshold for checking splines
  float checkIPMSplinesThetaDiffThreshold;
  ///ThetaThreshold Threshold for checking splines
  float checkIPMSplinesThetaThreshold;

  ///Final Threshold for declaring a valid spline
  float finalSplineScoreThreshold;

  ///Use ground plane when sending to map or not
  bool useGroundPlane;

  ///Whether to check colors or not
  bool checkColor;
  ///Size of window to use
  int checkColorWindow;
  ///Number of bins to use for histogram
  int checkColorNumBins;
  ///Min ratio of yellow points
  float checkColorNumYellowMin;
  ///Min RG diff
  float checkColorRGMin;
  ///Max RG diff
  float checkColorRGMax;
  ///Min GB diff
  float checkColorGBMin;
  ///Min RB diff
  float checkColorRBMin;
  ///RBF Threshold
  float checkColorRBFThreshold;
  ///Whether to use RBF or not
  bool checkColorRBF;

  ///Whether to clear part of the IPM image
  bool ipmWindowClear;
  ///Left corrdinate of window to keep in IPM
  int ipmWindowLeft;
  ///Left corrdinate of window to keep in IPM
  int ipmWindowRight;

  ///Whether to check lane width or not
  bool checkLaneWidth;
  ///Mean of lane width to look for
  float checkLaneWidthMean;
  ///Std deviation of lane width to look for
  float checkLaneWidthStd;
} LaneDetectorConf;

//function definitions
/**
 * This function gets a 1-D gaussian filter with specified
 * std deviation and range
 *
 * \param kernel input mat to hold the kernel (2*w+1x1)
 *      column vector (already allocated)
 * \param w width of kernel is 2*w+1
 * \param sigma std deviation
 */
void mcvGetGaussianKernel(CvMat *kernel, unsigned char w, FLOAT sigma);


/**
 * This function gets a 1-D second derivative gaussian filter
 * with specified std deviation and range
 *
 * \param kernel input mat to hold the kernel (2*w+1x1)
 *      column vector (already allocated)
 * \param w width of kernel is 2*w+1
 * \param sigma std deviation
 */
void mcvGet2DerivativeGaussianKernel(CvMat *kernel, unsigned char w,
                                     FLOAT sigma);


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

#define FILTER_LINE_HORIZONTAL 0
#define FILTER_LINE_VERTICAL 1
void mcvFilterLines(const CvMat *inImage, CvMat *outImage, unsigned char wx=2,
                    unsigned char wy=2, FLOAT sigmax=1, FLOAT sigmay=1,
                    LineType lineType=LINE_HORIZONTAL);


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
 */
#define HV_LINES_HORIZONTAL 0
#define HV_LINES_VERTICAL   1
void mcvGetHVLines(const CvMat *inImage, vector<Line> *lines,
                   vector<FLOAT> *lineScores, LineType lineType=LINE_HORIZONTAL,
                   FLOAT linePixelWidth=1., bool binarize=false,
                   bool localMaxima=false, FLOAT detectionThreshold=1.,
                   bool smoothScores=true);

/** This function binarizes the input image i.e. nonzero elements
 * become 1 and others are 0.
 *
 * \param inImage input & output image
 */
void mcvBinarizeImage(CvMat *inImage);

/** This function gets the maximum value in a vector (row or column)
 * and its location
 *
 * \param inVector the input vector
 * \param max the output max value
 * \param maxLoc the location (index) of the first max
 * \param ignore don't the first and last ignore elements
 *
 */
void mcvGetVectorMax(const CvMat *inVector, double *max, int *maxLoc,
                     int ignore=0);

/** This function gets the qtile-th quantile of the input matrix
 *
 * \param mat input matrix
 * \param qtile required input quantile probability
 * \return the returned value
 *
 */
FLOAT mcvGetQuantile(const CvMat *mat, FLOAT qtile);

/** This function thresholds the image below a certain value to the threshold
 * so: outMat(i,j) = inMat(i,j) if inMat(i,j)>=threshold
 *                 = threshold otherwise
 *
 * \param inMat input matrix
 * \param outMat output matrix
 * \param threshold threshold value
 *
 */
void mcvThresholdLower(const CvMat *inMat, CvMat *outMat, FLOAT threshold);

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
                     vector<float> *lineScores, const CameraInfo *cameraInfo,
                     LaneDetectorConf *stopLineConf);

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
void mcvLines2Mat(const vector<Line> *lines, CvMat *mat);

/** This function converts matrix into n array of lines
 *
 * \param mat input matrix , it has 2x2*size where size is the
 *  number of lines, first row is x values (start.x, end.x) and second
 *  row is y-values
 * \param  lines the rerurned vector of lines
 *
 *
 */
void mcvMat2Lines(const CvMat *mat, vector<Line> *lines);

/** This function intersects the input line with the given bounding box
 *
 * \param inLine the input line
 * \param bbox the bounding box
 * \param outLine the output line
 *
 */
void mcvIntersectLineWithBB(const Line *inLine, const CvSize bbox,
                            Line *outLine);

/** This function checks if the given point is inside the bounding box
 * specified
 *
 * \param inLine the input line
 * \param bbox the bounding box
 * \param outLine the output line
 *
 */
bool mcvIsPointInside(FLOAT_POINT2D point, CvSize bbox);

/** This function converts an INT mat into a FLOAT mat (already allocated)
 *
 * \param inMat input INT matrix
 * \param outMat output FLOAT matrix
 *
 */
void mcvMatInt2Float(const CvMat *inMat, CvMat *outMat);

/** This function draws a line onto the passed image
 *
 * \param image the input iamge
 * \param line input line
 * \param line color
 * \param width line width
 *
 */
void mcvDrawLine(CvMat *image, Line line, CvScalar color=CV_RGB(0,0,0),
                 int width=1);

/** This function draws a rectangle onto the passed image
 *
 * \param image the input image
 * \param rect the input rectangle
 * \param color the rectangle color
 * \param width the rectangle width
 *
 */
void mcvDrawRectangle (CvMat *image, CvRect rect,
                       CvScalar color=CV_RGB(255,0,0), int width=1);

/** This initializes the LaneDetectorinfo structure
 *
 * \param fileName the input file name
 * \param stopLineConf the structure to fill
 *
 *
 */
void mcvInitLaneDetectorConf(char * const fileName,
                            LaneDetectorConf *laneDetectorConf);

void SHOW_LINE(const Line line, char str[]="Line:");
void SHOW_SPLINE(const Spline spline, char str[]="Spline");

/** This fits a parabola to the entered data to get
 * the location of local maximum with sub-pixel accuracy
 *
 * \param val1 first value
 * \param val2 second value
 * \param val3 third value
 *
 * \return the computed location of the local maximum
 */
double mcvGetLocalMaxSubPixel(double val1, double val2, double val3);


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
                               bool group, FLOAT groupTthreshold);

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
                                  Line *outLine);

/** This function gets the local maxima in a matrix and their positions
 *  and its location
 *
 * \param inMat input matrix
 * \param localMaxima the output vector of local maxima
 * \param localMaximaLoc the vector of locations of the local maxima,
 *       where each location is cvPoint(x=col, y=row) zero-based
 * \param threshold threshold to return local maxima above
 *
 */
void mcvGetMatLocalMax(const CvMat *inMat, vector<double> &localMaxima,
                       vector<CvPoint> &localMaximaLoc, double threshold=0.0);

/** This function gets the locations and values of all points
 * above a certain threshold
 *
 * \param inMat input matrix
 * \param maxima the output vector of maxima
 * \param maximaLoc the vector of locations of the maxima,
 *       where each location is cvPoint(x=col, y=row) zero-based
 * \param threshold the threshold to get all points above
 *
 */
void mcvGetMatMax(const CvMat *inMat, vector<double> &maxima,
                  vector<CvPoint> &maximaLoc, double threshold);

/** This function gets the local maxima in a vector and their positions
 *
 * \param inVec input vector
 * \param localMaxima the output vector of local maxima
 * \param localMaximaLoc the vector of locations of the local maxima,
 *
 */
void mcvGetVectorLocalMax(const CvMat *inVec, vector<double> &localMaxima,
                          vector<int> &localMaximaLoc);

/** This functions implements Bresenham's algorithm for getting pixels of the
 * line given its two endpoints

 *
 * \param line the input line
  *
 */
//void mcvGetLinePixels(const Line &line, vector<int> &x, vector<int> &y)
CvMat * mcvGetLinePixels(const Line &line);

/** This functions implements Bresenham's algorithm for getting pixels of the
 * line given its two endpoints

 *
 * \param line the input line
 * \param x a vector of x locations for the line pixels (0-based)
 * \param y a vector of y locations for the line pixels (0-based)
 *
 */
//void mcvGetLinePixels(const Line &line, vector<int> &x, vector<int> &y);

/** This functions implements Bresenham's algorithm for getting pixels of the
 * line given its two endpoints
 *
 *
 * \param im the input image
 * \param inLine the input line
 * \param outLine the output line
 *
 */
void mcvGetLineExtent(const CvMat *im, const Line &inLine, Line &outLine);

/** This functions converts a line defined by its two end-points into its
 *  r and theta (origin is at top-left corner with x right and y down and theta
 *  measured positive clockwise<with y pointing down> -pi<theta<pi)
 *
 *
 * \param line input line
 * \param r the returned r (normal distance to the line from the origin)
 * \param outLine the output line
 *
 */
void mcvLineXY2RTheta(const Line &line, float &r, float &theta);


/** This function checks if the given point is inside the rectangle specified
 *
 * \param inLine the input line
 * \param rect the specified rectangle
 *
 */
bool mcvIsPointInside(FLOAT_POINT2D &point, const Line &rect);

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
                                    Line &outLine);


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
                 LineState* state = NULL);


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
void mcvGetLines(const CvMat* image, LineType lineType, vector<Line> &lines,
                 vector<float> &lineScores, vector<Spline> &splines,
                 vector<float> &splineScores, LaneDetectorConf *lineConf,
                 LineState *state);

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
                         IPMInfo &ipmInfo, CameraInfo &cameraInfo);

/** This function gets the indices of the non-zero values in a matrix

 * \param inMat the input matrix
 * \param outMat the output matrix, with 2xN containing the x and y in
 *    each column
 * \param floatMat whether to return floating points or integers for
 *    the outMat
 */
CvMat* mcvGetNonZeroPoints(const CvMat *inMat, bool floatMat);

/** This functions implements RANSAC algorithm for line fitting
    given an image

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
                      Line *lineXY, float *lineRTheta, float *lineScore);

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
void mcvFitRobustLine(const CvMat *points, float *lineRTheta, float *lineAbc);

/** This function groups nearby lines
 *
 * \param inLines vector of lines
 * \param outLines vector of grouped lines
 * \param groupThreshold the threshold used for grouping
 * \param bbox the bounding box to intersect with
 */
void mcvGroupLines(vector<Line> &lines, vector<float> &lineScores,
                   float groupThreshold, CvSize bbox);


/** This function performs a RANSAC validation step on the detected lines
 *
 * \param im the input image
 * \param lines vector of input lines
 * \param lineScores the scores of input lines
 * \param lineConf the parameters controlling its operation
 * \param lineType the type of line to work on (LINE_HORIZONTAL or
 *    LINE_VERTICAL)
 */
void mcvGetRansacLines(const CvMat *im, vector<Line> &lines,
                       vector<float> &lineScores, LaneDetectorConf *lineConf,
                       LineType lineType);


/** This function sets the matrix to a value except for the mask window passed
 * in
 *
 * \param inMat input matrix
 * \param mask the rectangle defining the mask: (xleft, ytop, width, height)
 * \param val the value to put
 */
void  mcvSetMat(CvMat *inMat, CvRect mask, double val);

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
                        CameraInfo &cameraInfo, CvSize imSize);

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
                      CameraInfo &cameraInfo, CvSize imSize);

/** This function draws a spline onto the passed image
 *
 * \param image the input iamge
 * \param spline input spline
 * \param spline color
 *
 */
void mcvDrawSpline(CvMat *image, Spline spline, CvScalar color, int width);


/** This functions implements RANSAC algorithm for spline fitting
    given an image

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
 *      score
 * \param splineScoreLengthRatio Ratio of spline length to use
 * \param splineScoreAngleRatio Ratio of spline angle to use
 * \param splineScoreStep Step to use for spline score computation
 * \param prevSplines the splines from the previous frame, to use as initial
 *      seeds
 *   pass NULL to ignore this input
 *
 */
void mcvFitRansacSpline(const CvMat *image, int numSamples, int numIterations,
                        float threshold, float scoreThreshold, int numGoodFit,
                        int splineDegree, float h, Spline *spline,
                        float *splineScore, int splineScoreJitter,
                        float splineScoreLengthRatio,
                        float splineScoreAngleRatio, float splineScoreStep,
                        vector<Spline> *prevSplines = NULL);

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
                         vector<float> &splineScores, LineState* state);

/** This function returns pixel coordinates for the Bezier
 * spline with the given resolution.
 *
 * \param spline input spline
 * \param h the input resolution
 * \param box the bounding box
 * \param extendSpline whether to extend spline with straight lines or not
 *    (default false)
 * \return computed points in an array Nx2 [x,y], returns NULL if empty output
 */
CvMat* mcvGetBezierSplinePixels(Spline &spline, float h, CvSize box,
                                bool extendSpline=false);


/** This function evaluates Bezier spline with given resolution
 *
 * \param spline input spline
 * \param h the input resolution
 * \param tangents the tangents at the two end-points of the spline [t0; t1]
 * \return computed points in an array Nx2 [x,y]
 */
CvMat* mcvEvalBezierSpline(const Spline &spline, float h, CvMat *tangents=NULL);

/** This function fits a Bezier spline to the passed input points
 *
 * \param points the input points
 * \param degree the required spline degree
 * \return spline the returned spline
 */
Spline mcvFitBezierSpline(CvMat *points, int degree);

/** This function fits a Bezier spline to the passed input points
 *
 * \param inPOints Nx2 matrix of points [x,y]
 * \param outPOints Nx2 matrix of points [x,y]
 * \param dim the dimension to sort on (0: x, 1:y)
 * \param dir direction of sorting (0: ascending, 1:descending)
 */
void mcvSortPoints(const CvMat *inPoints, CvMat *outPoints,
		   int dim, int dir);

/** This function samples uniformly with weights
 *
 * \param cumSum cumulative sum for normalized weights for the differnet
 *    samples (last is 1)
 * \param numSamples the number of samples
 * \param randInd a 1XnumSamples of int containing the indices
 * \param rng a pointer to a random number generator
 *
 */
void mcvSampleWeighted(const CvMat *cumSum, int numSamples,
                       CvMat *randInd, CvRNG *rng);


/** This function computes the cumulative sum for a vector
 *
 * \param inMat input matrix
 * \param outMat output matrix
 *
 */
void mcvCumSum(const CvMat *inMat, CvMat *outMat);

/** This functions gives better localization of points along lines
 *
 * \param im the input image
 * \param inPoints the input points Nx2 matrix of points
 * \param outPoints the output points Nx2 matrix of points
 * \param numLinePixels Number of pixels to go in normal direction for
 *    localization
 * \param angleThreshold Angle threshold used for localization
 *        (cosine, 1: most restrictive, 0: most liberal)
 *
 */
void mcvLocalizePoints(const CvMat *im, const CvMat *inPoints,
                       CvMat *outPoints, int numLinePixels,
                       float angleThreshold);

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
                     vector<CvPoint2D32f> &peaks, vector<float> &peakVals,
                     bool positivePeak=true, bool smoothPeaks=true);

/** This functions chooses the best peak that minimizes deviation
 * from the tangent direction given
 *
 * \param peaks the peaks found
 * \param peakVals the values for the peaks
 * \param peak the returned peak
 * \param peakVal the peak value for chosen peak, -1 if nothing
 * \param contThreshold the threshold to get peak above
 * \param tangent the tangent line along which the peak was found normal
 *          to (normalized)
 * \param prevPoint the previous point along the tangent
 * \param angleThreshold the angle threshold to consider for valid peaks
 * \return index of peak chosen, -1 if nothing
 *
 */
int mcvChooseBestPeak(const vector<CvPoint2D32f> &peaks,
                      const vector<float> &peakVals, CvPoint2D32f &peak,
                      float &peakVal, float contThreshold,
                      const CvPoint2D32f &tangent,
                      const CvPoint2D32f &prevPoint, float angleThreshold);

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
                        CvRect bbox, bool smoothPeaks=true);

/** This functions extends a point along the tangent and gets the normal line
 * at the new point
 *
 * \param curPoint the current point to extend
 * \param tangent the tangent at this point (not necessarily normalized)
 * \param linePixelsTangent the number of pixels to go in tangent direction
 * \param linePixelsNormal the number of pixels to go in normal direction
 * \return the normal line at new point
 */
Line mcvGetExtendedNormalLine(CvPoint2D32f &curPoint, CvPoint2D32f &tangent,
                              int linePixelsTangent, int linePixelsNormal,
                              CvPoint2D32f &nextPoint);

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
                    const CvPoint2D32f &prevPoint, float angleThreshold);



/** \brief This function groups together bounding boxes
 *
 * \param size the size of image containing the lines
 * \param boxes a vector of output grouped bounding boxes * \param type the
 *          type of lines (LINE_HORIZONTAL or LINE_VERTICAL)
 * \param groupThreshold the threshold used for grouping (ratio of overlap)
 */
void mcvGroupBoundingBoxes(vector<CvRect> &boxes, LineType type,
                           float groupThreshold);


/** This functions normalizes the given vector
 *
 * \param vector the input vector to normalize
 */
CvPoint2D32f mcvNormalizeVector(const CvPoint2D32f &v);

/** This functions normalizes the given vector
 *
 * \param vector the input vector to normalize
 */
CvPoint2D32f mcvNormalizeVector(const CvPoint &v);


/** This functions normalizes the given vector
 *
 * \param x the x component
 * \param y the y component
 */
CvPoint2D32f mcvNormalizeVector(float x, float y);


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
                        int  jitterVal, float lengthRatio,
                        float angleRatio);

/** This functions returns a vector of jitter from the input maxJitter value
 * This is used for computing spline scores for example, to get scores
 * around the rasterization of the spline
 *
 * \param maxJitter the max value to look around
 *
 * \return the required vector of jitter values
 */
vector<int> mcvGetJitterVector(int maxJitter);

/** This functions adds two vectors and returns the result
 *
 * \param v1 the first vector
 * \param v2 the second vector
 * \return the sum
 */
CvPoint2D32f mcvAddVector(CvPoint2D32f v1, CvPoint2D32f v2);

/** This functions gets the average direction of the set of points
 * by computing the mean vector between points
 *
 * \param points the input points [Nx2] matrix
 * \param forward go forward or backward in computation (default true)
 * \return the mean direction
 *
 */
CvPoint2D32f  mcvGetPointsMeanVector(const CvMat *points, bool forward = true);


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
                          float MeanThetaThreshold, float MeanRThreshold,
                          float centroidThreshold);


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
void mcvGetSplineFeatures(const Spline& spline, CvPoint2D32f* centroid=0,
                          float* theta=0, float* r=0, float* length=0,
                          float* meanTheta=0, float* meanR=0,
                          float* curveness=0);

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
void mcvGetPointsFeatures(const CvMat* points,
                          CvPoint2D32f* centroid=0,
                          float* theta=0, float* r=0,
                          float* length=0, float* meanTheta=0,
                          float* meanR=0, float* curveness=0);


/** This functions computes difference between two vectors
 *
 * \param v1 first vector
 * \param v2 second vector
 * \return difference vector v1 - v2
 *
 */
CvPoint2D32f  mcvSubtractVector(const CvPoint2D32f& v1, const CvPoint2D32f& v2);

/** This functions computes the vector norm
 *
 * \param v input vectpr
 * \return norm of the vector
 *
 */
float  mcvGetVectorNorm(const CvPoint2D32f& v);


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
                        float thetaThreshold, float rThreshold);


/** This functions converts a line to a spline
 *
 * \param line the line
 * \param degree the spline degree
 *
 * \return the returned spline
 *
 */
Spline mcvLineXY2Spline(const Line& line, int degree);

/** This function checks if the given point is inside the rectangle specified
 *
 * \param inLine the input line
 * \param rect the specified rectangle
 *
 */
bool mcvIsPointInside(FLOAT_POINT2D &point, const CvRect &rect);


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
                 float size=.25f, CvScalar color=CV_RGB(1,1,1));

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
                   float thetaThreshold);


/** This function makes some checks on points and decides
 * whether to keep them or not
 *
 * \param points the array of points to check
 *
 * \return code that determines what to do with the points
 *
 */
int mcvCheckPoints(const CvMat* points);

/** This functions gets the angle of the line with the horizontal
 *
 * \param line the line
 *
 * \return the required angle (radians)
 *
 */
float mcvGetLineAngle(const Line& line);

/** This function groups nearby splines
 *
 * \param splines vector of splines
 * \param lineScores scores of input lines
 */
void mcvGroupSplines(vector<Spline> &splines, vector<float> &scores);

/** This functions multiplies a vector by a scalar
 *
 * \param v the vector
 * \param s the scalar
 * \return the sum
 */
CvPoint2D32f mcvMultiplyVector(CvPoint2D32f v, float s);


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
                            bool rbf, float rbfThreshold);

/** \brief This function extracts bounding boxes from splines
 *
 * \param splines vector of splines
 * \param type the type of lines (LINE_HORIZONTAL or LINE_VERTICAL)
 * \param size the size of image containing the lines
 * \param boxes a vector of output bounding boxes
 */
void mcvGetSplinesBoundingBoxes(const vector<Spline> &splines, LineType type,
                                CvSize size, vector<CvRect> &boxes);


/** \brief This function extracts bounding boxes from lines
 *
 * \param lines vector of lines
 * \param type the type of lines (LINE_HORIZONTAL or LINE_VERTICAL)
 * \param size the size of image containing the lines
 * \param boxes a vector of output bounding boxes
 */
void mcvGetLinesBoundingBoxes(const vector<Line> &lines, LineType type,
                              CvSize size, vector<CvRect> &boxes);


/** \brief This function takes a bunch of lines, and check which
 * 2 lines can make a lane
 *
 * \param lines vector of lines
 * \param scores vector of line scores
 * \param wMu expected lane width
 * \param wSigma std deviation of lane width
 */
void mcvCheckLaneWidth(vector<Line> &lines, vector<float> &scores,
                       float wMu, float wSigma);

} // namespace LaneDetector

#endif /*LANEDETECTOR_HH_*/
