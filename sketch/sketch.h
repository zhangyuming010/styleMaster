#ifndef STYLESUMIAO_H
#define STYLESUMIAO_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;

class StyleSumiao
{
private:
	Mat originalImageGray, reverseOriginalImage;
	Mat dftReverseImage, invDftReverseImage;
	Mat resultImage, resultReferenceImage, resultImageMix;
	Mat planes[2];
	ofstream outFile;
	//int imageWidth, imageHeight;

public:
	bool MyLoadImg(string imageSrc, string imageOut, int r=15, float parameter=1.03);
	bool InverseColor(Mat &reverseOriginalImage);
	bool DFTImage(Mat reverseOriginalImageTemp, Mat &dftReverseImage);
	bool DFTShift(Mat dftReverseImage);
	bool InverseDFTShift(Mat &invDftReverseImage);
	bool GaussianFilter(int r);
	bool MixImages(float parameter);
	int GetWidth(){ return originalImageGray.cols; };
	int GetHeight(){ return originalImageGray.rows; };

};





#endif