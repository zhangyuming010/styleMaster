/*
This class convet an image to a cartoon style, image is worked on LAB color space

*/
#ifndef STYLE_CARTOON_H
#define STYLE_CARTOON_H
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace std;
using namespace cv;

class StyleCartoon
{
private:
	Mat originalImage;
	Mat sobelResult;
	Mat resultImage;
	//Mat rgbPlanes[3];
	//ofstream outFile;
	//int imageWidth, imageHeight;

public:
	bool MyLoadImg(string imageSrc,string output, int bilateralPara=16, float edgePara=0.3,  int quantinizePara=8);
	bool EdgeDetect(Mat labPlanesZero, Mat &sobelResult, float edgePara);
	bool QuantinizeLuminance(Mat &labPlanesZero,  int quantinizePara);
	bool AddEdgeToResult();

	int GetWidth(){ return originalImage.cols; };
	int GetHeight(){ return originalImage.rows; };

};


#endif