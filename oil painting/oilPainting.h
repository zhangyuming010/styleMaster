/*
This is an implementation of this paper [1], which can convert an image to an oil painting style.

[1] Xu L, Lu C, Xu Y, et al. Image smoothing via L 0 gradient minimization[C]//ACM Transactions on Graphics (TOG). ACM, 2011, 30(6): 174.

*/
#ifndef STYLESVT_H
#define STYLESVT_H
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>
#include "opencv2/opencv.hpp" 
#include <stdio.h>
using namespace std;
using namespace cv;

class oilPainting
{

private:
	Mat ImgSrc;
	Mat ImgResult;
	double betamax;
	int ImgWidth;
	int ImgHeight;
	int ColorDim;
	int Channels;
	double kappa;
	double lambda;

public:
	void printmat(Mat A);//
	void RowShift(Mat &A);
	void ColShift(Mat &A);
	void CircShift2Center(Mat &A,int row,int col);//center shift
	Mat MyPsf2Otf(Mat &A,int a,int b);
	Mat Myfft2(Mat &A);
	Mat H_Diff1(Mat &A);
	Mat V_Diff1(Mat &A);
	Mat H_Diff2(Mat &A);
	Mat V_Diff2(Mat &A);
	
	int getCols(){return ImgWidth;}
	int getRows(){return ImgHeight;}
	bool LoadImg(string img,string save_to,double lambda1=0.02 ,double kappa1 = 4.0);
	Mat GetSrcImg(){return ImgSrc;}
	Mat GetResultImg(){return ImgResult;}
	bool StoreImg(string save_image_to);
	Mat Convert(Mat Im);
	oilPainting(){betamax=1000;}

};
#endif