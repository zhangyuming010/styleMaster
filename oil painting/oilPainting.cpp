#include "oilPainting.h"

using namespace std;
using namespace cv;

void oilPainting::printmat(Mat A)//print Mat
{

	cout<<"Mat:"<<endl;

	cout<<"r(default) = "<<"\n"<<A<<" , "<<endl<<endl;
}


bool  oilPainting::LoadImg(string img,string save_to,double lambda1 ,double kappa1)
{
	ImgSrc = imread(img/*,CV_LOAD_IMAGE_GRAYSCALE*/);
	lambda = lambda1;
	kappa = kappa1;
	if( ImgSrc.empty() )
	{
		//printf("Cannot read image file: %s\n");
		return false;
	}
	Rect rect(0,0,ImgSrc.cols,ImgSrc.rows);        
	ImgSrc(rect).copyTo(ImgSrc);  
	ImgSrc.convertTo(ImgSrc,CV_64F,1.0/255,0);//equivalent to im2double in matlab

	ImgHeight = ImgSrc.rows;
	ImgWidth = ImgSrc.cols;
	//printmat(ImgSource);
	Channels = ImgSrc.channels();
	if(Channels>0)
	{
		//change color space
		//cvtColor(ImgSrc, ImgSrc, cv::COLOR_BGRA2BGR);
		std::vector<cv::Mat> mv;
		//split channels
	     split(ImgSrc, mv);

		#pragma omp parallel sections
		 {
			 #pragma omp section
				 mv[0]= Convert(mv[0]);
			 #pragma omp section
				 mv[1]= Convert(mv[1]);
			 #pragma omp section
				 mv[2]= Convert(mv[2]);
		 }
		 merge(mv, ImgSrc);
	}

	ImgSrc.convertTo(ImgSrc,CV_64F,255,0);
    imwrite( save_to,ImgSrc);
	return true;

}


bool oilPainting::StoreImg(string save_image_to)
{
	if (oilPainting::ImgResult.empty())
		return false;
	imwrite( save_image_to,oilPainting::ImgResult);
	return true;
}

void oilPainting::RowShift(Mat &A)
{

	Mat B(Size(A.cols,A.rows),CV_64F);
	A.row(0).copyTo(B.row(B.rows-1));

	#pragma omp parallel for
	for(int k=1;k<A.rows;k++)
		A.row(k).copyTo(B.row(k-1));
	//printmat(B);
	B.copyTo(A);
	//printmat(A);
}

void oilPainting::ColShift(Mat &A)
{
	Mat B(Size(A.cols,A.rows),CV_64F);
	A.col(0).copyTo(B.col(B.cols-1));

	#pragma omp parallel for
	for(int k=1;k<A.cols;k++)
		A.col(k).copyTo(B.col(k-1));

	B.copyTo(A);
	//printmat(A);
}

void oilPainting::CircShift2Center(Mat &A,int row,int col)
{
	
	int w_shift = col/2;
	int h_shift = row/2;

	#pragma omp parallel
	{
		#pragma omp for
	for(int i=0;i<w_shift;i++)
		ColShift(A);
		#pragma omp for
	for(int j=0;j<h_shift;j++)
		RowShift(A);
	}
	//printmat(A);
}

Mat oilPainting::MyPsf2Otf(Mat &A,int a,int b)
{
	Mat B= Mat::zeros(/*ImgSource.size()*/a,b, CV_64F);

	copyMakeBorder(A, B, 0, B.rows-A.rows, 0, B.cols-A.cols, BORDER_CONSTANT, Scalar::all(0));//fill in 0
	//printmat(B);
	CircShift2Center(B,A.rows,A.cols);		
	//printmat(B);

	Mat planes[] = {Mat_<float>(B), Mat::zeros(B.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);

	dft(complexImg, complexImg);

	split(complexImg, planes);
	
	//printmat(planes[0]);//real part
	//printmat(planes[1]);//image part

	magnitude(planes[0], planes[1], planes[0]);
	return planes[0];

}

Mat oilPainting::Myfft2(Mat &A)
{
	if( A.empty() )
	{

		printf("Cannot read image file: %s\n");
		//return false;
	}
	Mat padded;
	/*int M = getOptimalDFTSize( A.rows );
	int N = getOptimalDFTSize( A.cols );
	copyMakeBorder(A, padded, 0, M - A.rows, 0, N - A.cols, BORDER_CONSTANT, Scalar::all(0));*/
	A.copyTo(padded);
	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg,CV_DXT_FORWARD);
	//printmat(complexImg);
	return complexImg;

}

Mat oilPainting::H_Diff1(Mat &A)
{
	Mat B(Size(A.cols+1,A.rows),CV_64F);
	#pragma omp parallel for
	for(int i=0;i<A.cols;i++)
	     A.col(i).copyTo(B.col(i));

	A.col(0).copyTo(B.col(A.cols));

	#pragma omp parallel for
	for(int i=0;i<B.cols-1;i++)
		B.col(i)=B.col(i+1)-B.col(i);
//	printmat(B);
	Mat C = B(Range::all(), Range(0, A.cols));
	return C;
}

Mat oilPainting::H_Diff2(Mat &A)
{
	
	Mat B(Size(A.cols+1,A.rows),CV_64F);

	#pragma omp parallel for
	for(int i=0;i<A.cols;i++)
		A.col(i).copyTo(B.col(i+1));

	A.col(A.cols-1).copyTo(B.col(0));

	#pragma omp parallel for
	for(int i=0;i<B.cols-1;i++)
		B.col(i)=B.col(i)-B.col(i+1);

	//	printmat(B);
	Mat C = B(Range::all(), Range(0, A.cols));
	return C;
}



Mat oilPainting::V_Diff1(Mat &A)
{
	
	Mat B(Size(A.cols,A.rows+1),CV_64F);

	#pragma omp parallel for
	for(int i=0;i<A.rows;i++)
		A.row(i).copyTo(B.row(i));

	A.row(0).copyTo(B.row(A.rows));

	#pragma omp parallel for
	for(int i=0;i<B.rows-1;i++)
		B.row(i)=B.row(i+1)-B.row(i);

	//	printmat(B);

	Mat C = B(Range(0, A.rows),Range::all());
	return C;
}

Mat oilPainting::V_Diff2(Mat &A)
{
	Mat B(Size(A.cols,A.rows+1),CV_64F);

	#pragma omp parallel for
	for(int i=0;i<A.rows;i++)
		A.row(i).copyTo(B.row(i+1));

	A.row(A.rows-1).copyTo(B.row(0));

	#pragma omp parallel for
	for(int i=0;i<B.rows-1;i++)
		B.row(i)=B.row(i)-B.row(i+1);
	//	printmat(B);

	Mat C = B(Range(0, A.rows),Range::all());
	return C;
}

Mat oilPainting::Convert(Mat ImgSource)
{
	//if(!LoadImg(Im))
	//{
	//	cout<<"error"<<endl;
	//	return false;
	//}	
	Mat fx=(Mat_<double>(1,2) << 1,-1);
	Mat fy=(Mat_<double>(2,1) << 1,-1);

	Mat otfFx(ImgSource.size(),CV_64F);
	Mat otfFy(ImgSource.size(),CV_64F);

	otfFx =MyPsf2Otf(fx,ImgHeight,ImgWidth);
	otfFy =MyPsf2Otf(fy,ImgHeight,ImgWidth);

	multiply(otfFx,otfFx,otfFx);
	multiply(otfFy,otfFy,otfFy);

	Mat Denormin2=otfFx+otfFy;

	//printmat(Denormin2);

	Mat Normin1= Myfft2(ImgSource);

	//printmat(Normin1);

	Mat Denormin;
	Mat t;
	Mat h;
	Mat v ;

	double beta=2*lambda;

	while(beta<betamax)
	{
		Denormin = 1.0 +beta*Denormin2;

		h = H_Diff1(ImgSource);
		v = V_Diff1(ImgSource);
		
		Mat h_2;
		Mat v_2;

		multiply(h,h,h_2);
		multiply(v,v,v_2);

		t= h_2+v_2;
		t=t-lambda/beta;
		t=t/abs(t);
	
		Mat h1,v1;
		multiply(h,t,h1);
		h=(h + h1)/2;
		multiply(v,t,v1);
		v=(v+v1)/2;
		
		Mat Normin2 = H_Diff2(h);
		Normin2 = Normin2 + V_Diff2(v);

		//
		Mat FS = Normin1+beta*Myfft2(Normin2);
	    Mat plane1[2] = {Mat_<float>(FS), Mat::zeros(FS.size(), CV_64F)};
		split(FS,plane1);
	
		Mat Denormin_temp;
		plane1[0] = plane1[0]/Denormin;
		plane1[1] = plane1[1]/Denormin;		
	    merge(plane1, 2, FS);

		dft(FS, ImgSource, DFT_SCALE|DFT_INVERSE);//ifft2

		Mat planes[] = {Mat_<float>(ImgSource), Mat::zeros(ImgSource.size(), CV_64F)};

		split(ImgSource, planes);

		ImgSource = planes[0];
		
		beta = beta*kappa;
	}

	return ImgSource;

}