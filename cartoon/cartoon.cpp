#include "cartoon.h"

bool StyleCartoon::MyLoadImg(string imageSrc,string output, int bilateralPara, float edgePara, int quantinizePara)
{
// this parameter can be ajusted
	Mat labPlanes[3];
	Mat bilateralResult;

	double start = omp_get_wtime( );
	originalImage = imread(imageSrc);  //8 bit image
	if( originalImage.empty() )
	{
		cout << "Cannot read image file:" << imageSrc << endl;
		return false;
	}
	//imshow("originalImage", originalImage);
	//moveWindow("originalImage", 0, 0);
	//waitKey(0);
 // the last three parameter value are consult to opencv_tutorials Page 161
	bilateralFilter(originalImage, bilateralResult, bilateralPara, bilateralPara*2, bilateralPara/2);
	/*
	imshow("bilateralFilter", bilateralResult);
	moveWindow("bilateralFilter", 550, 550);*/
// change to labPlanes
	cvtColor(bilateralResult, bilateralResult, CV_BGR2Lab);
	split(bilateralResult, labPlanes);

	if ( false == EdgeDetect(labPlanes[0], sobelResult, edgePara) )
	{
		cout << "EdgeDetect error !" << endl;
		return false;
	}
	//imshow("sobelResult", sobelResult);
	//moveWindow("sobelResult", 0, 550);
	//waitKey(0);
	if ( false == QuantinizeLuminance(labPlanes[0],quantinizePara) )
	{
		cout << "QuantinizeLuminance error !" << endl;
		return false;
	}
	merge(labPlanes, 3, resultImage);
	cvtColor(resultImage, resultImage, CV_Lab2BGR);
	
	if ( false == AddEdgeToResult() )
	{
		cout << "AddEdgeToResult error !" << endl;
		return false;
	}

	double end = omp_get_wtime( );
	cout << "cost time£º"<< end -start <<"\n";
	resultImage.convertTo(resultImage,CV_8U,255.0,0);
	//equalizeHist(resultImage, resultImage);
	imwrite(output, resultImage);


	//waitKey(0);
	return true;
}

bool StyleCartoon::EdgeDetect(Mat labPlanesZero, Mat &sobelResult, float edgePara)
{
	Mat sobelHorizontal, sobelVertical;
	float horizontalSquare, verticalSquare;
	float maxSobelValue = 0;
// this can be an ajust parameter
	float minimumEdgeStrength = edgePara;

	labPlanesZero.convertTo(sobelHorizontal, CV_32FC1, 1.0/255, 0);
	labPlanesZero.convertTo(sobelVertical, CV_32FC1, 1.0/255, 0);
	sobelResult.create(GetHeight(),GetWidth(), CV_32FC1);
	Sobel(sobelHorizontal, sobelHorizontal, -1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(sobelVertical, sobelVertical, -1, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	//printf( "rows:%d, cols:%d\n", GetHeight(), GetWidth());
	// here is not possible to use omp 
	for (int i=0; i<GetHeight(); ++i)
	{
		for (int j=0; j<GetWidth(); ++j)
		{
			horizontalSquare = sobelHorizontal.at<float>(i,j) * sobelHorizontal.at<float>(i,j);
			verticalSquare = sobelVertical.at<float>(i,j) * sobelVertical.at<float>(i,j);
			sobelResult.at<float>(i,j) = sqrt(horizontalSquare + verticalSquare);
			
			if ( sobelResult.at<float>(i,j) < minimumEdgeStrength )
			{
				sobelResult.at<float>(i,j) = 0;
			}
			sobelResult.at<float>(i,j) = 1- sobelResult.at<float>(i,j);
		}
	}

	return true;
}

bool StyleCartoon::QuantinizeLuminance(Mat &labPlanesZero, int quantinizePara)
{
	int quantinizeLevel = quantinizePara;
	float quantinizeTemp = 0, quantinizeParameter = 0;

	quantinizeParameter = (float)100/(quantinizeLevel-1);
#pragma omp parallel for
	for ( int i=0; i<GetHeight(); ++i )
	{
	
		for ( int j=0; j<GetWidth(); ++j )
		{
			quantinizeTemp = quantinizeParameter*(floor(labPlanesZero.at<unsigned char>(i,j)/quantinizeParameter));
			labPlanesZero.at<unsigned char>(i,j) = (unsigned char)quantinizeTemp;
		}
	}

	return true;
}

bool StyleCartoon::AddEdgeToResult()
{
	Mat rgbPlanes[3];

	resultImage.convertTo(resultImage, CV_32FC1, 1.0/255, 0);
	split(resultImage, rgbPlanes);
#pragma omp parallel for
	for (int i=0; i<3; ++i)
	{
		rgbPlanes[i] = rgbPlanes[i].mul(sobelResult);
	}
	merge(rgbPlanes, 3, resultImage);

	return true;
}