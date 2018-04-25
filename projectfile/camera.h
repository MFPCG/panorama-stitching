#pragma once
#include<opencv2/opencv.hpp>
using namespace cv;

extern VideoCapture capture1, capture2;
extern Mat frame1,frame2;


void OpenCamera(void);
//void OpenCameraOne(void);
void CaptureOnce(Mat& img1, Mat& img2);
void CaptureOnce3(Mat& img1, Mat& img2, Mat& img3);
Mat CaptureOnceOne();
void Capture3(Mat& img1, Mat& img2, Mat& img3);
