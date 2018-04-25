#include"camera.h"
#include<iostream>
#include<unistd.h>
using namespace std;
#define	CAMERA_WIDTH	2592
#define CAMERA_HEIGHT	1944
VideoCapture capture1, capture2,capture3;
Mat frame1, frame2,frame3;

void OpenCamera()
{
	capture1.open(0);
	capture1.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
	capture1.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
	//capture1.release();
	capture2.open(1);
	capture2.set(CV_CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
	capture2.set(CV_CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
	//capture2.release();
	int i=0;
	while (i++>10)
	{
		capture1 >> frame1;
		capture2 >> frame2;
		usleep(30000);
	}
}
void CaptureOnce(Mat& img1,Mat& img2)
{
	namedWindow("1", CV_WINDOW_NORMAL);
	namedWindow("2", CV_WINDOW_NORMAL);

	capture1.open(0);
	capture1.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	capture1.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);
	//capture1.release();
	capture2.open(2);
	capture2.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);
	//capture2.release();

	while (1)
	{
		capture1 >> frame1;
		imshow("1", frame1);

		capture2 >> frame2;
		imshow("2", frame2);

		if (waitKey(30) == 27) {
			//imwrite("1.jpg", frame);
			break;
		}
	}
	img1 = frame1;
	img2 = frame2;

	destroyWindow("1");
	destroyWindow("2");
}

Mat CaptureOnceOne()
{
	Mat img;
	namedWindow("1", CV_WINDOW_NORMAL);

	capture1.open(1);
	capture1.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	capture1.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);
	//capture1.release();

	while (1)
	{
		capture1 >> frame1;
		imshow("1", frame1);

		if (waitKey(30) == 27) {
			//imwrite("1.jpg", frame);
			break;
		}
	}
	img = frame1;

	destroyWindow("1");
	return img;
}

void CaptureOnce3(Mat& img1, Mat& img2, Mat& img3)
{
	int b1, b2, b3,i=0;;

	capture1 >> frame1;
	capture2 >> frame2;
	// b1 = capture1.open(0);
	// capture1.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	// capture1.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);
	// while (1)
	// {
	// 	if(!b1) continue;
	// 	capture1 >> frame1;
	// 	if(frame1.empty()) continue;
		
	// 	if (waitKey(50) == 27 || i++>10) {
	// 		break;
	// 	}
	// }
	// capture1.release();
	// b2 = capture2.open(1);
	// capture2.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	// capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);
	// while (1)
	// {
	// 	if(!b2) continue;
	// 	capture2 >> frame2;
	// 	if(frame2.empty()) continue;

	// 	if (waitKey(50) == 27|| i++>20) {
	// 		break;
	// 	}
	// }
	capture2.release();
	b3 = capture3.open(2);
	capture3.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	capture3.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);
	while(!capture3.isOpened()) cout<<"open fail"<<endl;
	while (i++<20)
	{
		if(!b3) continue;
		capture3 >> frame3;
		if(frame3.empty()) continue;
		usleep(30000);
	}
	capture3.release();

	img1 = frame1;
	img2 = frame2;
	img3 = frame3;

	capture2.open(1);
	capture2.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);

}

void Capture3( Mat& img1, Mat& img2, Mat& img3)
{
	int b1, b2, b3;
	namedWindow("1", CV_WINDOW_NORMAL);
	namedWindow("2", CV_WINDOW_NORMAL);
	namedWindow("3", CV_WINDOW_NORMAL);

	b1 = capture1.open(1);
	capture1.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	capture1.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);

	b2 = capture2.open(2);
	capture2.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	capture2.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);

	b3 = capture3.open(3);
	capture3.set(CV_CAP_PROP_FRAME_WIDTH, 2592);
	capture3.set(CV_CAP_PROP_FRAME_HEIGHT, 1944);

	while (1)
	{
		capture1 >> frame1;
		imshow("1", frame1);
		capture2 >> frame2;
		imshow("2", frame2);
		capture3 >> frame3;
		imshow("3", frame3);


		if (waitKey(30) == 27) {
			//imwrite("1.jpg", frame);

			break;
		}
	}
	img1 = frame1;
	img2 = frame2;
	img3 = frame3;

	destroyWindow("1");
	destroyWindow("2");
	destroyWindow("3");

	capture1.release();
	capture2.release();
	capture3.release();
}