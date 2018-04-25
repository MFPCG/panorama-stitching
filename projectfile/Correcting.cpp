//注意：原文中公式14，15，17，18，22，23有误，均未考虑符号问题。双经度法适用与视场接近180°以及小于180°情况
#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
//#include<opencv2\opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/calib3d/calib3d.hpp>
//#include<opencv2/stitching/stitcher.hpp>
#include "camera.h"
#include "Correcting.h"
using namespace std;
//using namespace cv;
#define R 700//183//球面半径，hough梯度法   180    740
#define X0 740//183//圆心坐标，根据文献中算法可求   328  740
#define Y0 740//183    //   236   740
//#define CV_PI 3.1415926

Mat RotateImg(Mat &img);
Mat ImgRotate(const Mat& ucmatImg, double dDegree);

//获取有效区域
Mat GetAvailableArea(Mat& img,int X,int Y)
{
	//int  X = 1311, Y = 972;//圆心
	//int X = 1299, Y = 968;//1
	//int X = 1359, Y = 963;//2
	//int X = 1311, Y = 972;//3
	Mat result;
	result.create(2 * R, 2 * R, CV_8UC3);
	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.ptr<uchar>(i)[j * 3] = img.ptr<uchar>(i + Y - R)[(j + X - R) * 3];
			result.ptr<uchar>(i)[j * 3 + 1] = img.ptr<uchar>(i + Y - R)[((j + X - R) * 3) + 1];
			result.ptr<uchar>(i)[j * 3 + 2] = img.ptr<uchar>(i + Y - R)[((j + X - R) * 3) + 2];
		}
	}
	//imwrite("img/1.jpg", result);
	/*cvNamedWindow("tmp", CV_WINDOW_NORMAL);
	Mat tmp = img.clone();
	circle(tmp, Point(X, Y), R, Scalar(0, 0, 255), 1);
	imshow("tmp",tmp);
	waitKey(0);*/
	return result;
}

Mat xiaojiejie(Mat& img)
{
	Mat result;
	result.create(Size(2 * R, 2 * R), CV_8UC3);

	int u, v, xx, yy;
	double r, fi, theta, fai, sita, rr, x, y, z, f;

	for (int i = 0; i < img.rows; i++)//行，y
	{
		for (int j = 0; j < img.cols; j++)//列，x
		{
			u = j - R;//论文中不对
			v = R - i;//
			r = sqrt(u*u + v*v);
			if (!r) fi = 0;
			else if (u >= 0) fi = asin(v / r);
			else fi = CV_PI - asin(v / r);
			f = R * 2 / CV_PI;/////f=r/theta 焦距
			theta = r / f;/////入射角
			x = f*sin(theta)*cos(fi);
			y = f*sin(theta)*sin(fi);
			z = f*cos(theta);
			rr = sqrt(x*x + z*z);
			sita = CV_PI / 2 - atan(y / rr);
			if (z >= 0) fai = acos(x / rr);
			else fai = CV_PI - acos(x / rr);
			xx = round(f*sita);
			yy = round(f*fai);
			//xx = fai / i;
			//yy = sita / j;
			//xx = round(f*sita*cos(fai) + X0);
			//yy = round(f*sita*sin(fai) + Y0);
			if (xx<0 || yy<0 || xx>=result.cols || yy>=result.rows) continue;
			yy = img.rows - yy;

			// 无插值
			result.ptr<uchar>(xx)[yy * 3] = img.ptr(i)[j * 3];
			result.ptr<uchar>(xx)[yy * 3 + 1] = img.ptr(i)[j * 3 + 1];
			result.ptr<uchar>(xx)[yy * 3 + 2] = img.ptr(i)[j * 3 + 2];
			//result.ptr<uchar>(i)[j * 3] = img.ptr(ix)[iy * 3] * (1 - abs(x - iy)) + img.ptr(ix)[(iy + 1) * 3] * (1 - abs(y - (iy - 1)));
			//result.ptr<uchar>(i)[j * 3 + 1] = img.ptr(ix)[iy * 3 + 1] * (1 - abs(x - iy)) + img.ptr(ix)[(iy + 1) * 3 + 1] * (1 - abs(y - (iy - 1)));
			//result.ptr<uchar>(i)[j* 3 + 2] = img.ptr(ix)[iy * 3 + 2] * (1 - abs(x - iy)) + img.ptr(ix)[(iy + 1) * 3 + 2] * (1 - abs(y - (iy - 1)));


		}
	}
	return result;
}

Mat SourceToTarget(Mat& img)
{
	Mat result;
	result.create(Size(2 * R, 2 * R), CV_8UC3);

	for (int i = 0; i < img.rows; i++)//行，y
	{
		for (int j = 0; j < img.cols; j++)//列，x
		{
			double f = R * 2 / CV_PI;///???
								  //double f=R;
			int u = X0 - j;
			int v = i - Y0;
			double sita = u / f;
			double fai = v / f;
			double a = (i - X0) / f;
			double b = CV_PI / 2 - (Y0 - j) / f;
			double cos_fi = sin(a)*sin(b) / sqrt(sin(a)*sin(a)*sin(b)*sin(b) + cos(a)*cos(a));
			double sin_fi = cos(a) / sqrt(sin(a)*sin(a)*sin(b)*sin(b) + cos(a)*cos(a));
			double x = (f*a*cos_fi) + X0;
			double y = (f*a*sin_fi) + Y0;
			int ix = (int)x;
			int iy = (int)y;
			if (ix<0 || ix>=img.cols) continue;
			if (iy<0 || iy>=img.rows) continue;

			// 无插值
			result.ptr<uchar>(iy)[ix * 3] = img.ptr(i)[j * 3];
			result.ptr<uchar>(iy)[ix * 3 + 1] = img.ptr(i)[j * 3 + 1];
			result.ptr<uchar>(iy)[ix * 3 + 2] = img.ptr(i)[j * 3 + 2];
			//result.ptr<uchar>(i)[j * 3] = img.ptr(ix)[iy * 3] * (1 - abs(x - iy)) + img.ptr(ix)[(iy + 1) * 3] * (1 - abs(y - (iy - 1)));
			//result.ptr<uchar>(i)[j * 3 + 1] = img.ptr(ix)[iy * 3 + 1] * (1 - abs(x - iy)) + img.ptr(ix)[(iy + 1) * 3 + 1] * (1 - abs(y - (iy - 1)));
			//result.ptr<uchar>(i)[j* 3 + 2] = img.ptr(ix)[iy * 3 + 2] * (1 - abs(x - iy)) + img.ptr(ix)[(iy + 1) * 3 + 2] * (1 - abs(y - (iy - 1)));


		}
	}
	return result;
}

//存在问题：目标图像第一列没有找到对应源图像的点，导致图片左侧有一条黑线，目标图上(col==0)即(x==0)对应到了源图的(720,1440)
Mat CorrectImg_TargetToSource(Mat& img)//目标图像到源图像的经纬映射
{
	Mat result;
	result.create(Size(2 * R, 2 * R), CV_8UC3);
	
	//debug
	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.ptr<uchar>(i)[j * 3] = 0;
			result.ptr<uchar>(i)[j * 3 + 1] = 0;
			result.ptr<uchar>(i)[j * 3 + 2] = 255;
		}
	}

	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			double f = R * 2 / CV_PI;
			//double f = R * 2 / (CV_PI*(185.0/180.0));///???
			double a = (j+1) / f;//+1
			double b = CV_PI / 2 - (i+1 ) / f;//+1
			double cos_fi = sin(a)*sin(b) / sqrt(sin(a)*sin(a)*sin(b)*sin(b) + cos(a)*cos(a));
			double sin_fi = cos(a) / sqrt(sin(a)*sin(a)*sin(b)*sin(b) + cos(a)*cos(a));
			double sita = atan(sqrt(sin(a)*sin(a)*sin(b)*sin(b)+cos(a)*cos(a)) / sin(a) / cos(b));
			double x = f*sita*cos_fi + R;//R==X0
			double y = f*sita*sin_fi + R;//R==Y0
			int ix = (int)x;//round();
			int iy = (int)y;
			//iy = 2*R - iy -1;

			if (ix<0 || ix>=img.cols) continue;
			if (iy<0 || iy>=img.rows) continue;
			// 无插值,多对一应无需插值????
			result.ptr<uchar>(i)[j * 3] = img.ptr(iy)[ix * 3];
			result.ptr<uchar>(i)[j * 3 + 1] = img.ptr(iy)[ix * 3 + 1];
			result.ptr<uchar>(i)[j * 3 + 2] = img.ptr(iy)[ix * 3 + 2];
			/*result.ptr<uchar>(i)[j * 3] = img.ptr(i)[iy * 3] * (1 - abs(x - iy)) 
				+ img.ptr(ix)[(iy + 1) * 3] * (1 - abs(y - (iy - 1)));
			result.ptr<uchar>(i)[j * 3 + 1] = img.ptr(i)[iy * 3 + 1] * (1 - abs(x - iy)) 
				+ img.ptr(ix)[(iy + 1) * 3 + 1] * (1 - abs(y - (iy - 1)));
			result.ptr<uchar>(i)[j* 3 + 2] = img.ptr(i)[iy * 3 + 2] * (1 - abs(x - iy)) 
				+ img.ptr(ix)[(iy + 1) * 3 + 2] * (1 - abs(y - (iy - 1)));*/
			/*result.ptr<uchar>(i)[j * 3] = result.ptr<uchar>(i)[j * 3] * (1 - abs(x - ix)) + result.ptr<uchar>(i)[j * 3] * (1 - abs(x - (ix + 1))) * (1 - abs(y - iy)) +
				result.ptr<uchar>(i)[j * 3] * (1 - abs(x - ix)) + result.ptr<uchar>(i)[j * 3] * (1 - abs(x - (ix + 1)) * abs(1 - abs(y - iy));
			result.ptr<uchar>(i)[j * 3 + 1] = img.ptr(i)[iy * 3 + 1] * (1 - abs(x - iy))
				+ img.ptr(ix)[(iy + 1) * 3 + 1] * (1 - abs(y - (iy - 1)));
			result.ptr<uchar>(i)[j * 3 + 2] = img.ptr(i)[iy * 3 + 2] * (1 - abs(x - iy))
				+ img.ptr(ix)[(iy + 1) * 3 + 2] * (1 - abs(y - (iy - 1)));*/
			

		}
	}
	//result = RotateImg(result);
	//result = ImgRotate(result,270);
	Mat t, f;
	transpose(result, t);//转置，x和y交换
	flip(t, f, -1);//翻转
	//imwrite("img/2.jpg", f);
	return f;
}

//存在问题：图片旋转后左侧出现一条黑线,
Mat RotateImg(Mat &img)
{
	Point2f center(R , R );
	double angle = 270;//旋转角度
	double scale = 1;//缩放尺度
	Mat rotateMat;
	rotateMat = getRotationMatrix2D(center, angle, scale);//计算二维旋转变换矩阵
	Mat rotateImg;
	rotateImg.create(2 * R, 2 * R, CV_8UC3);
	
	//debug
	for (int i = 0; i < rotateImg.rows; i++)
	{
		for (int j = 0; j < rotateImg.cols; j++)
		{
			rotateImg.ptr<uchar>(i)[j * 3] = 0;
			rotateImg.ptr<uchar>(i)[j * 3 + 1] = 0;
			rotateImg.ptr<uchar>(i)[j * 3 + 2] = 255;
		}
	}

	warpAffine(img, rotateImg, rotateMat, img.size());//仿射变换
	return rotateImg;
	
}

//旋转图片
Mat ImgRotate(const Mat& ucmatImg, double dDegree)
{
	Mat ucImgRotate;

	double a = sin(dDegree  * CV_PI / 180);
	double b = cos(dDegree  * CV_PI / 180);
	int width = ucmatImg.cols;
	int height = ucmatImg.rows;
	int width_rotate = int(height * fabs(a) + width * fabs(b));
	int height_rotate = int(width * fabs(a) + height * fabs(b));

	Point center = Point(ucmatImg.cols / 2, ucmatImg.rows / 2);

	Mat map_matrix = getRotationMatrix2D(center, dDegree, 1.0);
	//map_matrix.at<double>(0, 2) += (width_rotate - width) / 2;     // 修改坐标偏移
	//map_matrix.at<double>(1, 2) += (height_rotate - height) / 2;   // 修改坐标偏移

	warpAffine(ucmatImg, ucImgRotate, map_matrix, { width_rotate, height_rotate },
		CV_INTER_CUBIC | CV_WARP_FILL_OUTLIERS, BORDER_CONSTANT, cvScalarAll(0));

	return ucImgRotate;
}

Mat Correcting(Mat& img,int X,int Y)
{
	Mat res = img;
	res = GetAvailableArea(res,X,Y);
	res = CorrectImg_TargetToSource(res);

	return res;
}

Mat	MergeImg(Mat& a, Mat& b)
{
	Mat result;
	result.create(2 * R, 4 * R, CV_8UC3);

	/*//debug
	for (int i = 0; i < result.rows; i++)
	{
		for (int j = 0; j < result.cols; j++)
		{
			result.ptr<uchar>(i)[j * 3] = 0;
			result.ptr<uchar>(i)[j * 3 + 1] = 0;
			result.ptr<uchar>(i)[j * 3 + 2] = 255;
		}
	}*/

	for (int i = 0; i < b.rows; i++)
	{
		for (int j = 0; j < b.cols; j++)
		{
			result.ptr<uchar>(i)[(2 * R + j) * 3] = b.ptr<uchar>(i)[j * 3];
			result.ptr<uchar>(i)[(2 * R + j) * 3 + 1] = b.ptr<uchar>(i)[(j * 3) + 1];
			result.ptr<uchar>(i)[(2 * R + j) * 3 + 2] = b.ptr<uchar>(i)[(j * 3) + 2];
		}
	}
	for (int i = 0; i < a.rows; i++)
	{
		for (int j = 0; j < a.cols; j++)
		{
			result.ptr<uchar>(i)[j * 3] = a.ptr<uchar>(i)[j * 3];
			result.ptr<uchar>(i)[j * 3 + 1] = a.ptr<uchar>(i)[(j * 3) + 1];
			result.ptr<uchar>(i)[j * 3 + 2] = a.ptr<uchar>(i)[(j * 3) + 2];
		}
	}
	
	return result;
}

//void get_test_stitch_img()//get 4 CV_PIcture
//{
//	Mat img1, img2;
//	CaptureOnce(img1, img2);
//	img1 = GetAvailableArea(img1);
//	img2 = GetAvailableArea(img2);
//	img1 = CorrectImg_TargetToSource(img1);
//	img2 = CorrectImg_TargetToSource(img2);
//	imwrite("stitch/1.jpg", img1);
//	imwrite("stitch/2.jpg", img2);
//	CaptureOnce(img1, img2);
//	img1 = GetAvailableArea(img1);
//	img2 = GetAvailableArea(img2);
//	img1 = CorrectImg_TargetToSource(img1);
//	img2 = CorrectImg_TargetToSource(img2);
//	imwrite("stitch/3.jpg", img1);
//	imwrite("stitch/4.jpg", img2);
//	waitKey(0);
//}

//void test_stitch()
//{
//	Mat img1, img2;
//	CaptureOnce(img1, img2);
//	img1 = GetAvailableArea(img1);
//	img2 = GetAvailableArea(img2);
//	img1 = CorrectImg_TargetToSource(img1);
//	img2 = CorrectImg_TargetToSource(img2);
//	
//	img1 = MergeImg(img1, img2);
//	
//	imwrite("11-11-1.jpg", img1);
//}

//int main(int argc, char** argv)
//{
//	//namedWindow("原图", CV_WINDOW_NORMAL);
//	//namedWindow("矫正", CV_WINDOW_NORMAL);
//	//char str[2][20] = { "11-7-1.jpg","11-7-2.jpg" };
//	//char tmp[20];
//	//Mat a[2];
//	//for (int i = 0; i < 2; i++)
//	//{
//	//	Mat img = imread(str[i]);
//	//	img = GetAvailableArea(img);
//	//	//imshow("原图", img);
//	//	Mat target = CorrectImg_TargetToSource(img);
//	//	//imshow("矫正", target);
//	//	sprintf(tmp,"out/11-15-%d.jpg",i);
//	//	imwrite(tmp,target);
//	//	a[i] = target;
//	//	//waitKey(0);
//	//}
//	/*Mat res;
//	res = MergeImg(a[0], a[1]);
//	imwrite("out/11-7-res5.jpg", res);*/
//	
//	//Mat img = imread("WIN_20161114_21_31_56_Pro.jpg");
//	//img = GetAvailableArea(img);
//	//img=CorrectImg_TargetToSource(img);
//	////img=MergeImg(img, img);
//	////imshow("原图", img);
//	//imwrite("why.jpg",img);
//	//waitKey(0);
//
//	/*Mat img;
//	img = imread("11-7-2.jpg");
//	img = GetAvailableArea(img);
//	img = CorrectImg_TargetToSource(img);
//	imwrite("box_in_scene.png", img);*/
//
//	Mat img;
//
//	img = imread("3-2/WIN_20161123_20_11_12_Pro.jpg");
//	img = GetAvailableArea(img);
//	img = CorrectImg_TargetToSource(img);
//	imwrite("3-2/1.jpg",img);
//	return 0;
//}


