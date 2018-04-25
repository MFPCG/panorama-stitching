//ע�⣺ԭ���й�ʽ14��15��17��18��22��23���󣬾�δ���Ƿ������⡣˫���ȷ��������ӳ��ӽ�180���Լ�С��180�����
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
#define R 700//183//����뾶��hough�ݶȷ�   180    740
#define X0 740//183//Բ�����꣬�����������㷨����   328  740
#define Y0 740//183    //   236   740
//#define CV_PI 3.1415926

Mat RotateImg(Mat &img);
Mat ImgRotate(const Mat& ucmatImg, double dDegree);

//��ȡ��Ч����
Mat GetAvailableArea(Mat& img,int X,int Y)
{
	//int  X = 1311, Y = 972;//Բ��
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

	for (int i = 0; i < img.rows; i++)//�У�y
	{
		for (int j = 0; j < img.cols; j++)//�У�x
		{
			u = j - R;//�����в���
			v = R - i;//
			r = sqrt(u*u + v*v);
			if (!r) fi = 0;
			else if (u >= 0) fi = asin(v / r);
			else fi = CV_PI - asin(v / r);
			f = R * 2 / CV_PI;/////f=r/theta ����
			theta = r / f;/////�����
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

			// �޲�ֵ
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

	for (int i = 0; i < img.rows; i++)//�У�y
	{
		for (int j = 0; j < img.cols; j++)//�У�x
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

			// �޲�ֵ
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

//�������⣺Ŀ��ͼ���һ��û���ҵ���ӦԴͼ��ĵ㣬����ͼƬ�����һ�����ߣ�Ŀ��ͼ��(col==0)��(x==0)��Ӧ����Դͼ��(720,1440)
Mat CorrectImg_TargetToSource(Mat& img)//Ŀ��ͼ��Դͼ��ľ�γӳ��
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
			// �޲�ֵ,���һӦ�����ֵ????
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
	transpose(result, t);//ת�ã�x��y����
	flip(t, f, -1);//��ת
	//imwrite("img/2.jpg", f);
	return f;
}

//�������⣺ͼƬ��ת��������һ������,
Mat RotateImg(Mat &img)
{
	Point2f center(R , R );
	double angle = 270;//��ת�Ƕ�
	double scale = 1;//���ų߶�
	Mat rotateMat;
	rotateMat = getRotationMatrix2D(center, angle, scale);//�����ά��ת�任����
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

	warpAffine(img, rotateImg, rotateMat, img.size());//����任
	return rotateImg;
	
}

//��תͼƬ
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
	//map_matrix.at<double>(0, 2) += (width_rotate - width) / 2;     // �޸�����ƫ��
	//map_matrix.at<double>(1, 2) += (height_rotate - height) / 2;   // �޸�����ƫ��

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
//	//namedWindow("ԭͼ", CV_WINDOW_NORMAL);
//	//namedWindow("����", CV_WINDOW_NORMAL);
//	//char str[2][20] = { "11-7-1.jpg","11-7-2.jpg" };
//	//char tmp[20];
//	//Mat a[2];
//	//for (int i = 0; i < 2; i++)
//	//{
//	//	Mat img = imread(str[i]);
//	//	img = GetAvailableArea(img);
//	//	//imshow("ԭͼ", img);
//	//	Mat target = CorrectImg_TargetToSource(img);
//	//	//imshow("����", target);
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
//	////imshow("ԭͼ", img);
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


