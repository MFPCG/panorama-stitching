#include <iostream>  
#include <string>
#include <fstream>  
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include "camera.h"
#include "Correcting.h"

#include<vector>
#include<sstream>
#include <stdio.h>

#ifdef TRANSFER
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>
#include <sys/file.h>
#include <ctime>
#endif

using namespace std;
using namespace cv;
using namespace cv::detail;

#define CAPTURE  true
#define ESTIMATE_CAMERA_PARAMS
#define TRANSFER

#ifdef TRANSFER

#define PORT 8888               //服务器端监听端口号
#define MAX_BUFFER 1024         //数据缓冲区最大值
#define FILE_NAME_LEN 30
#define MAXDATASIZE 1000

#endif //

bool try_use_gpu = true; //false;  
vector<Mat> imgs;
string path = "./18-3-30/1/";
string result_name = path+"result.jpg";

int parseCmdArgs() {
	Mat img1 = imread("18-1-25/3/dest2.jpg");
	Mat img2 = imread("18-1-25/3/dest3.jpg");
	//Mat img3 = imread("17-3-23/3.jpg");
	//Mat img4 = imread("17-6-10/result.jpg");
	imgs.push_back(img1);
	imgs.push_back(img2);
	return 0;
}

void GetPictureByCameraAndCorrect()
{
	Mat img1, img2, img3,img3left,img3right;
	//CaptureOnce3(img1, img2, img3);
	imgs.clear();
#if CAPTURE
	CaptureOnce3(img1, img2, img3);
	cout<<"get source img, correcting...."<<endl;

	imwrite(path + "source1.jpg", img1);
	imwrite(path + "source2.jpg", img2);
	imwrite(path + "source3.jpg", img3);
#else 
	img1 = imread(path + "source1.jpg");
	img2 = imread(path + "source2.jpg");
	img3 = imread(path + "source3.jpg");
#endif	
	//
	img1 = Correcting(img1,1299,968);
	img2 = Correcting(img2,1359,963);
	img3 = Correcting(img3,1311,972);

	img3left = img3(Range::all(), Range(0,img3.cols/2));
	img3right = img3(Range::all(), Range(img3.cols / 2, img3.cols));

	imgs.push_back(img1);
	imgs.push_back(img2);
	imgs.push_back(img3left);
	imgs.push_back(img3right);
	
	imwrite(path + "dest1.jpg", img1);
	imwrite(path + "dest2.jpg",img2);
	imwrite(path + "dest3.jpg", img3);
	imwrite(path + "dest3l.jpg", img3left);
	imwrite(path + "dest3r.jpg", img3right);

}
#ifdef DEFAULT
int main(int argc, char* argv[]) {

	double start_time = getTickCount();

#ifdef CAPTURE
	int retval = parseCmdArgs();
	if (retval) return -1;
#else
	GetPictureByCameraAndCorrect();
#endif
	Mat pano;
	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	Stitcher::Status status = stitcher.stitch(imgs, pano);

	if (status != Stitcher::OK) {
		cout << "Can't stitch images, error code = " << int(status) << endl;
		return -1;
	}
	cout << "use time: " << (getTickCount() - start_time) / getTickFrequency() << " sec" << endl;
	namedWindow("1", CV_WINDOW_NORMAL);
	imwrite(path+"result.jpg", pano);
	imshow("1", pano);
	cv::waitKey(0);
	return 0;

	//Mat img1, img2, img3;
	////
	//img1 = imread(path + "source2.jpg");
	////img2=imread(path + "source2.jpg");
	////img3=imread(path + "source3.jpg");
	////
	//img1 = Correcting(img1);
	////img2 = Correcting(img2);
	////img3 = Correcting(img3);
	////
	//imwrite(path + "ndest2.jpg", img1);
	////imwrite(path + "dest2.jpg", img2);
	////imwrite(path + "dest3.jpg", img3);
}
#else 
int main(int argc, char* argv[]) 
{

	//默认参数
	vector<String> img_names;
	double scale = 1;
	string features_type = "orb";//"surf"或"orb"特征类型
	float match_conf = 0.3f;
	float conf_thresh = 0.4f;	//setPanoConfidenceThresh
	string adjuster_method = "ray";//"reproj" or "ray" 调节器方法
	bool do_wave_correct = true;
	//detail::WaveCorrectKind wave_correct_type = WAVE_CORRECT_HORIZ;
	string warp_type = "spherical";//地图投影方式，球面投影
	//int expos_comp_type = detail::ExposureCompensator::GAIN_BLOCKS;
	string seam_find_type = "voronoi";
	float blend_strength = 5;
	int blend_type = Blender::MULTI_BAND;

	double start_time = getTickCount();
	double t;

	OpenCamera();

#ifdef TRANSFER
	struct sockaddr_in server_addr, client_addr;
    int server_sockfd, client_sockfd;
    int file_fd;
    int size, write_size;
    char buffer[MAX_BUFFER];

    if ((server_sockfd = socket(AF_INET, SOCK_STREAM, 0)) == -1)    //创建Socket
    {
        perror("Socket Created Failed!\n");
        exit(1);
    }
    printf("Socket Create Success!\n");

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(PORT);
    bzero(&(server_addr.sin_zero), 8);

    int opt = 1;
    int res = setsockopt(server_sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));    //设置地址复用
    if (res < 0)
    {
        perror("Server reuse address failed!\n");
        exit(1);
    }

    if (bind(server_sockfd, (struct sockaddr *)&server_addr, sizeof(server_addr)) == -1)  //绑定本地地址
    {
        perror("Socket Bind Failed!\n");
        exit(1);
    }
    printf("Socket Bind Success!\n");

    if (listen(server_sockfd, 5) == -1)                 //监听
    {
        perror("Listened Failed!\n");
        exit(1);
    }
    printf("Listening ....\n");

    socklen_t len = sizeof(client_addr);

    printf("waiting connection...\n");
    if ((client_sockfd = accept(server_sockfd, (struct sockaddr *)&client_addr, &len)) == -1)  //等待客户端连接
    {
        perror("Accepted Failed!\n");
        exit(1);
    }
    printf("connection established!\n");
    printf("waiting message...\n");

	memset(buffer, 0, sizeof(buffer));

#endif 

	Mat resulttmp;
	
#ifdef TRANSFER
	///接收拍照请求
	while ((size = read(client_sockfd, buffer, MAX_BUFFER))>0)
#else
	while(getchar())
#endif
	{
#ifdef TRANSFER
		buffer[size] = '\0';
        printf("Recv msg from client: %s\n", buffer);	
	    if(strcmp(buffer, "Shoot") == 0)
        {
			printf("Starting Shootting!\n");
		}
		else 
		{
			memset(buffer, 0, sizeof(buffer));
			break;
		}
#endif //

		time_t nowtime;
		nowtime = time(NULL);
		char str_time[19];
		strftime(str_time, sizeof(str_time), "%Y%m%d%H%M%S.jpg",localtime(&nowtime));
		result_name = path+str_time;

		t = getTickCount();
		GetPictureByCameraAndCorrect();
		cout << "get images, times: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
		
		for (int times = 0; times < 3; times++)
		{
			vector<Mat> images;
#ifdef ESTIMATE_CAMERA_PARAMS
			string CameraFile;
			switch (times)
			{
			case 0:
				CameraFile = "./params/Camera12.yml";
				images.push_back(imgs[0]);
				images.push_back(imgs[1]);
				break;
			case 1:
				CameraFile = "./params/Camera123l.yml";
				images.push_back(resulttmp);
				images.push_back(imgs[2]);
				break;
			case 2:
				CameraFile = "./params/Camera3r123l.yml";
				images.push_back(imgs[3]);
				images.push_back(resulttmp);
				break;
			default: break;
			}
			vector<CameraParams> cameras(2);
			FileStorage fs(CameraFile, FileStorage::READ);
			fs["c12f"] >> cameras[0].focal;
			fs["c12a"] >> cameras[0].aspect;
			fs["c12x"] >> cameras[0].ppx;
			fs["c12y"] >> cameras[0].ppy;
			fs["c12R"] >> cameras[0].R;
			fs["c12t"] >> cameras[0].t;
			fs["c21f"] >> cameras[1].focal;
			fs["c21a"] >> cameras[1].aspect;
			fs["c21x"] >> cameras[1].ppx;
			fs["c21y"] >> cameras[1].ppy;
			fs["c21R"] >> cameras[1].R;
			fs["c21t"] >> cameras[1].t;
			float warped_image_scale;
			warped_image_scale = static_cast<float>(cameras[0].focal + cameras[1].focal)*0.5f;
#endif

#ifdef FIND_FEATURE
			//2-调整图像的大小和找到特征的步骤
			cout << "2-Finding feature..." << endl;

			Ptr<FeaturesFinder> finder;
			if (features_type == "surf")
			{
				finder = makePtr<SurfFeaturesFinder>();
			}
			else if (features_type == "orb")
			{
				finder = makePtr<OrbFeaturesFinder>();
			}
			else
			{
				cout << "Unknown 2D features type: " << features_type << endl;
				return -1;
			}

			Mat full_img, img;
			vector<ImageFeatures> features(num_images);
			vector<Mat> images(num_images);
			vector<Size> full_img_sizes(num_images);

			for (int i = 0; i < num_images; ++i)
			{
				full_img = imread(img_names[i]);
				full_img_sizes[i] = full_img.size();

				if (full_img.empty())
				{
					cout << "Can't open image" << img_names[i] << endl;
					return -1;
				}

				resize(full_img, img, Size(), scale, scale);
				images[i] = img.clone();

				(*finder)(img, features[i]);
				features[i].img_idx = i;
				cout << "Features in image #" << i + 1 << " are : " << features[i].keypoints.size() << endl;
			}
			finder->collectGarbage();
			full_img.release();
			img.release();
			cout << "Finding features ,time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
#endif 

#ifndef ESTIMATE_CAMERA_PARAMS

			//3-特征匹配
			cout << "3-Pairwise mathcing..." << endl;
			t = getTickCount();
			vector<MatchesInfo> pairwise_matches;
			//BestOf2NearestMatcher matcher(false, match_conf);
			Ptr<detail::FeaturesMatcher> matcher = makePtr<detail::BestOf2NearestMatcher>(true, match_conf);
			(*matcher)(features, pairwise_matches);
			matcher->collectGarbage();
			cout << "pairwise matches size: " << pairwise_matches.size() << endl;
			cout << "Pairwise mathcing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

			for (int i = 0; i < pairwise_matches.size(); i++)
			{
				//cout << "H #" << i << pairwise_matches[i].H << endl;
			}

			//4-选取图像并匹配子集，以建立全景图像
			vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
			//leaveBiggestComponent的主要目的可以描述为“寻找所有配对中肯定属于一幅全景图像的图片”，主要通过的方法是“并查集”
			vector<Mat> img_subset;
			vector<String> img_names_subset;
			vector<Size> full_img_sizes_subset;
			for (size_t i = 0; i < indices.size(); ++i)
			{
				img_names_subset.push_back(img_names[indices[i]]);
				img_subset.push_back(images[indices[i]]);
				full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
				//
				//cout << "H #" << indices[i] << pairwise_matches[i].H << endl;
				//
			}
			//
			/*pairwise_matches[1].H.at<double>(1, 0) = 0.f;
			pairwise_matches[1].H.at<double>(1, 1) = 1.f;
			pairwise_matches[1].H.at<double>(1, 2) = 0.f;*/
			//Mat resultAll;
			//Mat img2t = images[1].clone();
			//warpPerspective(images[1], resultAll, pairwise_matches[1].H, Size(img2t.cols*2,img2t.rows));
			//Mat half(resultAll, Rect(0, 0, img2t.cols, img2t.rows));
			////images[0].copyTo(half);
			//imwrite("H.jpg", resultAll);
			//
			images = img_subset;
			img_names = img_names_subset;
			full_img_sizes = full_img_sizes_subset;

			if ((int)img_subset.size() < 2)
			{
				cout << "Need more images" << endl;
				return -1;
			}

			//粗略估计相机参数
			HomographyBasedEstimator estimator;
			vector<CameraParams> cameras;//存储相机参数
			if (!estimator(features, pairwise_matches, cameras))//求相机参数存入 cameras 中 //vector out of range!!!!????
			{
				cout << "Homography estimation failed." << endl;
				return -1;
			}

			for (size_t i = 0; i < cameras.size(); ++i)
			{
				Mat R;
				cameras[i].R.convertTo(R, CV_32F);
				cameras[i].R = R;
				cout << "Initial intrinsic (K)#" << indices[i] + 1 << ":\n" << cameras[i].K() << endl;
				cout << "Initial camera R#" << indices[i] + 1 << ":\n" << cameras[i].R << endl;
				cout << "Initial camera t#" << indices[i] + 1 << ":\n" << cameras[i].t << endl;
			}

			//5-细化全局相机参数
			Ptr<BundleAdjusterBase> adjuster;
			if (adjuster_method == "reproj")
			{	//'reproj'方法
				adjuster = makePtr<BundleAdjusterReproj>();
			}
			else if (adjuster_method == "ray")
			{	//’射线’方法
				adjuster = makePtr<BundleAdjusterRay>();
			}
			else
			{
				cout << "Unknown adjuster method : " << adjuster_method << endl;
				return -1;
			}

			adjuster->setConfThresh(conf_thresh);
			if (!(*adjuster)(features, pairwise_matches, cameras))
			{
				cout << "Camera parmaters adjusting failed." << endl;
				return -1;
			}

			//找到中值焦距
			vector<double> focals;
			for (size_t i = 0; i < cameras.size(); ++i)
			{
				cout << "Camera (K)#" << indices[i] + 1 << ":\n" << cameras[i].K() << endl;
				cout << "camera R#" << indices[i] + 1 << ":\n" << cameras[i].R << endl;
				cout << "camera t#" << indices[i] + 1 << ":\n" << cameras[i].t << endl;
				focals.push_back(cameras[i].focal);
			}
			//设置旋转矩阵y轴不变
			/*cameras[0].R.at<float>(1, 0) = 0.f;
			cameras[0].R.at<float>(1, 1) = 1.f;
			cameras[0].R.at<float>(1, 2) = 0.f;
			cameras[1].R.at<float>(1, 0) = 0.f;
			cameras[1].R.at<float>(1, 1) = 1.f;
			cameras[1].R.at<float>(1, 2) = 0.f;*/
			//
			sort(focals.begin(), focals.end());
			float warped_image_scale;
			if (focals.size() % 2 == 1)
			{
				warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
			}
			else
			{
				warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2])*0.5f;
			}

			//6-载波相关（可选）
			if (do_wave_correct)
			{
				vector<Mat> rmats;
				for (size_t i = 0; i < cameras.size(); ++i)
				{
					rmats.push_back(cameras[i].R.clone());
				}

				waveCorrect(rmats, wave_correct_type);
				for (size_t i = 0; i < cameras.size(); ++i)
				{
					cameras[i].R = rmats[i];
				}
			}
			cout << "after wave correct:" << endl;
			for (size_t i = 0; i < cameras.size(); ++i)
			{
				cout << "Camera (K)#" << indices[i] + 1 << ":\n" << cameras[i].K() << endl;
				cout << "camera R#" << indices[i] + 1 << ":\n" << cameras[i].R << endl;
				cout << "camera t#" << indices[i] + 1 << ":\n" << cameras[i].t << endl;
			}

			//保存相机参数
			/*FileStorage fs("Camera3r123l.yml", FileStorage::WRITE);
			fs << "c12f" << cameras[0].focal;
			fs << "c12a" << cameras[0].aspect;
			fs << "c12x" << cameras[0].ppx;
			fs << "c12y" << cameras[0].ppy;
			fs << "c12R" << cameras[0].R;
			fs << "c12t" << cameras[0].t;
			fs << "c21f" << cameras[1].focal;
			fs << "c21a" << cameras[1].aspect;
			fs << "c21x" << cameras[1].ppx;
			fs << "c21y" << cameras[1].ppy;
			fs << "c21R" << cameras[1].R;
			fs << "c21t" << cameras[1].t;
			fs.release();*/
			////
#endif

		//7-弯曲图像
			cout << "7-Warping images (auxiliary)..." << endl;
			t = getTickCount();
			int num_images = 2;
			vector<Point> corners(num_images);
			vector<Mat> masks_warped(num_images);
			vector<Mat> images_warped(num_images);
			vector<Size> sizes(num_images);
			vector<Mat> masks(num_images);

			//准备图像掩码
			for (int i = 0; i < num_images; i++)
			{
				masks[i].create(images[i].size(), CV_8U);
				masks[i].setTo(Scalar::all(255));
			}

			//地图投影
			Ptr<WarperCreator> warper_creator;
			if (warp_type == "spherical")//球体
			{
				warper_creator = new cv::SphericalWarper();
			}

			if (!warper_creator)
			{
				cout << "Can't create the following warper" << warp_type << endl;
				return -1;
			}

			Ptr<detail::RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale*scale));

			for (int i = 0; i < num_images; ++i)
			{
				Mat_<float> K;
				cameras[i].K().convertTo(K, CV_32F);
				float swa = (float)scale;
				K(0, 0) *= swa;
				K(0, 2) *= swa;
				K(1, 1) *= swa;
				K(1, 2) *= swa;

				corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);

				sizes[i] = images_warped[i].size();

				warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
			}

			vector<Mat> images_warped_f(num_images);
			for (int i = 0; i < num_images; ++i)
			{
				images_warped[i].convertTo(images_warped_f[i], CV_32F);
			}
			cout << "Warping images, times: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

#ifdef EXPOSURE_COMPENSTATOR
			//8-补偿曝光误差
			cout << "8-exposure compensator.." << endl;
			t = getTickCount();
			Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
			compensator->feed(corners, images_warped, masks_warped);
			cout << "compensator times: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
#endif 

			//9-找到掩码接缝
			cout << "9-find seam.." << endl;
			t = getTickCount();
			Ptr<SeamFinder> seam_finder;
			if (seam_find_type == "no")
			{
				seam_finder = new detail::NoSeamFinder();
			}
			else if (seam_find_type == "voronoi")
			{
				seam_finder = new detail::VoronoiSeamFinder();
			}
			else if (seam_find_type == "gc_color")
			{
				seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
			}
			else
			{
				cout << "Unknown seam finder : " << seam_find_type << endl;
				return -1;
			}

			if (!seam_finder)
			{
				cout << "Can't create the following seam finder" << seam_find_type << endl;
				return -1;
			}

			seam_finder->find(images_warped_f, corners, masks_warped);

			cout << "find seam, times: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

			//释放未使用的内存
			//images.clear();
			images_warped.clear();
			images_warped_f.clear();
			masks.clear();

			//10-创建一个混合器
			cout << "10-blend image..." << endl;
			t = getTickCount();
			//Ptr<Blender> blender = Blender::createDefault(blend_type, false);
			Ptr<Blender> blender = new detail::MultiBandBlender(try_use_gpu);
			Size dst_sz = resultRoi(corners, sizes).size();
			// float blend_width = sqrt(static_cast<float>(dst_sz.area()))*blend_strength / 100.f;
			// if (blend_strength < 1.f)
			// {
			// 	blender = Blender::createDefault(Blender::NO, false);
			// }
			// else if (blend_type == Blender::MULTI_BAND)
			// {
			// 	MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
			// 	mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
			// 	cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
			// }
			// else if (blend_type == Blender::FEATHER)
			// {
			// 	FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
			// 	fb->setSharpness(1.f / blend_width);
			// 	cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
			// }

			blender->prepare(corners, sizes);

			cout << "blend prepare, time: " << (getTickCount() - t) / getTickFrequency() << " sec" << endl;

			//11-（图像）合成步骤
		/*	cvNamedWindow("warpimg", CV_WINDOW_NORMAL);
			cvNamedWindow("warpmask", CV_WINDOW_NORMAL);*/
			cout << "Compositing..." << endl;
			t = getTickCount();
			Mat img_warped, img_warped_s;
			Mat dilated_mask, seam_mask, mask, mask_warped;
			Mat full_img, img;

			for (int img_idx = 0; img_idx < num_images; ++img_idx)
			{
				//cout << "Compositing image #" << indices[img_idx] + 1 << endl;
				cout << "Compositing image #" << img_idx + 1 << endl;

				//11.1-读入图像，必要时调整图像大小
				//full_img = imread(img_names[img_idx]);
				full_img = images[img_idx];

				if (abs(scale - 1) > 1e-1)
				{
					resize(full_img, img, Size(), scale, scale);
				}
				else
				{
					img = full_img;
				}

				full_img.release();
				Size img_size = img.size();

				Mat K;
				cameras[img_idx].K().convertTo(K, CV_32F);

				//11.2.1-弯曲当前图像//双线性插值
				warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
				//imshow("warpimg", img_warped);


				//11.2.2-弯曲当前掩码//最近邻插值
				mask.create(img_size, CV_8U);
				mask.setTo(Scalar::all(255));
				warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
				//imshow("warpmask", mask_warped);
				//waitKey(0);

				//11.3-补偿曝光误差
				//compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
				img_warped.convertTo(img_warped_s, CV_16S);
				img_warped.release();
				img.release();
				mask.release();

				dilate(masks_warped[img_idx], dilated_mask, Mat());
				resize(dilated_mask, seam_mask, mask_warped.size());
				mask_warped = seam_mask&mask_warped;

				//11.4-合成图像步骤
				blender->feed(img_warped_s, mask_warped, corners[img_idx]);
			}
			Mat  result_mask;
			blender->blend(resulttmp, result_mask);

			cout << "Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
		}
		resize(resulttmp, resulttmp, Size(resulttmp.rows * 2, resulttmp.rows));
		imwrite(result_name, resulttmp);

#ifdef TRANSFER
		///发送图片
		int readbytes = 0, sendbytes = 0, totalbytes = 0;
		file_fd = open(result_name.c_str(), O_RDWR, 0666);
		if(file_fd < 0)
		{
			perror("open file failed!");
			exit(1);
		}

		memset(buffer, 0, sizeof(buffer));
		while((readbytes = read(file_fd, buffer,MAXDATASIZE)) )
		{
			if((sendbytes=send(client_sockfd, buffer, readbytes,0))==-1){
				perror("send");
				exit(1);
			}
			totalbytes += sendbytes;
		}
		close(file_fd);
		printf("sent %d bytes ..\n", totalbytes);
			
    		//}	
#endif // TRANSFER

		cout << "Finished, total time: " << ((getTickCount() - start_time) / getTickFrequency()) << " sec" << endl;
	}

#ifdef TRANSFER
	close(client_sockfd);   //关闭Socket
    close(server_sockfd);
#endif // TRANSFER

	return 0;
}
#endif

