#include <iostream>
#include <Windows.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace cv::dnn;

static string class_name[] = { "0", "1", "2",
						  "3", "4","5", "6", "7", "8", "9" };
Net net;
cv::VideoCapture cap;
int predict(Mat mat)
{
	Mat frame_32F;
	mat.convertTo(frame_32F, CV_32FC1);
	Mat blob = blobFromImage(mat / 255.0,
		1.0,
		Size(32, 32),
		Scalar(0, 0, 0));
	net.setInput(blob);
	Mat out = net.forward();
	Point maxclass;
	minMaxLoc(out, NULL, NULL, NULL, &maxclass);
	cout << "预测结果为：" << class_name[maxclass.x] << endl;
	cout << out;
	return atoi(class_name[maxclass.x].c_str());
}
void resizeFill(Mat& src, Mat& dst, Size size)
{
	Mat temp;
	double scale = min((double)size.width / src.cols, (double)size.height / src.rows);
	resize(src, temp, Size(), scale, scale);
	resize(dst, dst, size);
	copyMakeBorder(temp, dst, 0, size.height - temp.rows, 0, size.width - temp.cols, BORDER_CONSTANT, Scalar(0, 0, 0));
}
int catchNum(Mat& OriginImg)
{
	Mat img = OriginImg.clone();
	Mat testimg, testimg2;
	Mat img3;
	Mat imgWrite = img.clone();
	vector<Mat> channels;
	split(img, channels);//分离BGR
	static int num = 0;
	dilate(channels[2], img3, getStructuringElement(MORPH_RECT, Size(7, 7)));
	erode(img3, img3, getStructuringElement(MORPH_RECT, Size(20, 20)));
	threshold(img3, img3, 170, 255, THRESH_BINARY);
	Mat result;
	bitwise_or(img, img, result, img3);
	split(result, channels);//分离BGR
	erode(channels[0], channels[0], getStructuringElement(MORPH_RECT, Size(3, 3)));
	dilate(channels[0], channels[0], getStructuringElement(MORPH_RECT, Size(12, 12)));
	inRange(channels[0], 150, 255, channels[0]);
	dilate(channels[0], channels[0], getStructuringElement(MORPH_RECT, Size(3, 3)));
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(channels[0], contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	RotatedRect contoursRrect;
	Rect contoursNormalRect;
	int maxlength = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		contoursRrect = minAreaRect(contours[i]);
		if (max(contoursRrect.size.height, contoursRrect.size.width) > maxlength)
		{
			maxlength = max(contoursRrect.size.height, contoursRrect.size.width);
		}
	}
	for (int i = 0; i < contours.size(); i++)
	{
		contoursRrect = minAreaRect(contours[i]);
		contoursNormalRect = boundingRect(contours[i]);
		if ((max(contoursRrect.size.height, contoursRrect.size.width) > maxlength * 0.5) &&
			(contoursRrect.center.x > 100) && (contoursRrect.center.x < 540)
			&& (contoursRrect.center.y < 400) && (contoursRrect.center.y > 80))//isValid
		{
			Point2f vertices[4];
			for (int i = 0; i < 4; i++)
				rectangle(imgWrite, contoursNormalRect, Scalar(0, 255, 0), 2);
			if (contoursRrect.boundingRect().x >= 0 &&
				contoursRrect.boundingRect().y >= 0 &&
				contoursRrect.boundingRect().x + contoursRrect.boundingRect().width <= img.cols &&
				contoursRrect.boundingRect().y + contoursRrect.boundingRect().height <= img.rows)
			{
				Mat a = (channels[0])(contoursNormalRect);
				if (!a.empty())
				{
					resizeFill(a, a, Size(64, 128));
					cv::imshow("cut_pic", a);
					//the below code is used for dataset to train the model
					//char path[30];
					//sprintf_s(path, sizeof(path), "D://test2//%d.jpg", num++);
					//imwrite(path, a);
					int predictNum = predict(a);
					putText(imgWrite, to_string(predictNum), contoursRrect.center, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
				}
			}
		}
	}

	cv::imshow("rected_pic", imgWrite);
	cv::imshow("2value_pic", channels[0]);
	return 0;
}

void init()//Windows下的相机配置，Linux下请自行修改（慎重）
{
	string path = "D:\\Desktop\\py-program\\frozen_models\\frozen_graph.pb";
	net = readNetFromTensorflow(path);
	printf("模型加载成功\n");

	cout << cv::getBuildInformation() << endl;
	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640); //图像的宽，需要相机支持此宽
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480); //图像的高，需要相机支持此高
	int deviceID = 1; //相机设备号
	cap.open(deviceID); //打开相机
	if (!cap.isOpened()) //判断相机是否打开
	{
		std::cerr << "ERROR!!Unable to open camera\n";
		exit(-1);
	}
	printf("Brightness:%f\n", cap.get(CAP_PROP_BRIGHTNESS));
	printf("Contrast:%f\n", cap.get(CAP_PROP_CONTRAST));
	printf("Saturation:%f\n", cap.get(CAP_PROP_SATURATION));
	printf("Hue:%f\n", cap.get(CAP_PROP_HUE));
	printf("Gamma:%f\n", cap.get(CAP_PROP_GAMMA));

	//Exposure and Gain is not supported
	//cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25);//关闭自动曝光
	//cap.set(CAP_PROP_APERTURE, 10);//光圈
	//cap.set(CAP_PROP_TEMPERATURE,1);//白平衡
	cap.set(CAP_PROP_SETTINGS, 1);//设置
	cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 2.6);
	cap.set(cv::CAP_PROP_EXPOSURE, 1);
	printf("AUTO_EXPOSURE:%f\n", cap.get(CAP_PROP_AUTO_EXPOSURE));
	printf("Exposure:%f\n", cap.get(CAP_PROP_EXPOSURE));
	cap.set(CAP_PROP_BRIGHTNESS, -255);//亮度 -255
	cap.set(CAP_PROP_CONTRAST, 10);//对比度 16
	cap.set(CAP_PROP_SATURATION, 36);//饱和度 36
	cap.set(CAP_PROP_HUE, 0); //色调 0
	cap.set(CAP_PROP_GAMMA, 250);//伽马 100
}
int main(int sleeptime)
{
	init();
	cv::Mat img;
	while (true)
	{
		cap >> img; //以流形式捕获图像
		catchNum(img);
		cv::imshow("raw_pic", img);
		cv::waitKey(1);
	}
	cap.release(); //释放相机捕获对象
}
