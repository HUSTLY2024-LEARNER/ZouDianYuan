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
	cout << "Ԥ����Ϊ��" << class_name[maxclass.x] << endl;
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
	split(img, channels);//����BGR
	static int num = 0;
	dilate(channels[2], img3, getStructuringElement(MORPH_RECT, Size(7, 7)));
	erode(img3, img3, getStructuringElement(MORPH_RECT, Size(20, 20)));
	threshold(img3, img3, 170, 255, THRESH_BINARY);
	Mat result;
	bitwise_or(img, img, result, img3);
	split(result, channels);//����BGR
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

void init()//Windows�µ�������ã�Linux���������޸ģ����أ�
{
	string path = "D:\\Desktop\\py-program\\frozen_models\\frozen_graph.pb";
	net = readNetFromTensorflow(path);
	printf("ģ�ͼ��سɹ�\n");

	cout << cv::getBuildInformation() << endl;
	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640); //ͼ��Ŀ���Ҫ���֧�ִ˿�
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480); //ͼ��ĸߣ���Ҫ���֧�ִ˸�
	int deviceID = 1; //����豸��
	cap.open(deviceID); //�����
	if (!cap.isOpened()) //�ж�����Ƿ��
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
	//cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25);//�ر��Զ��ع�
	//cap.set(CAP_PROP_APERTURE, 10);//��Ȧ
	//cap.set(CAP_PROP_TEMPERATURE,1);//��ƽ��
	cap.set(CAP_PROP_SETTINGS, 1);//����
	cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 2.6);
	cap.set(cv::CAP_PROP_EXPOSURE, 1);
	printf("AUTO_EXPOSURE:%f\n", cap.get(CAP_PROP_AUTO_EXPOSURE));
	printf("Exposure:%f\n", cap.get(CAP_PROP_EXPOSURE));
	cap.set(CAP_PROP_BRIGHTNESS, -255);//���� -255
	cap.set(CAP_PROP_CONTRAST, 10);//�Աȶ� 16
	cap.set(CAP_PROP_SATURATION, 36);//���Ͷ� 36
	cap.set(CAP_PROP_HUE, 0); //ɫ�� 0
	cap.set(CAP_PROP_GAMMA, 250);//٤�� 100
}
int main(int sleeptime)
{
	init();
	cv::Mat img;
	while (true)
	{
		cap >> img; //������ʽ����ͼ��
		catchNum(img);
		cv::imshow("raw_pic", img);
		cv::waitKey(1);
	}
	cap.release(); //�ͷ�����������
}
