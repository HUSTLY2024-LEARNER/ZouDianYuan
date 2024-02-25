#include "Horizon.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#define PI 3.1415926
using namespace std;
using namespace cv;

SerialPort serialPort;
TargetPlace currentPlace = ORIGINALPLACE;
AbsoluteDimension currentDimension = N;

cv::VideoCapture cap;
void findWall(Mat& OriginImg, double& theta)
{
	Mat img = OriginImg.clone();
	Mat img2;
	GaussianBlur(img, img, Size(7, 7), 0, 0);
	inRange(img, Scalar(0, 0, 0), Scalar(10, 10, 10), img2);
	dilate(img2, img2, getStructuringElement(MORPH_RECT, Size(20, 20)));
	erode(img2, img2, getStructuringElement(MORPH_RECT, Size(32, 32)));
	Canny(img2, img2, 0, 40);
	vector<Vec2f> lines;
	HoughLines(img2, lines, 2, PI / 180, 130);
	vector<float> thetas(lines.size());
	for (int i = 0; i < lines.size(); i++)
	{
		thetas[i] = lines[i][1];
	}
	sort(thetas.begin(), thetas.end());
	if (!thetas.empty())
	{
		float maintheta = thetas[(thetas.size() - 1) / 2];
		double truetheta = atan(0.62487 * tan(maintheta)) * 180.0 / PI;
		if (truetheta > 0)
			truetheta = 90 - truetheta;
		else
			truetheta = -90 - truetheta;
		//putText(img, to_string(truetheta), Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
		theta = truetheta;
		cout << "truetheta:" << truetheta << endl;
	}
}
void backToDimension(bool strictExam = 0)
{
	cap.set(CAP_PROP_BRIGHTNESS, 0.390625);
	double truetheta = -100.0;
	Mat img, img_discard;
	int count = 0;
	while (1)
	{
		for (int i = 0; i < 4; i++)
			cap >> img_discard;
		cap >> img;
		findWall(img, truetheta);

		if ((truetheta > -3) && (truetheta < 3))
			break;
		else if (truetheta < -95)
		{
			cout << "Error with findwall" << endl;
			return;
		}
		else if (truetheta < -60)
			rotateAbsolute(90 + truetheta);
		else if (truetheta > 60)
			rotateAbsolute(truetheta-90);
		else
		{
			rotateAbsolute((int)truetheta);
		}
		truetheta = -100;
		count++;
		if (count > 30)
			break;
	}
	return;
}

Point2f getFitCircle(Point2f pt1, Point2f pt2, Point2f pt3, double* _radius)
{
	Point2f point;
	double x1 = pt1.x, x2 = pt2.x, x3 = pt3.x;
	double y1 = pt1.y, y2 = pt2.y, y3 = pt3.y;
	double a = x1 - x2;
	double b = y1 - y2;
	double c = x1 - x3;
	double d = y1 - y3;
	double e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0;
	double f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0;
	double det = b * c - a * d;
	if (fabs(det) < 1e-5)
	{
		*_radius = -1;
		return point;
	}

	double x0 = -(d * e - b * f) / det;
	double y0 = -(a * f - c * e) / det;
	*_radius = hypot(x1 - x0, y1 - y0);
	point.x = x0;
	point.y = y0;
	return point;
}
int catchBlock(Mat& OriginImg, int mode, double* actnum, int strictMode = 0)//0:blue 1:purple //0for l0 1for l1 2for z0 3for z1 4for pass//-1 for error//5 for left 6 for right//strictMode
{
	Mat img = OriginImg.clone();
	GaussianBlur(img, img, Size(7, 7), 0, 0);
	vector<Mat> channels;
	split(img, channels);//分离BGR
	inRange(channels[0], 35, 255, channels[0]);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(channels[0], contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	int max_index = 0;
	if (contours.size())
	{
		for (int i = 1; i < contours.size(); i++)
			if (contourArea(contours[i]) > contourArea(contours[max_index]))
				max_index = i;
	}
	else
	{
		max_index = -1;
		return -1;
	}
	if (max_index >= 0)
	{
		Mat mask = Mat::zeros(img.size(), CV_8UC1);
		drawContours(mask, contours, max_index, Scalar(255, 255, 255), -1);
		bitwise_and(channels[0], mask, channels[0]);
		//imshow("mask", mask);
	}
	//imshow("channels", channels[0]);
	Moments m = moments(channels[0], true);
	Point p(m.m10 / m.m00, m.m01 / m.m00);
	//circle(img, p, 5, Scalar(0, 0, 255), 3);
	//putText(img, to_string(p.x), Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	//putText(img, to_string(p.y), Point(20, 60), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	//putText(img, to_string(p.x - 320), Point(20, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	double radius = 0;
	Point2f center;
	double recommend_angle = 0;
	double pass_length = 0;
	if (mode == 0)
	{
		center = getFitCircle(Point2f(81, 329), Point2f(286, 291), Point2f(487, 307), &radius);//BLUE

	}
	else
	{
		center = getFitCircle(Point2f(142, 260), Point2f(263, 237), Point2f(479, 237), &radius);//PURPLE
	}
	//recommend_angle = atan((29.8176 - p.x + 320) / 532.51) * 180.0 / PI;
	recommend_angle = asin((52.5 - p.x + 320) / 597.95) * 180.0 / PI + 0.5;
	pass_length = sqrt((center.x - p.x) * (center.x - p.x) + (center.y - p.y) * (center.y - p.y)) - radius;
	if ((pass_length > -30) && (pass_length < 30) && (strictMode != 0))
	{
		*actnum = recommend_angle;
		return 4;
	}
	if ((pass_length > -20) && (pass_length < 10))
	{
		*actnum = recommend_angle;
		return 4;
	}
	else
	{
		if(p.x>480)
			return 6;
		else if(p.x<160)
			return 5;

		if (pass_length > 60)return 0;
		else if (pass_length > 10)return 2;
		else if (pass_length < -60)return 3;
		else if (pass_length < -20)return 1;
	}
}
void catchTry(int mode, int setMode = 0,int strictMode=0)//0 for blue 1 for purple setMode!=0 when just need catch the brick
{
	cap.set(CAP_PROP_BRIGHTNESS, 0);
	double actnum = -100;
	Mat img, img_discard;
	bool JudgeSucceed = 0;
	while (1)
	{
		for (int i = 0; i < 4; i++)
			cap >> img_discard;
		cap >> img;
		int returnnum = catchBlock(img, mode, &actnum);
		cout << returnnum << endl;
		cout << actnum;
		if (JudgeSucceed)
		{
			if (returnnum == -1)
				return;
			else
			{
				if (setMode != 0)
					setClaw(1);
				moveSmall((Dimension)0);
				usleep(200000);
				JudgeSucceed = false;
				continue;
			}
		}
		if (returnnum == 0)
		{
			moveSmall((Dimension)0);
		}
		else if (returnnum == 1)
		{
			moveSmall((Dimension)1);

		}
		else if (returnnum == 2)
		{
			moveTooSmall((Dimension)0);
		}
		else if (returnnum == 3)
		{
			moveTooSmall((Dimension)1);
		}
		else if (returnnum == 4)
		{
			rotateChasis((int)(actnum));
			if (mode == 0)
				moveClaw((ClawState)0);
			else
				moveClaw((ClawState)1);
			setClaw(0);
			usleep(1000000);
			putStore();
			if (setMode == 0)
			{
				usleep(1000000);
				setClaw(1);
				usleep(300000);
			}
			JudgeSucceed = true;
		}
		else if (returnnum == -1)
		{
			return;
		}
		else if (returnnum == 5)
		{
			cross(LEFT);
		}
		else if (returnnum == 6)
		{
			cross(RIGHT);
		}
		usleep(200000);//Origin:500000
	}
}

void game()
{
	//specialMove();
	//moveWhite();
	//moveSmall(FORWARD);
	//moveSmall(FORWARD);
	//rotate(90);
	//moveUntilCorner(FORWARD);
	//rotate(-90);
	//catchTry(1);
	//moveLarge();
	//moveUntilCorner(FORWARD);
	//rotate(-90);
	//moveUntilCorner(FORWARD);
	//rotate(90);
	//catchTry(0);
	//backToMine();
	//moveSmall(FORWARD);
	//openStore();
	specialMove();
	moveWhite();
	moveSmall(FORWARD);
	moveSmall(FORWARD);
	rotate(90);
	moveUntilCorner(FORWARD);
	rotate(-90);
	catchTry(1);
	//moveLarge();
	//moveUntilCorner(FORWARD);
	rotateAbsolute(-90);
	moveWhite();
	rotateAbsolute(90);
	catchTry(0);
	backToMine();
	moveSmall(FORWARD);
	openStore();
}

void newgame()
{
	move1v2();

	//Mat img, img_discard;
	//for (int i = 0; i < 4; i++)
	//	cap >> img_discard;
	//cap >> img;
	//double actnum = -100;
	//catchBlock(img, 1, &actnum,1);
	//if (actnum <= -95)actnum = 0;
	//rotateChasis((int)(actnum));
	//moveClaw((ClawState)2);
	//setClaw(0);
	//rotateAbsolute(30);
	//rotateAbsolute(-30);
	//rotateAbsolute(30);
	//rotateAbsolute(-30);
	//rotateAbsolute(30);
	//rotateAbsolute(-30);
	//rotateAbsolute(30);
	//rotateAbsolute(-30);
	//putStore();
	//setClaw(1);
	//moveLarge();

	catchTry(1);
	backToDimension();
	catchTry(1);
	move2v2();
	catchTry(0);
	backToDimension();
	move3v2();
	catchTry(0,1);
	backToDimension();
	move4v2();
	openStore();
	setClaw(1);
	openStore();
}
void newgame2()
{
	move1v3();
	catchTry(1);
	backToDimension();
	move2v3();
	catchTry(1);
	backToDimension();
	catchTry(1);
	move3v3();
	openStore();
	move4v3();
	catchTry(0);
	backToDimension();
	move5v3();
	catchTry(0);
	backToDimension();
	move6v3();
	openStore();
}

int main()
{
	cout << "Robot start." << endl;
	
	Init();
	cout << "Select mode 0)for game 1)for debug 2)for new game 3)for new game2\n";
	int choose = 0;
	cin >> choose;
	if (choose == 0)
	{
		game();
		return 0;
	}
	else if (choose == 2)
	{
		newgame();
		return 0;
	}
	else if (choose == 3)
	{
		newgame2();
		return 0;
	}
	cout<<"Debug mode"<<endl;
	char buf[100];
	Mat img, img_discard;
	while (1)
	{
		fgets(buf, 100, stdin);
		//if (cap.isOpened())
		//{
			for (int i = 0; i < 4; i++)
				cap >> img_discard;
			cap >> img;
		//}
			if (buf[0] == 's')
			{
				beginWork();
			}
			else if (buf[0] == 'T')
			{
				backToDimension();
			}
			else if (buf[0] == 'Q')
			{
				//imwrite("2.jpg", img);

				//char temp;
				//std::cin >> temp;
				//std::terminate(); // TODO 检查图片
				catchTry(buf[1] - '0');

			}
			else if (buf[0] == 'm')
			{
				moveToward((TargetPlace)(buf[1] - '0'));
			}
			else if (buf[0] == 'z')
			{
				moveTooSmall((Dimension)(buf[1] - '0'));
			}
			else if (buf[0] == 'l')
			{
				moveSmall((Dimension)(buf[1] - '0'));
			}
			else if (buf[0] == 'r')
			{
				rotate(atoi(buf + 1));
			}
			else if (buf[0] == 'g')
			{
				specialMove();
			}
			else if (buf[0] == 'x')
			{
				backToMine();
			}
			else if (buf[0] == 'u')
			{
				rotateAbsolute(atoi(buf + 1));
			}
		else if (buf[0] == 'c')
		{
			moveClaw((ClawState)(buf[1] - '0'));
		}
		else if (buf[0] == 'b')
		{
			moveSmallAlong((Dimension)(buf[1] - '0'));
		}
		else if (buf[0] == 'o')
		{
			setClaw(atoi(buf + 1));
		}
		else if (buf[0] == 'a')
		{
			rotateChasis(atoi(buf + 1));
		}
		else if (buf[0] == 'i')
		{
			setLight(atoi(buf + 1));
		}
		else if (buf[0] == 'q')
		{
			break;
		}
		else if (buf[0] == 't')
		{
			openStore();
		}
		else if (buf[0] == 'p')
		{
			putStore();
		}
		else if (buf[0] == 'k')
		{
			moveUntilCorner((Dimension)(buf[1] - '0'));
		}
		else if (buf[0] == 'e')
		{
			moveWhite();
		}
		else if (buf[0] == 'v')
		{
			moveLarge();
		}
		else if (buf[0] == 'n')
		{
			cross((CrossDimension)(buf[1] - '0'));
		}
		else if (buf[0] == 'h')
		{
			cout << "s: begin work" << endl;
			cout << "m: move toward" << endl;
			cout << "e: move white" << endl;
			cout << "v: move large" << endl;
			cout << "l: move small" << endl;
			cout << "b: move small along" << endl;
			cout << "r: rotate" << endl;
			cout << "c: move claw" << endl;
			cout << "o: set claw" << endl;
			cout << "a: rotate chasis" << endl;
			cout << "i: set light" << endl;
			cout << "t: open store" << endl;
			cout << "p: put store" << endl;
			cout << "k: move until corner" << endl;
			cout << "n: cross" << endl;
			cout << "q: quit" << endl;
		}
		else
		{
			cout << "Unknown Command" << endl;
		}
		for(int i = 0; i < 100; i++)
			buf[i] = 0;
	}



	//Init();
	//CheckStart();
	//ActionType action;
	//while(1)
	//{
	//	action = detect();
	//	sendAct(action);
	//}



	return 0;
}
