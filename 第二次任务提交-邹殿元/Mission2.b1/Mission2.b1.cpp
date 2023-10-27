#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <math.h>



using namespace std;
using namespace cv;

int BGR2HSV(Mat &src,Mat &result)
{
    if ((result.cols != src.cols) || (result.rows != src.rows))
        return 0;
    int r, g, b;
    int v;
    double s, h;

    for(int i=0;i<src.rows;i++)
        for (int j = 0; j < src.cols; j++)
        {
            r = src.at<Vec3b>(i, j)[2];
            g = src.at<Vec3b>(i, j)[1];
            b = src.at<Vec3b>(i, j)[0];
            v = max({ r, g, b });
            s = v ? ((v - min({ r,g,b })) / (double)v) : 0;

            if (v == min({ r,g,b }))
                h = 0;
            else if (r == v)
                h = 60.0 * (g - b) / (v - min({ r,g,b }));
            else if (g == v)
                h = 120 + 60.0 * (b - r) / (v - min({ r, g, b }));
            else
                h = 240 + 60.0 * (r - g) / (v - min({ r,g,b }));
            if (h < 0)
                h += 360;
            h /= 2;
            s *= 255;
            result.at<Vec3b>(i, j)[0] = (unsigned char)h;
            result.at<Vec3b>(i, j)[1] = (unsigned char)s;
            result.at<Vec3b>(i, j)[2] = (unsigned char)v;
        }
    return 1;
}


int main()
{
    Mat a;
    a = imread("../srcimg/car3.png");
    if (a.empty()) {
        cout << "Path reading failed." << endl;
        return -1;
    }
    Mat result(a.rows, a.cols, CV_8UC3, Scalar(0, 0, 0));
    Mat resultGRAY(a.rows, a.cols, CV_8UC1, Scalar(0));
    if(!BGR2HSV(a, result))
        exit(-1);
    imshow("Test", a);
    imshow("HSV", result);
    waitKey(0);
    
	return 0;
}