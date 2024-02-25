#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <math.h>


using namespace std;
using namespace cv;

//双线性插值
Mat Resize(Mat srcimg, Size dstsize)
{
    Mat dstimg(dstsize, CV_8UC3);
    int dst_h = dstsize.height;
    int dst_w = dstsize.width;
    int src_h = srcimg.rows;
    int src_w = srcimg.cols;
    double scale_w = (double)src_w / dst_w;
    double scale_h = (double)src_h / dst_h;

    double src_x, src_y;
    int src_x_int, src_y_int;
    double src_x_float, src_y_float;
    for (int i = 0; i < dst_h; i++)
        for (int j = 0; j < dst_w; j++)
        {
            src_x = (j + 0.5) * scale_w - 0.5;
            src_y = (i + 0.5) * scale_h - 0.5;
            // 向下取整，代表靠近源点的左上角的那一点的行列号
            src_x_int = (int)src_x;
            src_y_int = (int)src_y;
            // 取出小数部分，用于构造权值
            src_x_float = src_x - src_x_int;
            src_y_float = src_y - src_y_int;

            if (src_x_int + 1 == src_w || src_y_int + 1 == src_h)
            {
                dstimg.at<Vec3b>(i, j) = srcimg.at<Vec3b>(src_y_int, src_x_int);
                continue;
            }
            dstimg.at<Vec3b>(i,j) = (1. - src_y_float) * (1. - src_x_float) * srcimg.at<Vec3b>(src_y_int, src_x_int)+
                (1. - src_y_float) * src_x_float * srcimg.at<Vec3b>(src_y_int, src_x_int + 1)+
                src_y_float * (1. - src_x_float) * srcimg.at<Vec3b>(src_y_int + 1, src_x_int)+
                src_y_float * src_x_float * srcimg.at<Vec3b>(src_y_int + 1, src_x_int + 1);
        }
    return dstimg;
}

int main()
{
    Mat a;
    a = imread("../srcimg/car3.png");
    if (a.empty()) {
        cout << "Path reading failed." << endl;
        return -1;
    }
    Mat result(Resize(a, Size(300, 300)));
    imshow("Origin", a);
    imshow("Result", result);
    waitKey(0);
    
	return 0;
}