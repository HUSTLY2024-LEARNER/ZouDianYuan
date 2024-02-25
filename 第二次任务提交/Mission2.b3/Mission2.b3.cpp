#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <math.h>



using namespace std;
using namespace cv;

Mat getCard(Mat& src, Mat erodekernel, Mat dilatekernel,int mode)//0:strict,+inf:casual
{
    Mat hsvImage,  binaryImage, closeImage, erodeImage, dilateImage;
    cvtColor(src, hsvImage, COLOR_BGR2HSV);
    inRange(hsvImage, Scalar(110 - mode, 100-mode, 150), Scalar(130 + mode, 255, 255), binaryImage);
    morphologyEx(binaryImage, closeImage, MORPH_CLOSE, erodekernel);
    erode(closeImage, erodeImage, erodekernel);
    dilate(erodeImage, dilateImage, dilatekernel);
    return dilateImage;
}
vector<vector<Point>> getExternalContours(Mat& dilateImage)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dilateImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return contours;
}
Mat LCC(const Mat& src) {
    int rows = src.rows;
    int cols = src.cols;
    int** I;
    I = new int* [rows];
    for (int i = 0; i < rows; i++) {
        I[i] = new int[cols];
    }
    int** inv_I;
    inv_I = new int* [rows];
    for (int i = 0; i < rows; i++) {
        inv_I[i] = new int[cols];
    }
    Mat Mast(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; i++) {
        uchar* data = Mast.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            I[i][j] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[1]) / 3.0;
            inv_I[i][j] = 255;
            *data = inv_I[i][j] - I[i][j];
            data++;
        }
    }
    GaussianBlur(Mast, Mast, Size(41, 41), BORDER_DEFAULT);
    Mat dst(rows, cols, CV_8UC3);
    for (int i = 0; i < rows; i++) {
        uchar* data = Mast.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < 3; k++) {
                float Exp = pow(2, (128 - data[j]) / 128.0);
                int value = int(255 * pow(src.at<Vec3b>(i, j)[k] / 255.0, Exp));
                dst.at<Vec3b>(i, j)[k] = value;
            }
        }
    }
    return dst;
}

int main()
{
    Mat a;
    a = imread("../srcimg/car3.png");
    if (a.empty()) {
        cout << "Path reading failed." << endl;
        return -1;
    }
    Mat gammaa=LCC(a);
    a = gammaa.clone();

    Mat resultImage;
    RotatedRect contoursRrect;


    int flag = 0;
    int maxarea_i = 0;
    int count = 0;
    vector<vector<Point>> contours;
    do{    //Get binary image and its contours
        Mat dilateImage = getCard(a, getStructuringElement(MORPH_RECT, Size(1, 1)),
            getStructuringElement(MORPH_RECT, Size(25, 25)), count);
        //imshow("a", dilateImage);
        //waitKey();
        contours = getExternalContours(dilateImage);

        double rectdegree = 1;//Rectangle similar degree
        for (int i = 0; i < contours.size(); i++)
        {
            contoursRrect = minAreaRect(contours[i]);
            if ((max(contoursRrect.size.height, contoursRrect.size.width) /
                min(contoursRrect.size.height, contoursRrect.size.width) > 1.8-count*0.03) &&
                (max(contoursRrect.size.height, contoursRrect.size.width) /
                    min(contoursRrect.size.height, contoursRrect.size.width) < 5))//isValidCarLicense
            {
                rectdegree = contourArea(contours[i]) / (double)contoursRrect.size.height / contoursRrect.size.width;
                //printf("%d:%f\n", i, rectdegree);
                if (rectdegree < 0.8-count*0.01) {
                    continue;
                }
                if (flag == 0)
                {
                    maxarea_i = i;
                    flag = 1;
                }
                if (contourArea(contours[i]) > contourArea(contours[maxarea_i]))
                    maxarea_i = i;
            }
        }
        count+=5;
    } while (flag == 0);

    contoursRrect = minAreaRect(contours[maxarea_i]);
    Mat cutImage = a(boundingRect(contours[maxarea_i]));//cut origin image
    Mat m;
    if (contoursRrect.size.height > contoursRrect.size.width)//correct angle 
    {
        contoursRrect.angle -= 90;
        m = getRotationMatrix2D(Point(0, cutImage.rows), contoursRrect.angle, 1.0);
    }
    else
    {
        m = getRotationMatrix2D(Point(cutImage.cols, 0), contoursRrect.angle, 1.0);
    }
    //Rotate the cut image
    Mat warpImage;
    warpAffine(cutImage, warpImage, m, cutImage.size() * 4);
    //Get binary image and its contours
    Mat dilateImage2=getCard(warpImage, getStructuringElement(MORPH_RECT, Size(1, 1)),
        getStructuringElement(MORPH_RECT, Size(25, 25)),count);
    vector<vector<Point>> contours2 = getExternalContours(dilateImage2);
    //cut the rotated image;
    if (!contours2.empty())
        resultImage = warpImage(boundingRect(contours2[0]));
    else
        resultImage = dilateImage2.clone();

    //imshow("Origin", a);//show the origin image
    resize(resultImage, resultImage, Size(200.0 * resultImage.cols / resultImage.rows,200));
    if (!resultImage.empty())
        imshow("Result", resultImage);
    else
        printf("Generate error!\n");
    waitKey(0);

	return 0;
}