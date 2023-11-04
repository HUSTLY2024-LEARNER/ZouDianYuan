#include "CMakeProject1.h"
using namespace cv;
using namespace std;
using namespace Eigen;

Mat getCard(Mat& src)//0:strict,+inf:casual
{
    Mat hsvImage, binaryImage, closeImage, erodeImage, dilateImage;
    cvtColor(src, hsvImage, COLOR_BGR2HSV);
    inRange(hsvImage, Scalar(90, 90, 150), Scalar(150, 255, 255), binaryImage);
    return binaryImage;
}

void findmin(vector<Point> array, Point2f& point)
{
    int min = array[0].y;
    Point2f temppoint = array[0];
    for (int i = 1; i < array.size(); i++)
        if (min > array[i].y)
        { 
            min = array[i].y;
            temppoint = array[i];
        }
    point = temppoint;
}
void findmax(vector<Point> array, Point2f& point)
{
    int max = array[0].y;
    Point2f temppoint = array[0];
    for (int i = 1; i < array.size(); i++)
        if (max < array[i].y)
        {
            max = array[i].y;
            temppoint = array[i];
        }
    point = temppoint;
}
int main(void)
{
    const static vector<Point3f> points_small_3d = { Point3f(-0.0275f, 0.0675f, 0.f),
                                                Point3f(0.0275f, 0.0675f, 0.f),
                                                Point3f(0.0275f, -0.0675f, 0.f),
                                                Point3f(-0.0275f, -0.0675f, 0.f) };
    Mat a = imread("/usr/1_raw.png");


    Mat dilateImage=getCard(a);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dilateImage, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    //for (int i = 0; i < contours.size(); i++)
    //contours[6]left 8right
    vector<Point2f> img_point(4);
    findmax(contours[8], img_point[0]);
    findmin(contours[8], img_point[1]);
    findmin(contours[6], img_point[2]);
    findmax(contours[6], img_point[3]);
    cout << img_point << endl;

    Mat cameraMat = (cv::Mat_<double>(3, 3) << 1900, 0, 960, 0, 1900, 540, 0, 0, 1);
    Mat rvec, tvec;
    Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0);
    solvePnP(points_small_3d, img_point, cameraMat, distCoeffs, rvec, tvec);
    Matx33d rmat;
    Rodrigues(rvec, rmat);
    cout << "rmat=" << rmat << endl;
    cout << "tvec=" << tvec << endl;
    
    cv::Affine3d affineMat(rmat, tvec);
    
    viz::Viz3d myWindow("Test");
    myWindow.showWidget("TranslatedCoordinate", viz::WCoordinateSystem(), affineMat);
    myWindow.spin();
    //imshow("Origin", a);
    //imshow("Test", dilateImage);
    //waitKey();
    //solvePnPRansac(points_small_3d,)
	return 0;
}