#include "CMakeProject1.h"
using namespace cv;
using namespace std;
using namespace Eigen;

int main(void)
{
	Eigen::Vector3d origin(1, 1, 1);

	double yaw, pitch, roll;
	printf("This program's order is roll ,pitch and yaw.\n");
	printf("Based on righthand system\n");
	printf("Input yaw pitch roll in order.( rad )\n");
	cin >> yaw >> pitch >> roll;
	Eigen::Vector3d eulerAngle(yaw, pitch, roll);

	Eigen::AngleAxisd rollAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitX()));
	Eigen::AngleAxisd pitchAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
	Eigen::AngleAxisd yawAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitZ()));

	Eigen::Matrix3d rotation_matrix;
	rotation_matrix = yawAngle * pitchAngle * rollAngle;

	Eigen::Affine3d poseMatrix = Translation3d(3, 0, 0) * rotation_matrix;
	viz::Viz3d myWindow("Euler angle");
	myWindow.showWidget("coordinate", viz::WCoordinateSystem());
	
	myWindow.showWidget("TranslatedCoordinate", viz::WCoordinateSystem(), poseMatrix);
	myWindow.spin();

	return 0;
}

