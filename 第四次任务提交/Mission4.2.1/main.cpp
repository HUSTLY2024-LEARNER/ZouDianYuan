#include <iostream>
#include <opencv2/opencv.hpp>
#include "Simulator.hpp"

using namespace std;

template<typename T, int x>
class LowPassFilter
{
private:
    Eigen::Matrix<T, x, 1> alpha;

public:
    Eigen::Matrix<T, x, 1> LastOutput = Eigen::Matrix<T, x, 1>::Zero();
    Eigen::Matrix<T, x, 1> Output;
    Eigen::Matrix<T, x, 1> Input;
    LowPassFilter(Eigen::Matrix<T, x, 1> alpha)
    {
		this->alpha = alpha;
	}
    LowPassFilter(T alpha)
    {
        this->alpha = Eigen::Matrix<T, x, 1>::Constant(alpha);
    }
    void update(void)
    {
        Output = alpha.cwiseProduct(LastOutput) + (Eigen::Matrix<T, x, 1>::Constant(1) - alpha).cwiseProduct(Input);
        LastOutput = Output;
    }
};


int main() {
    srand(1);
    LowPassFilter<double, 2> *lpf;
    lpf = new LowPassFilter<double, 2>(0.99);
    Simulator<double, 2> *simulator;
    simulator = new Simulator<double, 2>(Eigen::Vector2d(0, 0), 1, Eigen::Vector2d(0.1, 0.1)); // 输入为起始点与方差
    Eigen::Vector2d measurement;
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0, 0, 0));
    double t = 0;
    while (1) {
		measurement = simulator->getMeasurement(t);
		lpf->Input = measurement;
		lpf->update();
		cv::circle(img, cv::Point((int)(measurement[0] * 10 + 250), int(measurement[1] * 10 + 250)), 2, cv::Scalar(0, 0, 255), -1);
		cv::circle(img, cv::Point((int)(lpf->Output[0] * 10 + 250), (int)(lpf->Output[1] * 10 + 250)), 2, cv::Scalar(0, 255, 0), -1);

		cv::imshow("img", img);
		cv::waitKey(10);
		t++;
	}  
    return 0;
}
