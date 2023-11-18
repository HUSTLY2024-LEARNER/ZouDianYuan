#include <iostream>
#include <opencv2/opencv.hpp>
#include "KalmanFilter.hpp"
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
    void predict(Eigen::Matrix<T, x, 1> InputArray)
	{
        Input = InputArray;
	}
    void update(void)
    {
        Output = alpha.cwiseProduct(LastOutput) + (Eigen::Matrix<T, x, 1>::Constant(1) - alpha).cwiseProduct(Input);
        LastOutput = Output;
    }
};


int main() {
    srand(1);
    // 以一个静止滤波为例展示使用方法
    // 1. 初始化
    // 滤波器初始化
    KalmanFilter<double, 2, 2> *kf;
    kf = new KalmanFilter<double, 2, 2>();
    LowPassFilter<double, 2> *lpf;
    lpf = new LowPassFilter<double, 2>(0.98);
    // 仿真器初始化
    Simulator<double, 2> *simulator;
    simulator = new Simulator<double, 2>(Eigen::Vector2d(0, 0), 10, Eigen::Vector2d(10, 10), Eigen::Vector2d(-10, 10)); // 输入为起始点与方差
    //simulator = new Simulator<double, 2>(Eigen::Vector2d(0, 0), 3, Eigen::Vector2d(0.01, 0.01)); // 输入为起始点与方差


    // 2. 设置状态转移矩阵
    kf->transition_matrix << 1, 0,
            0, 1;
    // 3. 设置测量矩阵
    kf->measurement_matrix << 1, 0,
            0, 1;
    // 4. 设置过程噪声协方差矩阵
    kf->process_noise_cov << 0.01, 0,
            0, 0.01;
    // 5. 设置测量噪声协方差矩阵
    kf->measurement_noise_cov << 10, 0,
            0, 10;
    // 6. 设置控制向量
    kf->control_vector << 0,
            0;

    // 生成随机点
    Eigen::Vector2d measurement;
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(0, 0, 0));
    double t = 0;
    while (1) {
        measurement = simulator->getMeasurement(t);
        // 7. 预测
        kf->predict(measurement);
        lpf->predict(measurement);
        // 8. 更新
        kf->update();
        lpf->update();
        // 9. 获取后验估计
        Eigen::Vector2d estimate = kf->posteriori_state_estimate;
        Eigen::Vector2d estimate_low = lpf->Output;
        //img = cv::Mat::zeros(500, 500, CV_8UC3);
        // 10. 绘制出观测点和滤波点（平移到绘图中心），亦可采用其他可视化方法（matplotlib、VOFA+、Foxglove均可）
        cv::circle(img, cv::Point((int)(measurement[0] * 10 + 250), int(measurement[1] * 10 + 250)), 2, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, cv::Point((int)(estimate_low[0] * 10 + 250), (int)(estimate_low[1] * 10 + 250)), 2, cv::Scalar(255, 0, 0), -1);
        cv::circle(img, cv::Point((int)(estimate[0] * 10 + 250), (int)(estimate[1] * 10 + 250)), 2, cv::Scalar(0, 255, 0), -1);

        cv::imshow("img", img);
        cv::waitKey(10);
        t++;
    }

    return 0;
}
