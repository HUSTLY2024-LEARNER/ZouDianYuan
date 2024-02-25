#include "Horizon.hpp"

//#define DEVICE_NAME "/dev/pts/3"
#define DEVICE_NAME "/dev/ttyS3"
using namespace cv;

extern cv::VideoCapture cap;
void Init()
{
	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640); //ͼ��Ŀ���Ҫ���֧�ִ˿�
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480); //ͼ��ĸߣ���Ҫ���֧�ִ˸�
	int deviceID = 8; //����豸��
	cap.open(deviceID); //�����
	cap.set(CAP_PROP_BUFFERSIZE, 1);
	//Sleep(5000);
	if (!cap.isOpened()) //�ж�����Ƿ��
	{
		std::cerr << "ERROR!!Unable to open camera\n";
		exit(-1);
	}
	cap.set(CAP_PROP_BRIGHTNESS, 0.390625);//���� 0
	//cap.set(CAP_PROP_CONTRAST, 10);//�Աȶ� 16
	//cap.set(CAP_PROP_SATURATION, 36);//���Ͷ� 36
	//cap.set(CAP_PROP_HUE, 0); //ɫ�� 0
	//cap.set(CAP_PROP_GAMMA, 250);//٤�� 100

	printf("Brightness:%f\n", cap.get(CAP_PROP_BRIGHTNESS));
	printf("Contrast:%f\n", cap.get(CAP_PROP_CONTRAST));
	printf("Saturation:%f\n", cap.get(CAP_PROP_SATURATION));
	printf("Hue:%f\n", cap.get(CAP_PROP_HUE));
	printf("Gamma:%f\n", cap.get(CAP_PROP_GAMMA));

	//Exposure and Gain is not supported
	//cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25);//�ر��Զ��ع�
	//cap.set(CAP_PROP_APERTURE, 10);//��Ȧ
	//cap.set(CAP_PROP_TEMPERATURE,1);//��ƽ��
	//cap.set(CAP_PROP_SETTINGS, 1);//����

	Mat temp;
	cap >> temp;
	imwrite("1.jpg", temp);

	if(!serialPort.init(DEVICE_NAME, 460800))
	{
		cout << "Init Serial Port Failed" << DEVICE_NAME << endl;
		exit(0);
	}
	serialPort.Start();
	boost::system::error_code ec;
	//string buf = "Hello World!";
	//serialPort.write(buf, ec);
	beginWork();
	usleep(1000000);
	cout << "Receive Signal\n";
}