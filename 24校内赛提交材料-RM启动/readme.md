# ReadMe
## 目录结构
``` Shell
.
├── Code
│   ├── Horizon
│   │   ├── CMakeLists.txt
│   │   ├── bin
│   │   ├── build
│   │   ├── includes
│   │   │   ├── Horizon.hpp    
│   │   │   └── SerialPort.hpp #串口通信头文件
│   │   └── src
│   │       ├── SerialPort.cpp #串口通信实现
│   │       ├── control.cpp    #上位机发送指令实现
│   │       ├── init.cpp       #完成初始化操作
│   │       └── main.cpp       #算法重要实现文件+main函数
│   └── NumberClassify         
│       ├── NumberClassify.cpp #模型使用（可手动调用predict函数避开传统视觉方法处理）
│       ├── NumberClassify.py  #模型构建+训练
│       └── frozen_graph.pb    #已经训练好的模型
├── NumberClassify.md
├── NumberClassify.pdf
├── Report.md
├── Report.pdf
├── readme.md
├── 串口通信协议-上位机.md
└── 串口通信协议-上位机.pdf
```
## 代码运行
Horizon文件夹内为车上实装代码，使用CMake编译  
**本队伍摄像头广角很小，无畸变，使用USB连接，可调整亮度等参数**  
**可以*建议*使用本队伍实车的开发板与摄像头测试效果**~~以避免出现奇怪的bug~~  
**队伍实车开发板的开发目录 /home/sunrise/HorizonRobot**  
**队伍实车开发板的可执行文件目录 /home/sunrise/HorizonRobot/bin**  
**运行前请更改相机设备号，串口目录，模型目录**  
**注意：程序在运行前会调整相机参数，所有图像处理的参数的事先选取都在相应的相机参数下进行，如果与当前系统冲突可能导致程序无法运行成功或图形检测功能异常，可以手动更改程序定义的相机参数以达到合适的运行效果**  
* NumberClassify在Windows系统下  
* Horizon在Ubuntu系统下   

**运行前请检查OpenCV版本，Tensorflow版本，Python版本，CMake版本，Ubuntu/WSL/Windows版本**  
* OpenCV 4.8.0    
* Tensorflow 2.15.0  
* Python 3.11.5
* CMake 3.22.1


main.cpp文件函数解释：
```C++
void findWall(Mat& OriginImg, double& theta);
//根据图片使用OpenCV计算墙面角度
void backToDimension(bool strictExam = 0);
//根据角度自动校准车身姿态
Point2f getFitCircle(Point2f pt1, Point2f pt2, Point2f pt3, double* _radius);
//计算过三点的圆（可参考Report文档中的算法部分）
int catchBlock(Mat& OriginImg, int mode, double* actnum, int strictMode = 0);
//通过图片判断当前执行的动作
void catchTry(int mode, int setMode = 0,int strictMode=0);
//进行抓取
void game();
void newgame();
void newgame2();
//三种固定的车身行动模式，会自动调用一系列函数完成相应功能
```