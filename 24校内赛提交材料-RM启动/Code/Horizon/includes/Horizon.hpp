
#pragma once

#include <iostream>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include "SerialPort.hpp"

#define STARTWORK_SIGNAL string("s")
#define DATABEGIN_SIGNAL string("S")
#define END_SIGNAL string("\r\n")
//#define END_SIGNAL string("K")

#define SPECIALMOVE_SIGNAL string("g")
#define BACKTOMINE_SIGNAL string("x")
#define MOVETOOLITTLE_SIGNAL string("z")
#define MOVETOWARD_SIGNAL string("m")
#define WHITEMOVE_SIGNAL string("e")
#define SMALLMOVE_SIGNAL string("l")
#define SMALLMOVEALONG_SIGNAL string("b")
#define LARGEMOVE_SIGNAL string("v")
#define ROTATE_SIGNAL string("r")
#define MOVECLAW_SIGNAL string("c")
#define SETCLAW_SIGNAL string("o")
#define ROTATECHASIS_SIGNAL string("a")
#define SETLIGHT_SIGNAL string("i")
#define OPENSTORE_SIGNAL string("t")
#define PUTSTORE_SIGNAL string("p")
#define CROSS_SIGNAL string("n")
#define ROTATEABSOLUTE_SIGNAL string("u")
#define MOVE1V2_SIGNAL string("W")
#define MOVE2V2_SIGNAL string("X")
#define MOVE3V2_SIGNAL string("Y")
#define MOVE4V2_SIGNAL string("Z")

#define MOVE1V3_SIGNAL string("B")
#define MOVE2V3_SIGNAL string("C")
#define MOVE3V3_SIGNAL string("D")
#define MOVE4V3_SIGNAL string("E")
#define MOVE5V3_SIGNAL string("F")
#define MOVE6V3_SIGNAL string("G")

extern SerialPort serialPort;

typedef enum {
	BLUEBRICK_RIGHT,
	BLUEBRICK_LEFT,
	PURPLEBRICK_RIGHT,
	PURPLEBRICK_LEFT,
	ENERGY,
	ORIGINALPLACE
} TargetPlace;
typedef enum {
	FORWARD,
	BACKWARD
} Dimension;
typedef enum {
	BLUESTATE,
	PURPLESTATE
} ClawState;
typedef enum {
	LEFT,
	RIGHT
} CrossDimension;
typedef enum {
	N,E,S,W
}AbsoluteDimension;
typedef enum {
	TURNLEFT,
	TURNRIGHT
}RotateDimension;

extern TargetPlace currentPlace;
extern AbsoluteDimension currentDimension;

int waitSignal(string signal, char buf[], int size = 0);
int waitSignalUntil(string signal, char buf[], int size = 0, int ms_waittime = 0);
bool checkSignal(string checksignal, int ms_waittime = 0);
void beginWork();
void moveToward(TargetPlace target);
void specialMove();
void moveWhite();
void moveUntilCorner(Dimension dimension = FORWARD);
void moveSmall(Dimension dimension);
void moveTooSmall(Dimension dimension);
void moveSmallAlong(Dimension dimension);
void moveLarge();//Only for backward
void rotate(int angle);//angle not in rad + for left - for right
void moveClaw(ClawState clawstate);//0 for blue 1 for purple
void setClaw(bool open);
void rotateChasis(int angle);//angle not in rad + for left - for right
void setLight(bool open);
void openStore();
void putStore();
void backToMine();
void cross(CrossDimension dimension);
void rotateAbsolute(int angle);
void move1v2();
void move2v2();
void move3v2();
void move4v2();
void move1v3();
void move2v3();
void move3v3();
void move4v3();
void move5v3();
void move6v3();






void Init();