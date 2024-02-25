#include "Horizon.hpp"

void beginWork()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + STARTWORK_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(STARTWORK_SIGNAL, 1000));
}
void moveToward(TargetPlace target) {
	if (currentPlace == ORIGINALPLACE) {
		if (target == BLUEBRICK_LEFT) {
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			currentPlace = PURPLEBRICK_LEFT;
		}
		else if (target == BLUEBRICK_RIGHT) {
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			currentPlace = BLUEBRICK_RIGHT;
		}
		else if (target == PURPLEBRICK_LEFT) {
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			currentPlace = PURPLEBRICK_LEFT;
		}
		else if (target == PURPLEBRICK_RIGHT) {
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			currentPlace = PURPLEBRICK_RIGHT;
		}
		else if (target == ENERGY) {
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(90);
			currentPlace = ENERGY;
		}
	}
	else if (currentPlace == BLUEBRICK_LEFT) {
		if (target == BLUEBRICK_RIGHT) {
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			currentPlace = BLUEBRICK_RIGHT;
		}
		else if (target == PURPLEBRICK_LEFT) {
			rotate(90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			currentPlace = PURPLEBRICK_LEFT;
		}
		else if (target == PURPLEBRICK_RIGHT) {
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			currentPlace = PURPLEBRICK_RIGHT;
		}
		else if (target == ENERGY) {
			rotate(90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(90);
			currentPlace = ENERGY;
		}
	}
	else if (currentPlace == BLUEBRICK_RIGHT) {
		if (target == BLUEBRICK_LEFT) {
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			currentPlace = BLUEBRICK_LEFT;
		}
		else if (target == PURPLEBRICK_LEFT) {
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			currentPlace = PURPLEBRICK_LEFT;
		}
		else if (target == PURPLEBRICK_RIGHT) {
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			currentPlace = PURPLEBRICK_RIGHT;
		}
		else if (target == ENERGY) {
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(90);
			currentPlace = ENERGY;
		}
	}
	else if (currentPlace == PURPLEBRICK_LEFT) {
		if (target == BLUEBRICK_LEFT) {
			rotate(90);
			rotate(90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			currentPlace = BLUEBRICK_LEFT;
		}
		else if (target == BLUEBRICK_RIGHT) {
			rotate(90);
			rotate(90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			currentPlace = BLUEBRICK_RIGHT;
		}
		else if (target == PURPLEBRICK_RIGHT) {
			rotate(90);
			rotate(90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			currentPlace = PURPLEBRICK_RIGHT;
		}
		else if (target == ENERGY) {
			rotate(90);
			moveUntilCorner();
			rotate(90);
			currentPlace = ENERGY;
		}
	}
	else if (currentPlace == PURPLEBRICK_RIGHT) {
		if (target == BLUEBRICK_LEFT) {
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			currentPlace = BLUEBRICK_LEFT;
		}
		else if (target == BLUEBRICK_RIGHT) {
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			currentPlace = BLUEBRICK_RIGHT;
		}
		else if (target == PURPLEBRICK_LEFT) {
			rotate(-90);
			rotate(-90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			currentPlace = PURPLEBRICK_LEFT;
		}
		else if (target == ENERGY) {
			rotate(-90);
			rotate(-90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(90);
			currentPlace = ENERGY;
		}
	}
	else if (currentPlace == ENERGY) {
		if (target == PURPLEBRICK_LEFT) {
			rotate(90);
			moveUntilCorner();
			rotate(90);
			currentPlace = PURPLEBRICK_LEFT;
		}
		else if (target == PURPLEBRICK_RIGHT) {
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			currentPlace = PURPLEBRICK_RIGHT;
		}
		else if (target == BLUEBRICK_LEFT) {
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			currentPlace = BLUEBRICK_LEFT;
		}
		else if (target == BLUEBRICK_RIGHT) {
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			moveUntilCorner();
			rotate(-90);
			moveUntilCorner();
			moveUntilCorner();
			rotate(90);
			currentPlace = BLUEBRICK_RIGHT;
		}
	}
}
void specialMove()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + SPECIALMOVE_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(SPECIALMOVE_SIGNAL, 100));
}
void moveWhite()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + WHITEMOVE_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(WHITEMOVE_SIGNAL, 100));
}
void moveUntilCorner(Dimension dimension)
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVETOWARD_SIGNAL + to_string(dimension) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVETOWARD_SIGNAL, 100));
	usleep(700000);
	moveSmall(FORWARD);
	usleep(700000);
	moveSmall(FORWARD);
}
void moveTooSmall(Dimension dimension)
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVETOOLITTLE_SIGNAL + to_string(dimension) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVETOOLITTLE_SIGNAL, 100));
}
void moveSmall(Dimension dimension)
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + SMALLMOVE_SIGNAL + to_string(dimension) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	}while(!checkSignal(SMALLMOVE_SIGNAL, 100));
}
void moveSmallAlong(Dimension dimension)
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + SMALLMOVEALONG_SIGNAL + to_string(dimension) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(SMALLMOVEALONG_SIGNAL, 100));
}
void moveLarge()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + LARGEMOVE_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(LARGEMOVE_SIGNAL, 100));
}
void rotate(int angle)//angle not in rad + for left - for right
{
	boost::system::error_code ec;
	char temp[10] = { 0 };
	sprintf(temp,"%+03d", angle);
	string buf = DATABEGIN_SIGNAL + ROTATE_SIGNAL + string(temp) + END_SIGNAL;
	do{
	serialPort.write(buf, ec);
	currentDimension = (AbsoluteDimension)(((int)currentDimension - angle / 90) % 4);
	}while(!checkSignal(ROTATE_SIGNAL, 100));
}
void moveClaw(ClawState clawstate)//0 for blue 1 for purple
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVECLAW_SIGNAL + to_string(clawstate) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	}while(!checkSignal(MOVECLAW_SIGNAL, 100));
}
void setClaw(bool open)
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + SETCLAW_SIGNAL + to_string(open) + END_SIGNAL;
	do{
		serialPort.write(buf, ec);
	}while(!checkSignal(SETCLAW_SIGNAL, 100));
}
void rotateChasis(int angle)//angle not in rad + for left - for right
{
	boost::system::error_code ec;
	char temp[10] = { 0 };
	sprintf(temp, "%+03d", angle);
	string buf = DATABEGIN_SIGNAL + ROTATECHASIS_SIGNAL + string(temp) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(ROTATECHASIS_SIGNAL, 100));
}
void setLight(bool open)
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + SETLIGHT_SIGNAL + to_string(open) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(SETLIGHT_SIGNAL, 100));
}
void backToMine()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + BACKTOMINE_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(BACKTOMINE_SIGNAL, 100));
}
void openStore()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + OPENSTORE_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(OPENSTORE_SIGNAL, 100));
}
void putStore()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + PUTSTORE_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(PUTSTORE_SIGNAL, 100));
}
void cross(CrossDimension dimension)
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + CROSS_SIGNAL + to_string(dimension) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(CROSS_SIGNAL, 100));
}
void move1v2()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE1V2_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE1V2_SIGNAL, 100));
}
void move2v2()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE2V2_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE2V2_SIGNAL, 100));
}
void move3v2()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE3V2_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE3V2_SIGNAL, 100));
}
void move4v2()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE4V2_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE4V2_SIGNAL, 100));
}
void move1v3()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE1V3_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE1V3_SIGNAL, 100));
}
void move2v3()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE2V3_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE2V3_SIGNAL, 100));
}
void move3v3()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE3V3_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE3V3_SIGNAL, 100));
}
void move4v3()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE4V3_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE4V3_SIGNAL, 100));
}
void move5v3()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE5V3_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE5V3_SIGNAL, 100));
}
void move6v3()
{
	boost::system::error_code ec;
	string buf = DATABEGIN_SIGNAL + MOVE6V3_SIGNAL + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(MOVE6V3_SIGNAL, 100));
}
void rotateAbsolute(int angle)
{
	boost::system::error_code ec;
	char temp[10] = { 0 };
	sprintf(temp, "%+03d", angle);
	string buf = DATABEGIN_SIGNAL + ROTATEABSOLUTE_SIGNAL + string(temp) + END_SIGNAL;
	do {
		serialPort.write(buf, ec);
	} while (!checkSignal(ROTATEABSOLUTE_SIGNAL, 100));
}



bool checkSignal(string checksignal, int ms_waittime)
{
	char buf[1024];
	int size = waitSignalUntil(END_SIGNAL, buf, 1024, ms_waittime);
	if (size == 0)
		return false;
	else
	{
		if (strnstr(buf, checksignal.c_str(), 1024))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
}
int waitSignalUntil(string signal, char buf[], int size, int ms_waittime)
{
	boost::system::error_code ec;
	return serialPort.readUntil(buf, ec, signal, size, ms_waittime);
}
int waitSignal(string signal, char buf[], int size)
{
	boost::system::error_code ec;
	return serialPort.readUntil(buf, ec, signal, size);
}