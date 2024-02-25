#include "SerialPort.hpp"
#include <iostream>

shared_ptr<serial_port> SerialPort::serialPort = nullptr;
char SerialPort::receiveData[] = { 0 };
bool SerialPort::readbit = false;
int SerialPort::readflag = 0;
char SerialPort::undealingData[1024];

SerialPort::SerialPort() :
    portName("/dev/ttyUSB0"),
    baudRate(115200)
{
    memset(receiveData, 0, sizeof(receiveData));
}

SerialPort::~SerialPort()
{
    if (serialPort == nullptr)
        serialPort->close();
}

bool SerialPort::init(string port_name, uint baud_rate)
{
    portName = port_name;
    baudRate = baud_rate;

    return open();
}

void SerialPort::runService()
{
    startAsyncRead();
    io.run();
}

void SerialPort::Start()
{
    readbit = false;
    readflag = 0;
	std::thread t(&SerialPort::runService, this);
	t.detach();
}

bool SerialPort::open()
{
    try
    {
        if (serialPort == nullptr)
            serialPort = shared_ptr<serial_port>(new serial_port(io));

        serialPort->open(portName, errorCode);

        //设置串口参数
        serialPort->set_option(serial_port::baud_rate(baudRate));
        serialPort->set_option(serial_port::flow_control(serial_port::flow_control::none));
        serialPort->set_option(serial_port::parity(serial_port::parity::none));
        serialPort->set_option(serial_port::stop_bits(serial_port::stop_bits::one));
        serialPort->set_option(serial_port::character_size(8));

        return true;

    }
    catch (exception& err)
    {
        cout << "Exception Error: " << err.what() << endl;
    }

    return false;
}

void SerialPort::startAsyncRead()
{
    memset(receiveData, 0, sizeof(receiveData));
    serialPort->async_read_some(boost::asio::buffer(receiveData),
        boost::bind(handleRead,
            boost::asio::placeholders::error,
            boost::asio::placeholders::bytes_transferred));
}

void SerialPort::close()
{
    serialPort->close();
}

void SerialPort::write(string& buf, boost::system::error_code& ec)
{
    serialPort->write_some(boost::asio::buffer(buf), ec);
}

void SerialPort::handleRead(const boost::system::error_code& ec, size_t byte_read)
{
    readbit = true;
    cout<<"Receive:";
    cout.write(receiveData, byte_read);
    cout << endl;
    //char temp[1024];
    int k = 0;
    for (int i = readflag; i < sizeof(undealingData); i++)
    {
        while ((k < (int)byte_read) && (receiveData[k] == IGNORE_C_SIGNAL))
            k++;
        if (k >= (int)byte_read)
            break;
        undealingData[i] = receiveData[k++];
        readflag++;
    }
    //strncpy(undealingData + readflag, receiveData, min(byte_read, sizeof(undealingData) - readflag));
    cout<<"Undealing:";
    cout.write(undealingData, readflag);
    cout << endl;
    //There start another async read, maybe it is wrong
    //No it's correct, but I don't know why
    startAsyncRead();
}

int SerialPort::read(char buf[], boost::system::error_code& ec, int buf_size = 0)
{
    if (!readbit)
        return 0;
    int readNum = readflag;
    readflag = 0;
    if (buf_size == 0)
    {
        strncpy(buf, undealingData, readNum);
        readbit = false;
        return readNum;
    }
    else
    {
        strncpy(buf, undealingData, min(buf_size, readNum));
        readbit = false;
        return min(readNum, buf_size);
    }
}
bool strnstr(char* s1, const char* s2, int pos1)
{
    int l2;
    l2 = strlen(s2);
    if (!l2)
        return true;
    while (pos1 >= l2) {
        pos1--;
        if (!memcmp(s1, s2, l2))
            return true;
        s1++;
    }
    return false;
}
int SerialPort::readUntil(char buf[], boost::system::error_code& ec, string until, int buf_size, int ms_waittime)
{
    struct timeval begin, end;
    gettimeofday(&begin, 0);
    cout<<"Wait for "<<until<<endl;
    while (!strnstr(undealingData, until.c_str(), readflag))
    {
        if (ms_waittime != 0)
        {
			gettimeofday(&end, 0);
			long seconds = end.tv_sec - begin.tv_sec;
			long microseconds = end.tv_usec - begin.tv_usec;
            double elapsed = seconds * 1000.0 + microseconds * 0.001;
            if (elapsed > ms_waittime )
            {
				cout<<"Time out!"<<endl;
				return 0;
			}
		}
        //readflag = 4;
        //undealingData[0] = 's';
        //undealingData[1] = '\r';
        //undealingData[2] = '\n';
        //undealingData[3] = 'A';
        usleep(100000);
        if (readflag >= sizeof(undealingData))
        {
			cout<<"Fail to find!"<<endl;
			return 0;
		}
    }
    cout<<"Find!"<<endl;
    return read(buf, ec, buf_size);
}