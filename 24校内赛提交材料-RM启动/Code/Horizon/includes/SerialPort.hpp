#pragma once

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/array.hpp>
#include <memory>
#include <sys/time.h>
#include <thread>
#define IGNORE_C_SIGNAL 'A'

using namespace std;
using namespace boost::asio;
bool strnstr(char* s1, const char* s2, int pos1);
class SerialPort
{
public:
    SerialPort();
    ~SerialPort();

    static bool readbit;
    static int readflag;
    static char undealingData[1024];

    bool init(string port_name, uint baud_rate);
    void runService();
    bool open();
    void close();
    void write(string& buf, boost::system::error_code& ec);
    int read(char buf[], boost::system::error_code& ec, int buf_size);
    int readUntil(char buf[], boost::system::error_code& ec, string until, int buf_size=0, int ms_waittime=0);
    void Start();

    static void startAsyncRead();
    static void handleRead(const boost::system::error_code& ec, size_t byte_read);
private:
    boost::system::error_code errorCode;
    io_service io;
    static shared_ptr<serial_port> serialPort;
    string portName;
    uint baudRate;

    static char receiveData[1024];
};
