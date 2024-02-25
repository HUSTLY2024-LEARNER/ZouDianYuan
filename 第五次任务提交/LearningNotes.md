使用Visual Studio+WSL+CMake开发项目，编译通过但串口打开屡屡失败。搜索后发现可能因为WSL2不支持硬件直接访问，对COM的访问会失败。  
最终采用socat虚拟端口+minicom调试。
具体命令
```Shell
socat  -d  -d  PTY  PTY
```

```Shell
minicom  -D   /dev/pts/1    -b    115200
Ctrl+A X	退出程序
Ctrl+A W	启用/禁用自动换行，默认禁用
Ctrl+A E	启用/禁用输入显示，默认禁用
Ctrl+A C	清屏
```
```PowerShell
New-NetFirewallRule -DisplayName "WSL" -Direction Inbound  -InterfaceAlias "vEthernet (WSL (Hyper-V firewall))"  -Action Allow
```
  
**So difficult!!!!!**  
https://logi.im/script/achieving-access-to-files-and-resources-on-the-network-between-win10-and-wsl2.html
get win-ip:
```Shell
cat /etc/resolv.conf
```
get wsl-ip
```Shell
ip a |grep "global eth0"
```


*Reference*:
https://www.jianshu.com/p/c6bd1da4aa8c
https://zhuanlan.zhihu.com/p/266473873