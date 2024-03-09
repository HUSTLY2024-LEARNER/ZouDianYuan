# YOLOv8
训练YOLO时可以从中断点回复，要记得把模型换成中断产生的pt文件，再修改config resume=True，这时workers的大小完全由旧训练模型指定。  
workers不要设的太小（巨慢）和太大（C盘的虚拟内存会崩掉，占用50G+），适中可以保证GPU不会空置。  
如果发现gpu上下浮动特别明显，建议增加workers。  
可以通过nvidia-smi -lms 实时查看显卡真实占用  
必须通过在一开始定义model时更改model才能够改变模型架构，即
```Python
    model = YOLO("D:\Desktop\py-program\RoboMaster_YOLO\myyolov8s.yaml",task='detect')
```
不使用这种方法无法成功改变model。且pretrain参数失效  
增加网络架构的流程为
- 打开tasks.py 在开头和parsemodel处分别引入。添加：
```Python
    elif m in {ShuffleAttention}: #在此处添加自己的注意力机制，命名是刚刚的类名
                args = [ch[f], *args] 
```
- 在modules里面新建py代码，更改__init__.py 然后使用python -m py_compile xxx.py命令编译  
- 复制一个yolo.yaml 魔改即可
Reference: https://blog.csdn.net/qq_43471945/article/details/132685642