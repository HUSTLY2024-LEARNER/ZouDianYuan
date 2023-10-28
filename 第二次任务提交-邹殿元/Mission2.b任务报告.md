# 任务报告（*Mission2.b*）

## 时间
2023/10/22 ~2023/10/28

## 任务一
任务很基础，在网上搜索相关资料后，使用at方法生成了HSV对应图像。  
使用如下图片进行测试，输出结果如下。
[mission2.b1.srcimg] (Test)
![mission2.b1.resultimg]

## 任务二
使用双线性插值算法计算目标图像像素点对应的值，实现了任意规格的缩放。（不要求保持长宽比）  
使用如下图片进行测试，输出结果(300,300)如下。
[mission2.b2.srcimg] (Test)  
![mission2.b2.resultimg]

#### *Reference* :
https://blog.csdn.net/wgx571859177/article/details/78963267  
https://blog.csdn.net/pentiumCM/article/details/104720100  
https://www.cnblogs.com/jymg/p/11654554.html  

## 任务三
查阅大量资料后，确定流程如下：  
gamma矫正 -> BGR -> HSV -> (*inRange*)BinaryImage ->闭运算 -> 腐蚀 -> 膨胀 -> 检测轮廓 -> ***通过矩形度测试与长宽比确定合法轮廓（如不存在合法轮廓，则放宽标准并重复上述过程）*** -> 旋转至正确方向并切割 -> 输出  

在网上下载了10张车辆照片与原来2张一同测试，经过测试即便是 **有遮挡，目标小** 的车牌也能够基本准确地识别出相应区域。
使用如下图片进行测试，输出结果如下。  
[mission2.b3.srcimg1] (Test)
[mission2.b3.srcimg2] (Test)  
[mission2.b3.srcimg3] (Test)
[mission2.b3.srcimg4] (Test)  
[mission2.b3.srcimg5] (Test)
[mission2.b3.srcimg6] (Test)  
[mission2.b3.srcimg7] (Test)
[mission2.b3.srcimg8] (Test)  
[mission2.b3.srcimg9] (Test)
[mission2.b3.srcimg10] (Test)  
[mission2.b3.srcimg11] (Test)
[mission2.b3.srcimg12] (Test)  
[mission2.b3.resultimg1] (Result)
[mission2.b3.resultimg2] (Result)
[mission2.b3.resultimg3] (Result)
[mission2.b3.resultimg4] (Result)
[mission2.b3.resultimg5] (Result)
[mission2.b3.resultimg6] (Result)
[mission2.b3.resultimg7] (Result)
[mission2.b3.resultimg8] (Result)
[mission2.b3.resultimg9] (Result)
[mission2.b3.resultimg10] (Result)
[mission2.b3.resultimg11] (Result)
[mission2.b3.resultimg12] (Result)


[mission2.b1.srcimg]:  ./Mission2.b1/srcimg/car3.png
[mission2.b1.resultimg]:  ./Mission2.b1/resultimg/b1.1.png
[mission2.b2.srcimg]:  ./Mission2.b1/srcimg/car3.png
[mission2.b2.resultimg]:  ./Mission2.b2/resultimg/b2.1.png
[mission2.b3.srcimg1]:  ./Mission2.b3/srcimg/1.jpg
[mission2.b3.srcimg2]:  ./Mission2.b3/srcimg/2.jpg
[mission2.b3.srcimg3]:  ./Mission2.b3/srcimg/3.jpg
[mission2.b3.srcimg4]:  ./Mission2.b3/srcimg/4.jpg
[mission2.b3.srcimg5]:  ./Mission2.b3/srcimg/5.jpg
[mission2.b3.srcimg6]:  ./Mission2.b3/srcimg/6.jpg
[mission2.b3.srcimg7]:  ./Mission2.b3/srcimg/7.jpg
[mission2.b3.srcimg8]:  ./Mission2.b3/srcimg/8.jpg
[mission2.b3.srcimg9]:  ./Mission2.b3/srcimg/9.jpg
[mission2.b3.srcimg10]:  ./Mission2.b3/srcimg/10.jpg
[mission2.b3.srcimg11]:  ./Mission2.b3/srcimg/car5.jpg
[mission2.b3.srcimg12]:  ./Mission2.b3/srcimg/car3.png
[mission2.b3.resultimg1]:  ./Mission2.b3/resultimg/1.jpg
[mission2.b3.resultimg2]:  ./Mission2.b3/resultimg/2.jpg
[mission2.b3.resultimg3]:  ./Mission2.b3/resultimg/3.jpg
[mission2.b3.resultimg4]:  ./Mission2.b3/resultimg/4.jpg
[mission2.b3.resultimg5]:  ./Mission2.b3/resultimg/5.jpg
[mission2.b3.resultimg6]:  ./Mission2.b3/resultimg/6.jpg
[mission2.b3.resultimg7]:  ./Mission2.b3/resultimg/7.jpg
[mission2.b3.resultimg8]:  ./Mission2.b3/resultimg/8.jpg
[mission2.b3.resultimg9]:  ./Mission2.b3/resultimg/9.jpg
[mission2.b3.resultimg10]:  ./Mission2.b3/resultimg/10.jpg
[mission2.b3.resultimg11]:  ./Mission2.b3/resultimg/car5.jpg
[mission2.b3.resultimg12]:  ./Mission2.b3/resultimg/car3.png
