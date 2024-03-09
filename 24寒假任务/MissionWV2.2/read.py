import os
import cv2
import numpy as np

# 图片和标注所在的文件夹路径
img_folder = "D:\Desktop\py-program\RoboMaster_YOLO\data\images\\train"
txt_folder = "D:\Desktop\py-program\RoboMaster_YOLO\data\labels\\train"

# 获取图片文件名列表
img_files = sorted(os.listdir(img_folder))

# 遍历图片文件名列表
for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)
    txt_path = os.path.join(txt_folder, img_file[:-4] + ".txt")

    # 载入图像
    img = cv2.imread(img_path)

    # 读取对应的标注文件
    with open(txt_path, "r") as file:
        lines = file.read().splitlines()

# 对每一行标注（即每个目标）进行处理
    for line in lines:
        label, x1, y1, width, height = map(float, line.split())
        # 因为x1,y1,width,height是相对于图片宽高的比例值，我们要将它们转换为像素值
        x1, y1, width, height = x1 * img.shape[1], y1 * img.shape[0], width * img.shape[1], height * img.shape[0]
        # 将浮点数转换为整数
        x1, y1, width, height = int(x1), int(y1), int(width), int(height)
        # 计算矩形的左上角和右下角坐标，注意这里要把矩形中心坐标转成左上角坐标
        p1, p2 = (x1-width//2, y1-height//2), (x1+width//2, y1+height//2)
        # 在原图上画出矩形
        cv2.rectangle(img, p1, p2, (255, 0, 0), 2)
        # 在矩形旁边添加类别标签
        cv2.putText(img, str(int(label)), (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # 显示图像
    cv2.imshow('Image', img)
    # 等待1ms，如果在这期间用户按下'q'键，则跳出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭所有OpenCV窗口
cv2.destroyAllWindows()

