import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms

import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from torchvision.transforms import functional as F
from scipy.ndimage import gaussian_filter,map_coordinates
from numpy import random

epochs = 2
train_mode = 'pretrain'
pretrain_name = '99gradeModelTrain20epoch.pt'
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
weight_decay = 1e-5
log_interval = 10
random_seed = 5
torch.manual_seed(random_seed)


# 自定义随机颜色翻转转换
class RandomInvert(object):
    def __init__(self, p=0.5):
        super(RandomInvert, self).__init__()
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return ImageOps.invert(img)
        return img

# 自定义画线转换
class RandomDrawLines(object):
    def __init__(self, max_lines=5, line_color=255, width=1, p=0.5):
        self.max_lines = max_lines
        self.line_color = line_color
        self.width = width
        self.p = p
      
    def __call__(self, img):
        # 操作PIL图像
        if np.random.rand() < self.p:
            draw = ImageDraw.Draw(img)
            width, height = img.size
            for _ in range(np.random.randint(0, self.max_lines+1)):
                start = np.random.randint(0, width), np.random.randint(0, height)
                end = np.random.randint(0, width), np.random.randint(0, height)
                draw.line([start, end], fill=self.line_color, width=self.width)
            del draw
        return img

class RandomErosionDilation(object):
    def __init__(self, p=0.5, erosion_size=(1, 1), dilation_size=(1, 1)):
        self.p = p
        self.erosion_size = erosion_size
        self.dilation_size = dilation_size
        
    def __call__(self, img):
        if np.random.rand() < self.p:
            operation = np.random.choice(["erosion", "dilation"])
            if operation == "erosion":
                erode_kernel = np.ones(self.erosion_size, np.uint8)
                img_np = np.array(img)
                img_np = cv2.erode(img_np, erode_kernel, iterations=1)
                img = Image.fromarray(img_np)
            else:
                dilate_kernel = np.ones(self.dilation_size, np.uint8)
                img_np = np.array(img)
                img_np = cv2.dilate(img_np, dilate_kernel, iterations=1)
                img = Image.fromarray(img_np)
        return img

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=0.5):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0., 1.)
        return tensor

class ElasticTransformation(object):
    def __init__(self, alpha=1, sigma=0.2, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image = np.array(image)
            shape = image.shape
            dx = gaussian_filter((random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            dy = gaussian_filter((random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
            image = map_coordinates(image, indices, order=1).reshape(shape)
            image = Image.fromarray(image)
        return image
    
class RandomErasing(object):
    """
    随机擦除（遮盖）图像中的矩形区域。

    参数:
    p: 执行随机擦除的概率。
    scale: 擦除区域占原始图像面积的比例范围。
    ratio: 擦除区域的宽高比范围。
    value: 擦除区域的填充值，可以是 'random'、单一的数值或元组来指定随机值的范围。
    """
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        
        img = np.array(img, dtype=np.float32)
        
        h, w = img.shape[:2]
        area = h * w
        
        target_area = random.uniform(*self.scale) * area
        aspect_ratio = random.uniform(*self.ratio)

        erase_w = int(round(np.sqrt(target_area * aspect_ratio)))
        erase_h = int(round(np.sqrt(target_area / aspect_ratio)))

        # 确保擦除区域的尺寸不超出图像尺寸
        erase_w = min(erase_w, w-1)
        erase_h = min(erase_h, h-1)

        x = random.randint(0, h - erase_h)
        y = random.randint(0, w - erase_w)
        
        if self.value == 'random':
            fill_value = np.random.rand(erase_h, erase_w) * 255
        elif isinstance(self.value, tuple):
            fill_value = np.random.uniform(self.value[0], self.value[1], (erase_h, erase_w))
        else:
            fill_value = self.value
        
        img[x:x+erase_h, y:y+erase_w] = fill_value

        return Image.fromarray(img.astype(np.uint8), mode='L')

# 组合所有转换
transform_augmentation = transforms.Compose([
    RandomErosionDilation(p=0.3, erosion_size=(2, 2), dilation_size=(3, 3)),
    # 加入自定义画线转换
    RandomDrawLines(max_lines=5, line_color=1, p=0.5),
    #RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # 添加自定义的随机遮挡
    # 首先对图像进行随机颜色翻转
    RandomInvert(p=0.5),
    ElasticTransformation(alpha=2, sigma=0.25, p=0.3),  # 添加自定义的弹性变形
    transforms.ToTensor(),
    # 应用随机仿射变换 (随机拉伸、旋转、平移)
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # 应用随机高斯模糊
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2))], p=0.5),
    
    AddGaussianNoise(mean=0., std=0.1, p=0.5),  # 添加自定义的高斯噪声
    
    # 标准化图像数据
    transforms.Normalize((0.1307,), (0.3081,))
])
  
# 使用增强的转换重新定义数据加载器
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                                transform=transform_augmentation),
    batch_size=batch_size_train, shuffle=True
)
dataiter = iter(train_loader)

# 加载第一张图片和标签
images, labels = next(dataiter)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize 如果有需要的话，您可能需要根据使用的预处理步骤进行调整
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 显示图片
imshow(torchvision.utils.make_grid(images))


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

# 残差模块
class ResidualBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        # 保存输入数据，采用恒等映射
        residual = x
        #
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        # 返回结果
        return out
    
class ResNet_CNN(nn.Module):
    @staticmethod
    def make_layer(block, in_channels, block_channels, num_blocks, stride=1):
        downsample = None
        layers = []

        # 仅当stride不为1，或输入输出通道数不等时，使用downsample来匹配维度
        if stride != 1 or in_channels != block_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, block_channels * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block_channels * block.expansion),
            )
        
        # 第一个block可能需要downsample
        layers.append(block(in_channels, block_channels, stride, downsample))

        # 后续blocks的输入通道数已经是扩展之后的数量
        for _ in range(1, num_blocks):
            layers.append(block(block_channels * block.expansion, block_channels))

        return nn.Sequential(*layers)
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CNN, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, self.in_channels, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64 * block.expansion, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 128 * block.expansion, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 256 * block.expansion, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_blocks = [2, 2, 2, 2]  # 可以更改这些值以增加更多的残差块

if(train_mode=='pretrain'):
    model = torch.jit.load(pretrain_name).to(device)
elif(train_mode=='newtrain'):
    model = ResNet_CNN(ResidualBlock, num_blocks).to(device)

loss_f = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
# 训练模型
def train(epochs, loader):
    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0.0
        for batch_idx,(data,target) in enumerate(loader):
            data,target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_f(output,target)

            predict = output.argmax(dim=1)
            total += target.size(0)
            correct += (predict == target).sum().item()

            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}\taccuracy: {:.6f}%'.format(
                    epoch,batch_idx*len(data),len(loader.dataset),
                    100.*batch_idx/len(loader),loss.item(),100*(correct/total)
                ))

def test(loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    total = 0
    #torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += loss_f(output, label).item()
            predict = output.argmax(dim=1)
            #计算正确数量
            total += label.size(0)
            correct += (predict == label).sum().item()
        #计算损失值
        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/len(loader), 100*(correct/total)))

if __name__ == '__main__':
    print(model)
    #epochs=10
    train(epochs, train_loader)
    test(test_loader)

# 保存模型 state_dict()是一个字典，保存了网络中所有的参数
    # 转换并保存为torch.jit的模型
    example_input = torch.rand(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, "traced_model.pt")
