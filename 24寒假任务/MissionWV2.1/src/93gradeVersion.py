import torch
import torchvision
import torch.nn as nn
import numpy as np
import cv2
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt

epochs = 2
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)


# invert color of the dataset
class RandomInvert(object):
    def __init__(self, p=0.5):
        super(RandomInvert, self).__init__()
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return ImageOps.invert(img)
        return img

# draw lines on the dataset 
class RandomDrawLines(object):
    def __init__(self, max_lines=5, line_color=255, width=1, p=0.5):
        self.max_lines = max_lines
        self.line_color = line_color
        self.width = width
        self.p = p
      
    def __call__(self, img):
        # PIL Image
        if np.random.rand() < self.p:
            draw = ImageDraw.Draw(img)
            width, height = img.size
            for _ in range(np.random.randint(0, self.max_lines+1)):
                start = np.random.randint(0, width), np.random.randint(0, height)
                end = np.random.randint(0, width), np.random.randint(0, height)
                draw.line([start, end], fill=self.line_color, width=self.width)
            del draw
        return img

# erose & dilation onto the dataset
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

# Combine augment
transform_augmentation = transforms.Compose([
    RandomErosionDilation(p=0.3, erosion_size=(3, 3), dilation_size=(3, 3)),
    RandomDrawLines(max_lines=5, line_color=1, p=0.5),
    RandomInvert(p=0.5),
    transforms.ToTensor(),
    # random rotate & translate
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # random gaussianBlur
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2))], p=0.5),
    transforms.Normalize((0.1307,), (0.3081,))
])
  
# Data loader setup
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                                transform=transform_augmentation),
    batch_size=batch_size_train, shuffle=True
)
dataiter = iter(train_loader)

# Load the 1st image batch to show preprocess output
images, labels = next(dataiter)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
imshow(torchvision.utils.make_grid(images))

# Data for test loader setup
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

# Residual Block
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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        return out
    
# Below are old blocks and net
# 构建包含ResidualBlock的CNN
# class ResNet_CNN(nn.Module):
#     def __init__(self,num_classes=10):
#         super(ResNet_CNN,self).__init__()

#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.relu = nn.ReLU()
#         self.bn1 = nn.BatchNorm2d(64)
#         self.res2 = ResidualBlock(64,64)

#         # # 用一个自适应均值池化层将每个通道维度变成1*1，此句可选
#         # self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(64 * ResidualBlock.expansion, num_classes)


#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.res2(x)
#         # n个通道，每个通道1*1，输出n*1*1
#         x = self.avg_pool(x)
#         # maybe not need

#         # 将数据拉成一维
#         x = x.view(x.size(0),-1)
#         x = self.fc(x)
#         return x
    

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
#model = ResNet_CNN().to(device)
num_blocks = [2, 2, 2, 2]  # change it to add more block
model = ResNet_CNN(ResidualBlock, num_blocks).to(device)

loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
# train
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
    # torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += loss_f(output, label).item()
            predict = output.argmax(dim=1)
            # Calc proper percentage
            total += label.size(0)
            correct += (predict == label).sum().item()
        # loss & accuracy
        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss/len(loader), 100*(correct/total)))

if __name__ == '__main__':
    print(model)
    #epochs=10
    train(epochs, train_loader)
    test(test_loader)

    example_input = torch.rand(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, "traced_model.pt")
