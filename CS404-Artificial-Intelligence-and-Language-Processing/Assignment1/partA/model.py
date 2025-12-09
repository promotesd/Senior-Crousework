import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    cfgs = {
        'S': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features=features
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))

        self.classify=nn.Sequential(
            nn.Linear(in_features=512*1*1, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classify(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(cfg, in_channels=3, batch_norm=True):
        layers = []
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels=in_channels,
                                   out_channels=v,
                                   kernel_size=3,
                                   padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    @classmethod
    def vgg11(cls, num_classes=10, batch_norm=True, in_channels=3):
        cfg = cls.cfgs['A']
        features = cls.make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
        model = cls(features, num_classes=num_classes, init_weights=True)
        return model

    @classmethod
    def vgg16(cls, num_classes=10, batch_norm=True, in_channels=3):
        cfg = cls.cfgs['D']
        features = cls.make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
        model = cls(features, num_classes=num_classes, init_weights=True)
        return model
    
    @classmethod
    def SimpleVGG(cls, num_classes=10, batch_norm=True, in_channels=3):
        cfg = cls.cfgs['S']
        features = cls.make_layers(cfg, in_channels=in_channels, batch_norm=batch_norm)
        model = cls(features, num_classes=num_classes, init_weights=True)
        return model


##implement Res_X(X>=50)
class BottleNeck(nn.Module):
    expansion=4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=1, stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

        self.conv3=nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1 ,bias=False)
        self.bn3=nn.BatchNorm2d(out_channel*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self, x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out+=identity
        out=self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes, include_top=True, input_channel=3):
        super(ResNet, self).__init__()
        self.include_top=include_top
        self.num_classes=num_classes
        self.in_channel=64
        self.conv1=nn.Conv2d(in_channels=input_channel, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False )
        self.bn1=nn.BatchNorm2d(num_features=self.in_channel)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1=self._makelayer(block, 64, block_num[0])
        self.layer2=self._makelayer(block, 128, block_num[1],stride=2)
        self.layer3=self._makelayer(block, 256, block_num[2],stride=2)
        self.layer4=self._makelayer(block, 512, block_num[3],stride=2)

        if self.include_top:
            self.avgpool=nn.AdaptiveAvgPool2d((1,1))
            self.fc=nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
        

    def _makelayer(self, block, channel, block_num, stride=1):
        if stride!=1 or self.in_channel!=channel*block.expansion:
            dowmsample=nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion,kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        layers=[]
        layers.append(block(self.in_channel, channel, downsample=dowmsample, stride=stride))
        self.in_channel=channel*block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        if self.include_top:
            x=self.avgpool(x)
            x=torch.flatten(x, 1)
            x=self.fc(x)
        return x
    
# if __name__ == "__main__":

#     x_gray = torch.randn(2, 1, 224, 224)  # batch=2, C=1, H=W=224
#     vgg_gray = VGG.vgg16(num_classes=10, batch_norm=True, in_channels=1)
#     out_vgg_gray = vgg_gray(x_gray)
#     print("VGG gray output shape:", out_vgg_gray.shape)   # [2, 10]

#     x_gray = torch.randn(2, 1, 224, 224)
#     resnet_gray = ResNet(
#         block=BottleNeck,
#         block_num=[3, 4, 6, 3],   # ResNet50 配置
#         num_classes=10,
#         include_top=True,
#         input_channel=1           # 灰度图 → 1 通道
#     )
#     out_resnet_gray = resnet_gray(x_gray)
#     print("ResNet gray output shape:", out_resnet_gray.shape)   #[2, 10]

#     x_rgb = torch.randn(2, 3, 224, 224)
#     vgg_rgb = VGG.vgg16(num_classes=10, batch_norm=True, in_channels=3)
#     out_vgg_rgb = vgg_rgb(x_rgb)
#     print("VGG rgb output shape:", out_vgg_rgb.shape)   #[2, 10]

#     x_rgb = torch.randn(2, 3, 224, 224)
#     resnet_rgb = ResNet(
#         block=BottleNeck,
#         block_num=[3, 4, 6, 3],   # ResNet50
#         num_classes=10,
#         include_top=True,
#         input_channel=3
#     )
#     out_resnet_rgb = resnet_rgb(x_rgb)
#     print("ResNet rgb output shape:", out_resnet_rgb.shape)   #[2, 10]
