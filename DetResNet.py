import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

class ResNetFP(nn.Module):
    def __init__(self,
                 block=50,
                 mode='fe',
                 input_shape=None,
                 output_channels=None,
                 output_shape=None,
                 pretrained=True):
        """
        This is the same architecture of ResNet, but you can acquire hidden layer as outputs.
        :param block: int, the number of residual blocks, {18, 34, 50, 101, 152}
        :param mode: fe -> feature extract, conv -> image output, fc -> fully connected
        :param input_shape: list, the shape of input tensor, size->[channels, height, width]
        :param output_channels: int, if you want to use the 'conv' or 'fc' mode, please set the number of output channels.
        :param output_shape: option, list, you are able to set the output size, [height, width]
        :param pretrained: bool, using the pretrained weights with ImageNet or not.
        """
        super(ResNetFP, self).__init__()
        # resnet model
        if input_shape is None:
            input_shape = [3, 224, 224]
        if output_shape is None:
            output_shape = [1, 28, 28]
        base_resnet = select_resnet(block=block, pretrained=pretrained)
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((output_shape[1], output_shape[2]))
        self.fe_out_shape = self._get_conv_output(input_shape, True)
        if mode == 'fc' or mode == 'origin':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((output_shape[1], output_shape[2]))
        if mode == 'fe':
            self.additional_layer = nn.Sequential()
        elif mode == 'conv':
            if output_channels is None:
                print(pycolor.Color.RED
                      + "[Residual Block Error] "
                        "If you want to use the conv mode, please set the input of output_channels."
                      + pycolor.Color.RESET)
                sys.exit(-1)

            self.additional_layer = nn.Sequential(
                nn.Conv2d(in_channels=self.fe_out_shape[0],
                          out_channels=output_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
                nn.ReLU(inplace=True)
            )
        elif mode == 'fc':
            if output_channels is None:
                print(pycolor.Color.RED
                      + "[Residual Block Error] "
                        "If you want to use the fc mode, please set the input of output_channels."
                      + pycolor.Color.RESET)
                sys.exit(-1)
            self.additional_layer = nn.Sequential(
                Flatten(),
                nn.Linear(self.fe_out_shape[0], output_channels)
            )
        elif mode == 'origin':
            self.additional_layer = nn.Sequential(
                Flatten(),
                base_resnet.fc
            )
        else:
            print(pycolor.Color.RED
                  + "[Residual Block Error] "
                    "Please select the valid mode. {feature extract: fe, convolution: conv, fully connected: fc}"
                  + pycolor.Color.RESET)
            sys.exit(-1)
        self.fe_out_shape = self._get_conv_output(input_shape, False)  # this is for saving changed output shape

    def _get_conv_output(self, shape, first_flag):
        bs = 1
        input_ = Variable(torch.rand(bs, *shape))
        x = self.conv1(input_)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if not first_flag:
            x = self.additional_layer(x)
        return x.size()[1:]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        x = self.avgpool(l4)
        x = self.additional_layer(x)
        return x, l1, l2, l3, l4


class DetResNet50(nn.Module):
    """ This class is available for pretrained model of Detection Task. (50 blocks only.) """
    def __init__(self, 
                 input_shape=None,
                 output_channels=None,
                 output_shape=None,
                 resnet_model_name='faster_rcnn', 
                 mode='fe',
                 pretrained=True):
        super(DetResNet50, self).__init__()
        if input_shape is None:
            input_shape = [3, 224, 224]
        if output_shape is None:
            output_shape = [1, 28, 28]

        self.det_resnet_dict = {
            'faster_rcnn': torchvision.models.detection.fasterrcnn_resnet50_fpn,
            'mask_rcnn': torchvision.models.detection.maskrcnn_resnet50_fpn,
            'keypoint_rcnn': torchvision.models.detection.keypointrcnn_resnet50_fpn,
            'retina_net': torchvision.models.detection.retinanet_resnet50_fpn
        }
        base_resnet = self._select_resnet(model_name=resnet_model_name, pretrained=pretrained)
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((output_shape[1], output_shape[2]))
        self.fe_out_shape = self._get_conv_output(input_shape, True)
        if mode == 'fc' or mode == 'origin':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((output_shape[1], output_shape[2]))
        if mode == 'fe':
            self.additional_layer = nn.Sequential()
        elif mode == 'conv':
            if output_channels is None:
                print(pycolor.Color.RED
                      + "[Residual Block Error] "
                        "If you want to use the conv mode, please set the input of output_channels."
                      + pycolor.Color.RESET)
                sys.exit(-1)

            self.additional_layer = nn.Sequential(
                nn.Conv2d(in_channels=self.fe_out_shape[0],
                          out_channels=output_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0),
                nn.ReLU(inplace=True)
            )
        elif mode == 'fc':
            if output_channels is None:
                print(pycolor.Color.RED
                      + "[Residual Block Error] "
                        "If you want to use the fc mode, please set the input of output_channels."
                      + pycolor.Color.RESET)
                sys.exit(-1)
            self.additional_layer = nn.Sequential(
                Flatten(),
                nn.Linear(self.fe_out_shape[0], output_channels)
            )
        elif mode == 'origin':
            self.additional_layer = nn.Sequential(
                Flatten(),
                base_resnet.fc
            )
        else:
            print(pycolor.Color.RED
                  + "[Residual Block Error] "
                    "Please select the valid mode. {feature extract: fe, convolution: conv, fully connected: fc}"
                  + pycolor.Color.RESET)
            sys.exit(-1)
        self.fe_out_shape = self._get_conv_output(input_shape, False)  # this is for saving changed output shape

    # select resnet model
    def _select_resnet(self, model_name, pretrained):
        # load model
        if model_name in self.det_resnet_dict:
            base_model = self.det_resnet_dict[model_name]
            base_model = base_model(pretrained=pretrained).backbone.body
        else:
            print("[ResNet Model Name Error] This model name is unsupported.")
            sys.exit(-1)

        return base_model
    
    def _get_conv_output(self, shape, first_flag):
        bs = 1
        input_ = Variable(torch.rand(bs, *shape))
        x = self.conv1(input_)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if not first_flag:
            x = self.additional_layer(x)
        return x.size()[1:]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        x = self.avgpool(l4)
        x = self.additional_layer(x)
        return x, l1, l2, l3, l4


if __name__ == '__main__':
    input_shape = (3, 224, 224)
    model = DetResNet50(input_shape=input_shape, resnet_model_name='mask_rcnn')
    t = torch.rand(1, *input_shape)
    x, l1, l2, l3, l4 = model(t)
    print(x.size(), l1.size(), l2.size(), l3.size(), l4.size())
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    resnet = resnet_fpn_backbone('resnet18', pretrained=True)
    print(resnet)