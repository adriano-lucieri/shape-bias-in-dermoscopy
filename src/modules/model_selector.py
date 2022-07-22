from torchvision import models

import torch.nn as nn
import os

base_pretrained_path = '../../Tmp/PyTorch_Pretrained/'


def freeze_base(model):
    """Fine-tune only head of the model"""
    for param in model.parameters():
        param.requires_grad = False

def model_selector(model_name, num_classes, pretrained=True, freeze_base=False, num_channels=3):
    """
    Model Input Sizes are all 224x224 except for InceptionV3 that expects 299x299
    (https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)

    :param model_name:
    :param num_classes:
    :param pretrained:
    :param freeze_base:
    :return:
    """

    os.environ['TORCH_HOME'] = os.path.join(base_pretrained_path, model_name)

    if model_name == 'AlexNet':
        model = models.alexnet(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))

    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'ResNet34':
        model = models.resnet34(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'ResNet101':
        model = models.resnet101(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'ResNet152':
        model = models.resnet152(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'SENet_10':
        model = models.squeezenet1_0(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))

    elif model_name == 'SENet_11':
        model = models.squeezenet1_1(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))

    elif model_name == 'VGG11':
        model = models.vgg11(pretrained=pretrained)
    elif model_name == 'VGG11_BN':
        model = models.vgg11_bn(pretrained=pretrained)
    elif model_name == 'VGG13':
        model = models.vgg13(pretrained=pretrained)
    elif model_name == 'VGG13_BN':
        model = models.vgg13_bn(pretrained=pretrained)
    elif model_name == 'VGG16':
        model = models.vgg16(pretrained=pretrained)
    elif model_name == 'VGG16_BN':
        model = models.vgg16_bn(pretrained=pretrained)
    elif model_name == 'VGG19':
        model = models.vgg19(pretrained=pretrained)
    elif model_name == 'VGG19_BN':
        model = models.vgg19_bn(pretrained=pretrained)

    elif model_name == 'DenseNet121':
        model = models.densenet121(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == 'DenseNet161':
        model = models.densenet161(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == 'DenseNet169':
        model = models.densenet169(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == 'DenseNet201':
        model = models.densenet201(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        
        if num_channels != 3:
            model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == 'InceptionV3':
        model = models.inception_v3(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.fc = nn.Linear(2048, num_classes)

    elif model_name == 'GoogLeNet':
        model = models.googlenet(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.fc = nn.Linear(1024, num_classes)

    elif model_name == 'ShuffleNet':
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'MobileNet':
        model = models.mobilenet_v2(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1280, num_classes))
        #model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    elif model_name == 'ResNeXt50':
        model = models.resnext50_32x4d(pretrained=pretrained)
    elif model_name == 'ResNeXt101':
        model = models.resnext101_32x8d(pretrained=pretrained)

    elif model_name == 'WideResNet50':
        model = models.wide_resnet50_2(pretrained=pretrained)
    elif model_name == 'WideResNet101':
        model = models.wide_resnet101_2(pretrained=pretrained)

    elif model_name == 'mNASNet':
        model = models.mnasnet1_0(pretrained=pretrained)
    elif model_name == 'mobilenetV2':
        model = models.mobilenet_v2(pretrained=pretrained)
        if freeze_base: freeze_base(model)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(1280, num_classes))
    return model