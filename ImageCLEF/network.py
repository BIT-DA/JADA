import torch.nn as nn
import torchvision
from torch.autograd import Function
import torch
import torch.nn.functional as F


class ReverseLayerF(Function):
    r"""Gradient Reverse Layer(Unsupervised Domain Adaptation by Backpropagation)
    Definition: During the forward propagation, GRL acts as an identity transform. During the back propagation though,
    GRL takes the gradient from the subsequent level, multiplies it by -alpha  and pass it to the preceding layer.

    Args:
        x (Tensor): the input tensor
        alpha (float): \alpha =  \frac{2}{1+\exp^{-\gamma \cdot p}}-1 (\gamma =10)
        out (Tensor): the same output tensor as x
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ResBase(nn.Module):
    r"""Constructs a feature extractor based on ResNet-50 model.
    remove the last layer, replace with bottleneck layer(out_dim=256)

    1. extract the discriminative feature (minimize the label prediction loss)
    2. extract the domain-invariance feature to confuse the domain classifier (maximize domain loss)
    3. learn to generate target features near the support to fool the classifiers (minimize discrepancy loss)
    """

    def __init__(self):
        super(ResBase, self).__init__()
        model_res50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_res50.conv1
        self.bn1 = model_res50.bn1
        self.relu = model_res50.relu
        self.maxpool = model_res50.maxpool
        self.layer1 = model_res50.layer1
        self.layer2 = model_res50.layer2
        self.layer3 = model_res50.layer3
        self.layer4 = model_res50.layer4
        self.avgpool = model_res50.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.bottleneck = nn.Linear(model_res50.fc.in_features, 256)
        # init bottleneck parameters
        nn.init.normal_(self.bottleneck.weight.data, 0, 0.005)
        nn.init.constant_(self.bottleneck.bias.data, 0.1)

        self.__in_features = 256

    def forward(self, x):
        """
        :param x: the input Tensor as [bs, 3, 224, 224]
        :return: 256-dim feature
        """
        feature = self.feature_layers(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bottleneck(feature)
        return feature

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        parameter_list = [{'params': self.feature_layers.parameters(), 'lr_mult': 1, 'decay_mult': 2},
                          {'params': self.bottleneck.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
        return parameter_list

    def output_num(self):
        return self.__in_features


class Domain(nn.Module):
    r"""Long domain classifier
        connect to tne feature extractor via a gradient reverse layer that multiplies by
        a certain negative constant during the backpropagation-based training

        Distinguish the features as a source or target (minimize domain loss)

        Args:
            in_features: size of input layer unit, default: 256
            hidden_size: size of hidden layer unit, default: 1024
        """

    def __init__(self, in_features=256, hidden_size=1024):
        super(Domain, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.__in_features = 1
        self.init_weight()

    def forward(self, x, alpha):
        r"""flip all the samples' sign of gradients when back-propagation
        :param x: the input Tensor as [bs, features_dim]
        :param alpha: ratio
        :return: the domain label prediction(1 dimension and use BCEloss)
        """
        x = ReverseLayerF.apply(x, alpha)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def init_weight(self):
        nn.init.normal_(self.fc1.weight.data, 0, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 0.3)

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

    def output_num(self):
        return self.__in_features


class Classifier(nn.Module):
    r"""Task-specific Classifier & label predictor
    utilize the task-specific classifier as discriminators that try to detect target samples
    that are far from the support of the source.

    1. predict class labels (source, minimize prediction loss)
    2. align distributions (maximize discrepancy loss)

    Args:
        in_features: size of input layer unit, default: 256
        hidden_size: size of hidden layer unit, default: 128
        class_num: number of categories, default: 31(office-31)
    """

    def __init__(self, in_features=256, hidden_size=128, class_num=31):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, class_num)
        )
        # self.fc1 = nn.Linear(in_features, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, class_num)
        # self.relu1 = nn.ReLU(inplace=True)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        # self.init_weight()

    def forward(self, x, alpha):
        r"""flip the target samples' sign of gradients when back-propagation
        :param x: the input Tensor as [bs, features_dim]
        :param alpha: ratio
        :return: the class label prediction
        """
        xs = x[:x.size(0) // 2, :]
        xt = x[x.size(0) // 2:, :]
        xt = ReverseLayerF.apply(xt, alpha)
        x = torch.cat((xs, xt), 0)
        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        # x = self.fc3(x)
        x = self.classifier(x)
        return x

    def init_weight(self):
        nn.init.normal_(self.fc1.weight.data, 0, 0.01)
        nn.init.normal_(self.fc2.weight.data, 0, 0.01)
        nn.init.normal_(self.fc3.weight.data, 0, 0.3)

    def get_parameters(self):
        r"""get the different parameters of each layer"""
        return [{'params': self.parameters(), 'lr_mult': 10, 'decay_mult': 2}]

