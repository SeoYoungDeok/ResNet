import torch
from src.model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

x = torch.rand(size=(1, 3, 224, 224))

model18 = ResNet18()
model34 = ResNet34()
model50 = ResNet50()
model101 = ResNet101()
model152 = ResNet152()

predict = model152(x)
print(predict)
print(predict.shape)
