#------------------------------------------------------------------------
# EfficientNet 
# pip install git+https://github.com/shijianjian/EfficientNet-PyTorch-3D
#-------------------------------------------------------------------------

from efficientnet_pytorch_3d import EfficientNet3D
import torch
from torchsummary import summary

model = EfficientNet3D.from_name(
    "efficientnet-b0", 
    override_params={'num_classes': 2}, 
    in_channels=1
    )

summary(model, input_size=(1, 200, 200, 200))

model = model.cuda()
inputs = torch.randn((1, 1, 200, 200, 200)).cuda()
labels = torch.tensor([0]).cuda()
# test forward
num_classes = 2
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model.train()
for epoch in range(2):
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    print('[%d] loss: %.3f' % (epoch + 1, loss.item()))

print('Finished Training')

#-------------------------------------------
# ResNet50 from PyTorch
#-------------------------------------------
import torch
model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)

#-------------------------------------------
# ResNet3d18
#------------------------------------------
import torchvision.models as models
torchvision.models.video.r3d_18(pretrained: bool = False, progress: bool = True, **kwargs: Any)
model = models.video.r3d_18()



