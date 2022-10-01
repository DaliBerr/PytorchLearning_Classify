import torch
from torch import nn
from Tensorboard import *
# class Model(Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d()
#         self.maxp1 = nn.MaxPool2d()
#         self.conv2 = nn.Conv2d()
#     def forward(self):
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(3, 3), padding=1, ),
            nn.ReLU(),
            #9*64*64
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, ),
            nn.MaxPool2d(kernel_size=2),
        )   #9*32*32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=27, kernel_size=(3, 3), padding=1),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )  #27*16*16
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=27, out_channels=27, kernel_size=(3, 3), padding=1),
            #nn.ReLU(),
            # 32*64*64
            nn.MaxPool2d(kernel_size=2),
        )#48*32*32
        self.flat = nn.Sequential(
            nn.Flatten(),
            nn.Flatten()

        )
        self.line1 = nn.Sequential(

            nn.Linear(3*64*64,84),
            nn.Linear(84,10),
            nn.Linear(10,2),
        )
        self.i = 0
    def forward(self,img):
        writer.add_image("init_Graf_4",torch.reshape(img,(3,64,64) ),self.i)
        img = self.conv1(img)
        writer.add_image("conv1_Graf_4",torch.reshape(img,(3,64,64)),self.i)
        #img = self.conv2(img)
        #writer.add_image("conv2_Graf_4", torch.reshape(img, (3,48,48)), self.i)
       # img = self.conv3(img)
        #writer.add_image("conv3_Graf_3",torch.reshape(img,(3,128,128)),self.i)
        #writer.add_image("test",img,self.i)
        self.i+=1
        writer.close()
        #print(img)
        #img = self.flat(img)
        #print(img)
        img = img.view(-1)
        img = self.line1(img)
        output = img
        return output


# Model = nn.Sequential(
#     #3*256*256
#     nn.Conv2d(in_channels=3,out_channels=16,kernel_size= (3,3),padding=1,),
#     nn.ReLU(),
#     #16*256*256
#     nn.Conv2d(in_channels=16,out_channels=16,kernel_size= (3,3),padding=1,),
#     nn.MaxPool2d(kernel_size=2),
#     #16*128*128
#     nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2),
#     #16*64*64
#     nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1),
#     nn.ReLU(),
#     #32*64*64
#     nn.MaxPool2d(kernel_size=2),
#     #32*32*32
#     nn.Flatten(),
#     nn.Linear(in_features=(32*32*32),out_features=(32*16*16)),
#     nn.Linear(in_features=(32*16*16),out_features=(32*8*8)),
#     nn.Linear(in_features=(32*8*8),out_features=(32*8)),
#     nn.Linear(in_features=(32*8),out_features=(32*4)),
#     nn.Linear(in_features=(32*4),out_features=(2))
# )

