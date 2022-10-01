
from torchvision.transforms import transforms
from NN import *
from Dataset import MyData
from torch.utils.data import DataLoader
from Tensorboard import writer
import torch

print(torch.cuda.is_available())
root_path = "J:\\Download\\DataSet\\train"
#pretrans_root_path = "J:\\Download\\DataSet\\train\\PreTransformer"
Ants_Image_Path = "ants_image"
Ants_Label_Path = "ants_label"
Bees_Image_Path = "bees_image"
Bees_Label_Path = "bees_label"

transform =  transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
])

Ants_DataSet = MyData(root_path, Ants_Label_Path, Ants_Image_Path, transformer=transform)
Bees_DataSet = MyData(root_path=root_path,label_path=Bees_Label_Path,image_path=Bees_Image_Path,transformer=transform)
model = Model()
DataSet = Ants_DataSet + Bees_DataSet
#Trans(root_path,Label_path,Image_path,pretrans_root_path)

Data = DataLoader(dataset=DataSet, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
len = Data.__len__() * Data.batch_size
print(len)
loss_fn = nn.CrossEntropyLoss()
#5.优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

total_train_step = 0
epoch = 100


for i in range(epoch):
    print("—————————————第{}轮训练开始————————————".format(i + 1))
    model.train()
    for data in Data:
        img,label = data
        #print(label[0])
        if label[0] =='ants':
            label = torch.Tensor([1,0])

        else:
            label = torch.Tensor([0,1])
        output = model(img)
       # print(output)
        #print(label)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    torch.save(model, "Model{}.pth".format(i))
# for i in range(len):
#     img,label = Data.dataset.__getitem__(i)


#     writer.add_image(str(i),img,2)
#     writer.close()

