import torch
from torchvision import transforms
from NN import Model
from PIL import Image
img_path = "J:/Download/DataSet/val/bees/586474709_ae436da045.jpg"
import torchvision
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# testdataset = torchvision.datasets.CIFAR10(root="./TestCIFAR10",train=True,download=False)
# print(testdataset[0])

model = torch.load("Model30.pth")
img = Image.open(img_path)
trans =  transforms.Compose([
        transforms.Resize((64,64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
])
img = trans(img)
model.eval()
with torch.no_grad():
        out = model(img)
print(out)
