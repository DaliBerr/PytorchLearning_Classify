from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_path,label_path,image_path,transformer=None):
        self.root_path = root_path
        self.image_root_path = os.path.join(root_path,image_path)
        self.label_root_path = os.path.join(root_path,label_path)

        self.image_path = os.listdir(self.image_root_path)
        self.label_path = os.listdir(self.label_root_path)
        self.transformer = transformer

    def __getitem__(self, index):
        image_name = self.image_path[index]
        image_path = os.path.join(self.image_root_path,image_name)
        img = Image.open(image_path)
        if self.transformer is not None:
            img = self.transformer(img)

        label_name = self.label_path[index]
        label_path = os.path.join(self.label_root_path, label_name)
        label = open(label_path).read()
        return  img,label
    def __len__(self):
        return len(self.image_path)



# root_path = "J:\\Download\\DataSet\\train"
# label_path = "ants_label"
# image_path = "ants_image"


# b = MyData(root_path,label_path,image_path)
# c = b.__getitem__(0)
# print(b.image_path[0])
# print(c)
# print(b.__len__())