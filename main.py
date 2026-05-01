import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
class Imageprocessor(Dataset):
    def __init__(self,root_dir_path,transformations=None):
        super().__init__()
        self.root_dir_path=root_dir_path
        self.transformations=transformations
        self.all_img_paths=[os.path.join(root_dir_path,img) for img in os.listdir(root_dir_path)]
    
    def __len__(self):
        return len(self.all_img_paths)
    def __getitem__(self,idx):
        img_path=self.all_img_paths[idx]
        img=Image.open(img_path).convert("RGB")
        if self.transformations:
            img=self.transformations(img)
        return img
        
root_dir_path = "img_align_celeba/img_align_celeba"
transformations=transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    
])
dataset=Imageprocessor(root_dir_path,transformations)

dataloader=DataLoader(dataset,batch_size=128,shuffle=True)

# Generator Network
import torch.nn as nn
import torch.optim as optim
import numpy as np
class Generator(nn.Module):
    def __init__(self,z_dim=100,img_channels=3):
        super ().__init__()
        self.model=nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(),
            
            nn.Linear(256,512),
            nn.ReLU(),
            
            nn.Linear(512,1024),
            nn.ReLU(),
            
            nn.Linear(1024,64*64*img_channels),
            nn.Tanh()
        )
    def forward(self,z):
        img=self.model(z)
        img=img.view(img.size(0),3,64,64)
        return img

gen = Generator()
z = torch.randn(16, 100)
fake = gen(z)

#Discriminator Implementation
class Discriminator(nn.Module):
    def __init__(self,img_channels=3):
        super().__init__()
        self.model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64*img_channels,1024),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,img):
        return self.model(img)
    
GAN_loss=nn.BCELoss();
gen=Generator()
g_optimizer=optim.Adam(gen.parameters(),lr=0.0002,betas=(0.5,0.999))
dis=Discriminator()
d_optimizer=optim.Adam(dis.parameters(),lr=0.0002,betas=(0.5,0.999))

if torch.backends.mps.is_available():
    device=torch.device("mps")
elif torch.cuda.is_available():
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
print(device)
            
gen=gen.to(device)
dis=dis.to(device)
