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

def train(generator,discriminator,dataloader,epochs=10):
    for epoch in range(epochs):
        for i,img in enumerate(dataloader):
            real_img=img.to(device)
            batch_size=real_img.size(0)
            real_labels=torch.ones(batch_size,1).to(device)
            fake_labels=torch.zeros(batch_size,1).to(device)
            
            #Train the Discriminator
            d_optimizer.zero_grad()
            fake_imgs=generator(torch.randn(batch_size,100)).to(device)
            
            real_loss=GAN_loss(discriminator(real_img),real_labels)
            fake_loss=GAN_loss(discriminator(fake_imgs.detach()),fake_labels)
            d_loss=(real_loss + fake_loss)/2
            d_loss.backward()
            d_optimizer.step()
            
            #Train the Generator
            g_optimizer.zero_grad()
            g_loss=GAN_loss(discriminator(fake_imgs),real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            if i%50==0:
                print(f"for epoch {epoch+1}/{epochs} batch:{i+1} G-Loss:{g_loss}  D-Loss:{d_loss}")
            
        save_generated_images(generator,epoch,device)

import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch

def save_generated_images(generator, epoch, device, num_imgs=8):
    generator.eval()  
    with torch.no_grad(): 
        z = torch.randn(num_imgs, 100).to(device)
        generated_images = generator(z).cpu()
        
        generated_images = (generated_images + 1) / 2
        
        grid = torchvision.utils.make_grid(generated_images, nrow=4)
        
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.title(f"Epoch: {epoch+1}")
        plt.axis("off")
        plt.show()
    
    generator.train()

train(gen,dis,dataloader,epochs=5)
            