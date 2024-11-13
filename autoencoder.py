import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(16,8, kernel_size=3, stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8,16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,3,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

train_dataset = datasets.Flowers102(root= 'flowers',
                                    split='train',
                                    transform=transform,
                                    download=True)

test_dataset = datasets.Flowers102(root='flowers',
                                   split='test',
                                   transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=128)

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr =0.001)

num_epochs = 50
for epoch in range (num_epochs):
    for data in train_loader:
        img,_=data
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output,img)
        loss.backward()
        optimizer.step()
    if epoch % 5==0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


torch.save(model.state_dict(), 'conv_autoencoder.pth')


with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        recon = model(data)
        break

import matplotlib.pyplot as plt
plt.figure(dpi=500)
fig, ax =plt.subplots(2,7,figsize=(15,4))
for i in range (7):
    ax[0,i].imshow(data[i].cpu().numpy().transpose((1,2,0)))
    ax[0,i].imshow(recon[i].cpu().numpy().transpose((1,2,0)))
    ax[0,i].axis('OFF')
    ax[0,i].axis('OFF')
plt.show()

