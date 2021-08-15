import os
import numpy as np
import cv2
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import glob
import torchvision.transforms as transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())
        self.apply(weights_init)
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())
        self.apply(weights_init)
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

def BGR2RGB(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir),64,64,3), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        img = BGR2RGB(img)
        x[i, :, :] = cv2.resize(img,(64, 64))
        print(i)
    return x

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    import torchvision

    workspace_dir = './extra_data_2'
    # hyperparameters
    batch_size = 64
    z_dim = 100
    lr = 1e-4
    n_epoch = 15
    save_dir = os.path.join(workspace_dir, 'logs')
    os.makedirs(save_dir, exist_ok=True)

    # model
    G = Generator(in_dim=z_dim).cuda()
    D = Discriminator(3).cuda()
    G.train()
    D.train()

    # loss criterion
    criterion = nn.BCELoss()

    # optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    same_seeds(0)
    # dataloader (You might need to edit the dataset path if you use extra dataset.)
    dataset = readfile(os.path.join(workspace_dir, 'faces'),False)
    dataset1 =ImgDataset(dataset,None,train_transform)
    print(dataset[0].shape)
    dataloader = DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=4)

    import matplotlib.pyplot as plt

    z_sample = Variable(torch.randn(100, z_dim)).cuda()
    #
    for e, epoch in enumerate(range(n_epoch)):
        for i, data in enumerate(dataloader):
            imgs = data
            imgs = imgs.cuda()

            bs = imgs.size(0)

            """ Train D """
            z = Variable(torch.randn(bs, z_dim)).cuda()
            r_imgs = Variable(imgs).cuda()
            f_imgs = G(z)

            # label
            r_label = torch.ones((bs)).cuda()
            f_label = torch.zeros((bs)).cuda()

            # dis
            r_logit = D(r_imgs.detach())
            f_logit = D(f_imgs.detach())

            # compute loss
            r_loss = criterion(r_logit, r_label)
            f_loss = criterion(f_logit, f_label)
            loss_D = (r_loss + f_loss) / 2

            # update model
            D.zero_grad()
            loss_D.backward()
            opt_D.step()

            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)

            # compute loss
            loss_G = criterion(f_logit, r_label)

            # update model
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            print(
                f'\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',
                end='')
        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        # show generated image
        grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        G.train()
        if (e + 1) % 5 == 0:
            torch.save(G.state_dict(), os.path.join(workspace_dir, f'dcgan_g.pth'))
            torch.save(D.state_dict(), os.path.join(workspace_dir, f'dcgan_d.pth'))
