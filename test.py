import main
import torch
import os
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

workspace_dir = './extra_data_2'
# load pretrained model
G = main.Generator(100)
G.load_state_dict(torch.load(os.path.join(workspace_dir, 'dcgan_g.pth')))
G.eval()
G.cuda()
# generate images and save the result
n_output = 20
z_sample = Variable(torch.randn(n_output, 100)).cuda()
imgs_sample = (G(z_sample).data + 1) / 2.0
save_dir = os.path.join(workspace_dir, 'logs')
filename = os.path.join(save_dir, f'result.jpg')
torchvision.utils.save_image(imgs_sample, filename, nrow=10)
# show image
grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=10)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()