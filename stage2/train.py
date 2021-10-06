#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets2 import ImageDataset
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=6, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netG_A2C = Generator(opt.input_nc, opt.output_nc)
netG_C2A = Generator(opt.input_nc, opt.output_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)
netD_C = Discriminator(opt.output_nc)

if opt.cuda:
    #netG_A2B.cuda()
    netG_A2B.to(torch.device('cuda'))
    #netG_B2A.cuda()
    netG_B2A.to(torch.device('cuda'))
    #netG_A2C.cuda()
    netG_A2C.to(torch.device('cuda'))
    #netG_C2A.cuda()
    netG_C2A.to(torch.device('cuda'))
    #netD_A.cuda()
    netD_A.to(torch.device('cuda'))
    #netD_B.cuda()
    netD_B.to(torch.device('cuda'))
    #netD_C.cuda()
    netD_C.to(torch.device('cuda'))

    #netG_A2B = nn.DataParallel(netG_A2B, device_ids=[0, 1, 2])
    #netG_B2A = nn.DataParallel(netG_B2A, device_ids=[0, 1, 2])
    #netD_A = nn.DataParallel(netD_A, device_ids=[0, 1, 2])
    #netD_B = nn.DataParallel(netD_B, device_ids=[0, 1, 2])

    netG_A2B = nn.DataParallel(netG_A2B, device_ids=[0])
    netG_B2A = nn.DataParallel(netG_B2A, device_ids=[0])
    netG_A2C = nn.DataParallel(netG_B2A, device_ids=[0])
    netG_C2A = nn.DataParallel(netG_B2A, device_ids=[0])
    netD_A = nn.DataParallel(netD_A, device_ids=[0])
    netD_B = nn.DataParallel(netD_B, device_ids=[0])
    netD_C = nn.DataParallel(netD_B, device_ids=[0])

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netG_A2C.apply(weights_init_normal)
netG_C2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
netD_C.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters(), netG_A2C.parameters(), netG_C2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_C = torch.optim.Adam(netD_C.parameters(), lr=opt.lr, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_C = torch.optim.lr_scheduler.LambdaLR(optimizer_D_C, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
input_C = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
# target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
target_real = Tensor(opt.batchSize).fill_(1.0)
target_fake = Tensor(opt.batchSize).fill_(0.0)


# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
fake_C_buffer = ReplayBuffer()
fake_A2_buffer = ReplayBuffer()
fake_B2_buffer = ReplayBuffer()
fake_C2_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(opt.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # 归一化到[0, 1] 维度转换, 例如[128, 128, 1] --> [1, 128, 128]
                transforms.Normalize((0.5,), (0.5,)) ] # 将[0, 1]归一化到[-1, 1]  mean, std

#transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
#                transforms.RandomCrop(opt.size), 
#                transforms.RandomHorizontalFlip(),
#                transforms.ToTensor(),
#                transforms.Normalize(mean=[0.5], std=[0.5]) ]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######

epoch_values = []
d_loss_values = []
g_loss_values = []

for epoch in range(opt.epoch, opt.n_epochs):
    fp = open("loss.txt", "a")
    for i, batch in enumerate(dataloader):
        # Set model input
        real_A = input_A.copy_(batch['A'])
        real_B = input_B.copy_(batch['B'])
        real_C = input_C.copy_(batch['C'])

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # same_B = netG_A2B(real_B)
        # loss_identity_B = criterion_identity(same_B, real_B)*5.0
        # # G_B2A(A) should equal A if real A is fed
        # same_A = netG_B2A(real_A)
        # loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real) # generator让pred_fake接近1

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        fake_C = netG_A2C(real_A)
        pred_fake = netD_C(fake_C)
        loss_GAN_A2C = criterion_GAN(pred_fake, target_real) # generator让pred_fake接近1

        fake_A2 = netG_C2A(real_C)
        pred_fake = netD_A(fake_A2)
        loss_GAN_C2A = criterion_GAN(pred_fake, target_real) # generator让pred_fake接近1

        fake_B2 = netG_A2B(fake_A2)
        pred_fake = netD_B(fake_B2)
        loss_GAN_C2B = criterion_GAN(pred_fake, target_real) # generator让pred_fake接近1

        fake_C2 = netG_A2C(fake_A)
        pred_fake = netD_C(fake_C2)
        loss_GAN_B2C = criterion_GAN(pred_fake, target_real) # generator让pred_fake接近1

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        recovered_C = netG_A2C(fake_A2)
        loss_cycle_CAC = criterion_cycle(recovered_C, real_C)*10.0

        recovered_A2 = netG_C2A(fake_C)
        loss_cycle_ACA = criterion_cycle(recovered_A2, real_A)*10.0

        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_GAN_A2C + loss_GAN_C2A + loss_cycle_ABA + loss_cycle_BAB + loss_cycle_ACA + loss_cycle_CAC
        #loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB


        loss_G.backward(retain_graph=True)
        optimizer_G.step()

        loss_G.backward()
        optimizer_G.step()

        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach()) # fake_A 由 G 生成, detach 使更新不影响G

        fake_A2 = fake_A2_buffer.push_and_pop(fake_A2)
        pred_fake2 = netD_A(fake_A2.detach()) # fake_A2 由 G 生成, detach 使更新不影响G

        loss_D_fake = (criterion_GAN(pred_fake, target_fake) + criterion_GAN(pred_fake2, target_fake))*0.5

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.33
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())

        fake_B2 = fake_B2_buffer.push_and_pop(fake_B2)
        pred_fake2 = netD_B(fake_B2.detach())

        loss_D_fake = (criterion_GAN(pred_fake, target_fake) + criterion_GAN(pred_fake2, target_fake))*0.5

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.33
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_C.zero_grad()

        # Real loss
        pred_real = netD_C(real_C)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_C = fake_C_buffer.push_and_pop(fake_C)
        pred_fake = netD_C(fake_C.detach())

        fake_C2 = fake_C2_buffer.push_and_pop(fake_C2)
        pred_fake2 = netD_C(fake_C2.detach())

        loss_D_fake = (criterion_GAN(pred_fake, target_fake) + criterion_GAN(pred_fake2, target_fake))*0.5

        # Total loss
        loss_D_C = (loss_D_real + loss_D_fake)*0.33
        loss_D_C.backward()

        optimizer_D_C.step()


        if i==1 :
           epoch_values.append(epoch+1)
           g_loss_values.append(loss_G)
           d_loss_values.append(loss_D_A + loss_D_B + loss_D_C)

           log_format = "loss_G: %5f, loss_D: %5f \n"
           fp.write(log_format % (loss_G, loss_D_A + loss_D_B))


        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        #logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
        #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
        #             images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    fp.close()

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    if epoch % 20 == 19:

        torch.save(netG_A2B.state_dict(), 'output/{}_netG_A2B.pth'.format(epoch))
        torch.save(netG_B2A.state_dict(), 'output/{}_netG_B2A.pth'.format(epoch))
        torch.save(netD_A.state_dict(), 'output/{}_netD_A.pth'.format(epoch))
        torch.save(netD_B.state_dict(), 'output/{}_netD_B.pth'.format(epoch))

plt.plot(epoch_values, d_loss_values, '-r', label='d loss')
plt.plot(epoch_values, g_loss_values, '-b', label='g loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc='upper left')
plt.title("Training Loss")
plt.show()

###################################
