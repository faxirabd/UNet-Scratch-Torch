import torch
import torch.nn as nn
#https://debuggercafe.com/unet-from-scratch-using-pytorch/#download-code

def double_convol(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        )
    
    return conv

def image_croping(tensor, target_tensor):
    target_size= target_tensor.size()[2]
    tensor_size= tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta]
    

class UNetOrg(nn.Module):
    def __init__(self, classes):
        super(UNetOrg, self).__init__()
    
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_convol_1 = double_convol(3, 64)
        self.down_convol_2 = double_convol(64, 128)
        self.down_convol_3 = double_convol(128, 256)
        self.down_convol_4 = double_convol(256, 512)
        self.down_convol_5 = double_convol(512, 1024)
        
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2, 
            stride=2)
        
        self.up_conv_1 = double_convol(1024, 512)
        
        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2, 
            stride=2)
        
        self.up_conv_2 = double_convol(512, 256)
        
        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2, 
            stride=2)
        
        self.up_conv_3 = double_convol(256, 128)
        
        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2, 
            stride=2)
        
        self.up_conv_4 = double_convol(128, 64)
        
        # out_channels = number of classes
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=classes,
            kernel_size=1)
        
    def forward(self, x):
        #contracting part
        x1 = self.down_convol_1(x)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_convol_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_convol_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_convol_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_convol_5(x8)

        
        #distracting part
        x = self.up_trans_1(x9)
        y = image_croping(x7, x)
        x = self.up_conv_1(torch.cat([y, x], 1))
        
        x = self.up_trans_2(x)
        y = image_croping(x5, x)
        x = self.up_conv_2(torch.cat([y, x], 1))
        
        x = self.up_trans_3(x)
        y = image_croping(x3, x)
        x = self.up_conv_3(torch.cat([y, x], 1))
        
        x = self.up_trans_4(x)
        y = image_croping(x1, x)
        x = self.up_conv_4(torch.cat([y, x], 1))
        
        x = self.out(x)
        #print(x.size())
        return x

if __name__ == "__main__":
    image = torch.rand((1, 3, 572, 572))
    model = UNetOrg(5)
    print(model(image))