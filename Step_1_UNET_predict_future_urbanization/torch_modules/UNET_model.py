import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

  def forward(self,x):
    x = self.conv(x)
    return x

class UNET(nn.Module):
  def __init__(self,in_channels=3,out_channels=1,features=[64,128,256,512]):
    super().__init__()
    self.downs = nn.ModuleList()
    self.ups = nn.ModuleList()
    self.pool = nn.MaxPool2d(2,2)

    # down part of UNET
    for feature in features:
      self.downs.append(DoubleConv(in_channels,feature))
      in_channels = feature

    # up part of UNET
    for feature in reversed(features):
      self.ups.append(nn.ConvTranspose2d(feature*2,feature,2,2))
      self.ups.append(DoubleConv(feature*2,feature))

    # bottleneck and final-conv 
    self.bottleneck = DoubleConv(features[-1],features[-1]*2)
    self.finalconv = nn.Sequential(
        nn.Conv2d(features[0],out_channels,3,1,1),
        nn.Sigmoid())

  def forward(self,x):

    skip_connections = []

    # 1) downsample the input img
    for dwon in self.downs:
      x = dwon(x)
      skip_connections.append(x)
      x = self.pool(x)
    
    # 2) pass img to bottleneck block
    x = self.bottleneck(x)

    # 3) upsample the img
    skip_connections = skip_connections[::-1]

    for idx in range(0,len(self.ups),2):
      # 3-1) first upsamling img from pervious down layer
      x = self.ups[idx](x)

      # 3-2) then concat the upsampled img with its coresponding dwon layer
      skip_connection = skip_connections[idx//2]

      # in case downsampled layer has a different shape with upsampled layer
      if x.shape != skip_connection.shape:
        x = TF.resize(x,size=skip_connection.shape)
      # concat donw-up layers, passing the concated img to upsamling conv  
      concat_skip = torch.cat((skip_connection,x),dim=1)
      x = self.ups[idx+1](concat_skip)

    # 4) the last layer to squash output features into num. of out_channels
    return self.finalconv(x)    

def test():
  x = torch.randn((3,3,256,256))
  model = UNET()
  preds = model(x)

  print(preds.shape,x.shape)
  
if __name__ == '__main__':
  test()






