import torch
import model


a = torch.ones(1, 1, 512, 512, device="cuda:0")
encoder = model.DenseCap().cuda()
with torch.no_grad():
    print(encoder(a).shape)

b = torch.ones(1, 1024, 8, 8, device="cuda:0")
decoder = model.Decoder().cuda()
with torch.no_grad():
    print(decoder(b).shape)

