

from utils import *


import argparse

from FFANet import FFANet

from optimizer import Adam
from PIL import Image
from visdom import Visdom
from data.utils import decode_seg_map_sequence
from torchvision.utils import save_image
from torchvision import transforms
from thop import profile, clever_format
from ptflops import get_model_complexity_info

vis = Visdom()

pic_path = '/media/ddy/Seagate1/医学图像/CHAOS/MR1/test/Img/subj_16slice_26.png'
#pic = scipy.misc.imread(pic_path,mode='RGB')
pic = Image.open(pic_path)


tran3 = transforms.Grayscale(1)
tran1 = transforms.CenterCrop(256)
tran2 = transforms.ToTensor()
pic = tran2(tran1(tran3(pic)))
#pic = pic.resize((512,512),Image.BILINEAR)
pic = to_var(pic)

# pic = Normalize(pic)

# pic = np.transpose(pic,(2,0,1))
pic = pic.unsqueeze(0)

print("pic shape:{}".format(pic.shape))
# net = FPN([2, 4, 23, 3], 5)
net = PSPNet()
pthfile = r'/media/ddy/Seagate1/yu/Net9.15/model/Best_MR1.pth'
net.load_state_dict(torch.load(pthfile))
net = net.eval()
softMax = nn.Softmax()
if torch.cuda.is_available():
    net.cuda()
    softMax.cuda()
out = net(pic)

#计算参数量1
flops, params = profile(net, inputs=(pic, ))
flops, params = clever_format([flops, params], "%.3f")
print(flops,params)

#计算参数量2
macs, params = get_model_complexity_info(net, (1, 256, 256), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
print(flops)

plotout = torch.argmax(out, dim=1, keepdim=True)
plotout = plotout.squeeze()
# vis.surf(plotout.detach().cpu(), win='surfmap', opts=dict(title='surfmap'))

_, c, _, _ = out.size()
#heatmap
# for i in range (c):
#     print('i = ', i)
#     plotout = out.squeeze()
#     plotout = plotout[i]
#     print("plotout:{}".format(plotout.shape))
#     vis.heatmap(plotout.detach().cpu(),win="heatmap{}".format(i),opts=dict(title="heatmap{}".format(i)))
vis.heatmap(plotout.detach().cpu() , win="heatmap",opts=dict(title="heatmap"))

out = out.data.cpu().numpy()
out = np.argmax(out,axis=1)
pre = decode_seg_map_sequence(out, plot=True)
save_image(pre,r'./testjpg.png')