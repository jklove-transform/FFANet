from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar


import medicalDataLoader
#import crf
#from crf import NonRGBCRF

from utils import *
from visdom import Visdom

import argparse
from vovnet_cbam import vovnet57
from optimizer import Adam
from visdom import Visdom

viz = Visdom()

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def CRF_val(args):
    root_dir = '/media/ddy/Seagate1/医学图像/ISIC2017'
    batch_size_test_save = 1
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    test_set = medicalDataLoader.MedicalImageDataset('test',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    test_loader_save_images = DataLoader(test_set,
                                        batch_size=batch_size_test_save,
                                        num_workers=4,
                                        shuffle=False)
    net = vovnet57()
    # net = DAF_stack()
    pthfile = r'/media/ddy/Seagate1/yu/实验/Net9.15ISIC/model/Best_ISIC.pth'
    net.load_state_dict(torch.load(pthfile))
    total = len(test_loader_save_images)
    net.eval()
    if torch.cuda.is_available():
        net.cuda()
    img_names_ALL = []
    val_path = "./test/{}".format(args.dataset)
    ground_dir = "./Data_3D/TESTGROUND/{}".format(args.dataset)
    dicom_dir = "./Data_3D/DICOM_test/{}".format(args.dataset)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    softMax = nn.Softmax().cuda()
    for i, data in enumerate(test_loader_save_images):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)

        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImage(pred_y)

        str_1 = img_names[0].split('/Img/')
        str_subj = str_1[1]
        torchvision.utils.save_image(segmentation.data, os.path.join(val_path, str_subj))

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    # ======= Directories =======
    print("ground_dir：{}".format(ground_dir))
    # seg_dir = os.path.normpath(cwd + '/Data_3D/segmentation')
    seg_dir = val_path
    print("seg_dir：{}".format(seg_dir))
    print("dicom_dir：{}".format(dicom_dir))

    # ======= Volume Reading =======
    dices = 0
    JAs = 0
    ACCs = 0
    ses = 0
    for i in range(1, total + 1):
        Vseg = png_reader(ground_dir + f'/{1500+i}.png')
        Vref = png_reader(seg_dir + f'/{+i}.png')

        dice = dice_coef(Vseg, Vref)
        dices += dice

        metric = SegmentationMetric(2)
        metric.addBatch(Vseg, Vref)

        ACC = metric.pixelAccuracy()
        ACCs += ACC
        JA = metric.JA()
        JAs += JA
        se  = metric.SeSp()
        ses += se

    dice = dices / total
    JA = JAs / total
    ACC = ACCs / total
    se = ses / total

    print('DICE=%.3f  JA=%.3f  ACC=%.3f  SE=%.3f ' % (dice, JA, ACC, se))

    return dice, JA, se, ACC


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser.add_argument("--dataset",default="ISIC",type=str)
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--use_crf', default=False,
                        action='store_true', help='use crf or not')
    args=parser.parse_args()
    CRF_val(args)

