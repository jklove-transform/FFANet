from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
from torch import autograd, optim

import medicalDataLoader

from utils import *
from visdom import Visdom

#import dill
import argparse

from FFANet import FFANet
#from unet import Unet

from optimizer import Adam

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



def runTraining(args):
    print('------------------------------------------' )
    print('           Starting the train              ')
    print('------------------------------------------' )

    batch_size = args.batch_size
    batch_size_val = 1
    lr = args.lr

    epoch = args.epochs
    root_dir = args.root_dir
    model_dir = 'model'
    print(' Dataset: {} '.format(root_dir))
    transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])

    train_set = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=True,
                                                      equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=5,
                              shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=5,
                            shuffle=False)

                                                                    
    # Initialize
    net = FFANet()
    #net = torch.nn.DataParallel(net, device_ids=device_ids)
    print(" Save pth Name: {}".format(args.modelName))
    print('------------------------------------------')


    net.apply(weights_init)

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        net = net.cuda()
        softMax.cuda()
        CE_loss.cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)

    BestDice, BestEpoch = 0, 0

    Losses = []
    dataset = args.dataset
    #print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        net.train()
        lossVal = []
        totalImages = len(train_loader)

        print('Epoch {}/{}'.format(i, epoch - 1))

        print('v' * 10)

        for j, data in enumerate(train_loader):
            image, labels, img_names = data

            #print(data[0].shape)
            #print(data[1].shape)
            #print(data[2])

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            optimizer.zero_grad()
            MRI = to_var(image)
            Segmentation = to_var(labels)

            ################### Train ###################
            net.zero_grad()
            segmentation_prediction = net(MRI)
            #print('output:', segmentation_prediction.shape)
            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)
            # Cross-entropy loss
            loss = CE_loss(segmentation_prediction, Segmentation_class)

            loss.backward()
            optimizer.step()
            loss_visual = loss.cpu().data.numpy()

            vis.line([loss_visual], [i], win=f'LOSS:{dataset}', update='append')

            print(f"[Data:{dataset}] epoch:%d--------%d/%d,train_loss:%0.4f" % (i, j+1, totalImages, loss.item()))

            lossVal.append(loss_visual)

            '''printProgressBar(j + 1, totalImages,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Loss: {:.4f} ".format(
                                 loss_visual))'''
        # vis.scatter([[i, np.mean(lossVal)]],
        #             win=f'Loss:{dataset}',
        #             opts=dict(title=f'Loss:{dataset}'),
        #             update='append')


      
        #printProgressBar(totalImages, totalImages,
                             #done="[Training] Epoch: {}, Loss: {:.4f}".format(i,np.mean(lossVal)))
       
        # Save statistics
        modelName = args.modelName

        Losses.append(np.mean(lossVal))

        if i >= 4:

            dice, ravd, assd, mssd = inference(net, val_loader, args.dataset)

            currentDice = dice
            #viz.line([currentDice], [i], win='currentDice-cbam', update='append')
            vis.line([dice], [i], win=f'DICE:{dataset}', update='append')
            vis.line([ravd], [i], win=f'RAVD:{dataset}', update='append')
            vis.line([assd], [i], win=f'ASSD:{dataset}', update='append')
            vis.line([mssd], [i], win=f'MSSD:{dataset}', update='append')
            print("[val] metrics: (1): {:.4f} (2): {:.4f}  (3): {:.4f} (4): {:.4f}".format(dice,ravd,assd,mssd)) # MRI

            if currentDice > BestDice:
                BestDice = currentDice

                BestEpoch = i

                if currentDice > 0.40:

                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    torch.save(net.state_dict(), os.path.join(model_dir, "Best_" + modelName + ".pth"))

            print("<<<                                                                      >>>")
            print("<<<                   Best Dice: {:.4f} at epoch {}.                     >>>".format(BestDice, BestEpoch))
            print("<<<  Current epoch Dice: {:.4f} Ravd: {:.4f} Assd: {:.4f} Mssd: {:.4f}   >>>".format(dice, ravd, assd, mssd))
            # print("###    Best Dice in 3D: {:.4f} with Dice(1): {:.4f} Dice(2): {:.4f} Dice(3): {:.4f} Dice(4): {:.4f} ###".format(np.mean(BestDice3D),BestDice3D[0], BestDice3D[1], BestDice3D[2], BestDice3D[3] ))
            print(f"<<<                    current learning rate:{lr}                        >>>")
            print("<<<                                                                      >>>")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        if i % (BestEpoch + 50) == 0:
            for param_group in optimizer.param_groups:
                lr = lr*0.5
                param_group['lr'] = lr
                print(' ----------  New learning Rate: {}'.format(lr))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser.add_argument("--root_dir", type=str, default="/media/ddy/Seagate1/医学图像/CHAOS/MR1")

    parser.add_argument("--dataset",type=str, default="MR1")
    parser.add_argument("--modelName",type=str, default="MR1")

    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--epochs',type=int, default=250)
    parser.add_argument('--lr',type=float, default=0.001)
    args=parser.parse_args()
    # visualization
    vis = Visdom()
    dataset = args.dataset
    # viz.line([0.], [0], win='currentDice-cbam', opts=dict(title='currentDice-cbam'))
    vis.line([0.], [0], win=f'LOSS:{dataset}', opts=dict(title=f'LOSS:{dataset}'), update='append')
    vis.line([0.], [0], win=f'DICE:{dataset}', opts=dict(title=f'DICE:{dataset}'), update='append')
    vis.line([0.], [0], win=f'RAVD:{dataset}', opts=dict(title=f'RAVD:{dataset}'), update='append')
    vis.line([0.], [0], win=f'ASSD:{dataset}', opts=dict(title=f'ASSD:{dataset}'), update='append')
    vis.line([0.], [0], win=f'MSSD:{dataset}', opts=dict(title=f'MSSD:{dataset}'), update='append')

runTraining(args)
