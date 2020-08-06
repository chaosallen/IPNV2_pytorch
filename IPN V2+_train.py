import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import model
import utils
import shutil
import natsort
from options.train_options import TrainOptions
import scipy.misc as misc
import data_process.readData as readData
import data_process.BatchDataReader as BatchDataReader
from torchsummary import summary


def train_net(net,device):
    #train setting
    interval=opt.save_interval_post
    train_num = opt.train_ids[1] - opt.train_ids[0]
    val_num = opt.val_ids[1] - opt.val_ids[0]
    DATA_SIZE = opt.data_size
    best_valid_dice=0
    val_images = np.zeros((1, opt.channels, DATA_SIZE[1], DATA_SIZE[2]))
    model_save_path = os.path.join(opt.saveroot, 'checkpoints_V2+')
    best_model_save_path = os.path.join(opt.saveroot, 'best_model_V2+')
    # Read Data
    print("Start Setup dataset reader")
    train_records= readData.read_dataset_post(data_dir=opt.dataroot,feature_dir=opt.feature_dir, trainids=opt.train_ids, valids=opt.val_ids,modality=opt.modality_filename)
    print("Setting up dataset reader")
    train_dataset_reader = BatchDataReader.BatchDatset_post(train_records, opt.data_size, opt.channels, opt.batch_size, train_num, opt.saveroot)
    # Setting Optimizer
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=1e-6)
    elif opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), opt.lr, betas=(0.9, 0.99))
    elif opt.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(net.parameters(), opt.lr, weight_decay=1e-8)
    #Setting Loss
    criterion = nn.CrossEntropyLoss()
    #Start train
    for itr in range(0, opt.max_iteration_post):
        net.train()
        train_images, train_annotations = train_dataset_reader.read_batch_feature()

        train_images =train_images.to(device=device, dtype=torch.float32)
        train_annotations = train_annotations.to(device=device, dtype=torch.long)
        pred,_= net(train_images)
        loss = criterion(pred, train_annotations)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % 10 == 0:
            print(itr,loss.item())
        #Start Val
        with torch.no_grad():
            if itr % interval==0:
                #Save model
                torch.save(net.module.state_dict(),
                           os.path.join(model_save_path,f'{itr}.pth'))
                logging.info(f'Checkpoint {itr} saved !')
                #Calculate validation Dice
                val_Dice_sum = 0
                net.eval()
                valids = opt.val_ids
                featurelist0 = os.listdir(os.path.join(opt.dataroot, opt.modality_filename[0]))
                featurelist0 = natsort.natsorted(featurelist0)
                featurelist = featurelist0[valids[0]:valids[1]]
                for kk, cube in enumerate(featurelist):
                    label= misc.imread(os.path.join(opt.dataroot, opt.modality_filename[opt.in_channels], f'{cube}.bmp'))
                    val_images[0, :, :,:] = np.load(os.path.join(opt.feature_dir, cube+'.npy'))
                    images = torch.from_numpy(val_images)
                    images = images.to(device=device, dtype=torch.float32)
                    pred,_= net(images)
                    pred_argmax = torch.argmax(pred, dim=1)
                    result = pred_argmax[0, :, :].cpu().detach().numpy()
                    val_Dice_sum+= utils.cal_Dice(result,label)
                val_Dice=val_Dice_sum/val_num
                print("Step:{}, Valid_Dice:{}".format(itr,val_Dice))
                #save best model
                if val_Dice > best_valid_dice:
                    temp = '{:.6f}'.format(val_Dice)
                    os.mkdir(os.path.join(best_model_save_path,temp))
                    temp2= f'{itr}.pth'
                    shutil.copy(os.path.join(model_save_path,temp2),os.path.join(best_model_save_path,temp,temp2))

                    model_names = natsort.natsorted(os.listdir(best_model_save_path))
                    #print(len(model_names))
                    if len(model_names) == 4:
                        shutil.rmtree(os.path.join(best_model_save_path,model_names[0]))
                    best_valid_dice = val_Dice



if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TrainOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    if opt.plane_perceptron == 'UNet_3Plus':
        net = model.UNet_3Plus(in_channels=opt.planar_perceptron_channels, channels=opt.planar_perceptron_channels, n_classes=opt.n_classes)
    if opt.plane_perceptron == 'UNet':
        net = model.UNet(in_channels=opt.planar_perceptron_channels, channels=opt.planar_perceptron_channels, n_classes=opt.n_classes)

    net=torch.nn.DataParallel(net,[0]).cuda()
    #summary(net, (2,160,100,100), opt.batch_size)
    #load trained model
    if opt.load:
        net.load_state_dict(
            torch.load(opt.load, map_location=device)
        )
        logging.info(f'Model loaded from {opt.load}')
    #input the model into GPU
    #net.to(device=device)
    try:
        train_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)




