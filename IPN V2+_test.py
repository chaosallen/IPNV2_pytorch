import torch
import torch.nn as nn
import logging
import sys
import os
import model
import numpy as np
import scipy.misc as misc
from options.test_options import TestOptions
import natsort
from scipy import io

def test_net(net,device):
    DATA_SIZE = opt.data_size
    test_results = os.path.join(opt.saveroot, 'test_results_V2+')
    net.eval()
    test_images = np.zeros((1, opt.channels,DATA_SIZE[1], DATA_SIZE[2]))
    testids = opt.test_ids
    valids = opt.val_ids
    featurelist0 = os.listdir(os.path.join(opt.dataroot, opt.modality_filename[0]))
    featurelist0 = natsort.natsorted(featurelist0)
    featurelist = featurelist0[valids[0]:valids[1]]+featurelist0[testids[0]:testids[1]]
    for cube in featurelist:
        test_images[0, :, :, :] = np.load(os.path.join(opt.feature_dir, cube + '.npy'))
        images = torch.from_numpy(test_images)
        images = images.to(device=device, dtype=torch.float32)
        pred,featuremap= net(images)
        pred = torch.nn.functional.softmax(pred, dim=1)
        result=pred[0,1, :,:].cpu().detach().numpy()*255
        misc.imsave(os.path.join(test_results, cube + ".bmp"), result.astype(np.uint8))
        featuremap = np.squeeze(featuremap.cpu().detach().numpy(), 0)
        #io.savemat(os.path.join('logs/Features_V2+', cube + ".mat"), {'feature': featuremap})
        print(cube)

if __name__ == '__main__':
    #setting logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #loading options
    opt = TestOptions().parse()
    #setting GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    #loading network
    if opt.plane_perceptron == 'UNet_3Plus':
        net = model.UNet_3Plus(in_channels=opt.plane_perceptron_channels, channels=opt.plane_perceptron_channels, n_classes=opt.n_classes)
    if opt.plane_perceptron == 'UNet':
        net = model.UNet(in_channels=opt.plane_perceptron_channels, channels=opt.plane_perceptron_channels, n_classes=opt.n_classes)
    #load trained model
    restore_path = os.path.join(opt.saveroot, 'checkpoints_V2+', '4800.pth')
    net.load_state_dict(
        torch.load(restore_path, map_location=device)
    )
    #input the model into GPU
    net.to(device=device)
    try:
        test_net(net=net,device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
