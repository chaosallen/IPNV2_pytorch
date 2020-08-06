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
    BLOCK_SIZE = opt.block_size
    test_results = os.path.join(opt.saveroot, 'test_results')
    feature_results= opt.feature_dir
    net.eval()
    test_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], BLOCK_SIZE[1], BLOCK_SIZE[2]))
    cube_images = np.zeros((1, opt.in_channels, BLOCK_SIZE[0], DATA_SIZE[1], DATA_SIZE[2]))

    modalitylist = opt.modality_filename
    testids = opt.test_ids
    valids = opt.val_ids
    trainids= opt.train_ids
    cubelist0 = os.listdir(os.path.join(opt.dataroot, modalitylist[0]))
    cubelist0 = natsort.natsorted(cubelist0)
    cubelist =cubelist0[trainids[0]:trainids[1]]+cubelist0[valids[0]:valids[1]]+cubelist0[testids[0]:testids[1]]
    #cubelist = cubelist0[valids[0]:valids[1]] + cubelist0[testids[0]:testids[1]]

    vote_time=4
    for kk,cube in enumerate(cubelist):

        bscanlist = os.listdir(os.path.join(opt.dataroot, modalitylist[0], cube))
        bscanlist=natsort.natsorted(bscanlist)
        for i,bscan in enumerate(bscanlist):
            for j,modal in enumerate(modalitylist):
                if modal!=opt.modality_filename[-1]:
                    cube_images[0,j,:,:,i]=np.array(misc.imresize(misc.imread(os.path.join(opt.dataroot,modal,cube,bscan)),[BLOCK_SIZE[0], DATA_SIZE[1]], interp='nearest'))
        result =np.zeros((DATA_SIZE[1], DATA_SIZE[2]))
        featuremap=np.zeros((opt.plane_perceptron_channels,DATA_SIZE[1], DATA_SIZE[2]))
        votemap=np.zeros((DATA_SIZE[1], DATA_SIZE[2]))

        for i in range(0,DATA_SIZE[1]-BLOCK_SIZE[1]+BLOCK_SIZE[1]//vote_time,BLOCK_SIZE[1]//vote_time):
            for j in range(0,DATA_SIZE[2]-BLOCK_SIZE[2]+BLOCK_SIZE[2]//vote_time,BLOCK_SIZE[2]//vote_time):
                test_images[0, :, 0:BLOCK_SIZE[0], 0:BLOCK_SIZE[1], 0:BLOCK_SIZE[2]] = cube_images[0, :, :,i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]
                images = torch.from_numpy(test_images)
                images = images.to(device=device, dtype=torch.float32)
                pred,features = net(images)
                pred = torch.nn.functional.softmax(pred, dim=1)
                votemap[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]=votemap[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+1
                result[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]] = result[i:i+BLOCK_SIZE[1], j:j+BLOCK_SIZE[2]]+pred[0,1,0,:,:].cpu().detach().numpy()
                featuremap[:,i:i + BLOCK_SIZE[1], j:j + BLOCK_SIZE[2]] = featuremap[:,i:i + BLOCK_SIZE[1],j:j + BLOCK_SIZE[2]] + features[0,:,:,:].cpu().detach().numpy()

        result=result/votemap*255
        featuremap=featuremap/votemap
        print(cube)
        misc.imsave(os.path.join(test_results, cube + ".bmp"), result.astype(np.uint8))
        np.save(os.path.join(feature_results, cube + ".npy"),featuremap)
        #io.savemat(os.path.join(feature_results, cube + ".mat"), {'feature':featuremap})
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
    if opt.method == 'IPN':
        net = model.IPN(in_channels=opt.in_channels, channels=opt.channels, n_classes=opt.n_classes)
    if opt.method == 'IPN_V2':
        net = model.IPN_V2(in_channels=opt.in_channels, channels=opt.channels,plane_perceptron_channels=opt.plane_perceptron_channels, n_classes=opt.n_classes,
                           block_size=opt.block_size, plane_perceptron=opt.plane_perceptron)

    #load trained model
    bestmodelpath= os.path.join(opt.saveroot, 'best_model',natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])
    restore_path = os.path.join(opt.saveroot, 'best_model',natsort.natsorted(os.listdir(os.path.join(opt.saveroot, 'best_model')))[-1])+'/'+os.listdir(bestmodelpath)[0]
    print(restore_path)
    #restore_path = os.path.join(opt.saveroot, 'checkpoints', '27000.pth')
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
