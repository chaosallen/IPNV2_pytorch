import os
import numpy as np
import scipy.misc as misc
import h5py
import random
import torch
class BatchDatset:
    def __init__(self, records_list,modality,datasize,blocksize,channels,batch_size,cube_num,dataclass,saveroot):
        self.saveroot = saveroot
        self.filelist = records_list
        self.datasize = datasize
        self.blocksize = blocksize
        self.channels = channels
        self.batch_size = batch_size
        self.dataclass = dataclass
        self.modality=modality
        self.images = np.zeros((batch_size, channels, blocksize[0], blocksize[1], blocksize[2]))
        self.annotations = np.zeros((batch_size, 1, blocksize[1], blocksize[2]))
        self.transformkey=0
        self.top = 0
        self.left = 0
        self.isEpoch = False

        if datasize[0]!=blocksize[0]:#if reduce the height of data
            self.transformkey = 1

        #self.cube_num = len(list(self.filelist['label']))
        self.cube_num = cube_num

        self.data = np.zeros((channels, blocksize[0], datasize[1], datasize[2], self.cube_num), dtype=np.uint8)
        self.label = np.zeros((1, datasize[1], datasize[2], self.cube_num), dtype=np.uint8)
        self.read_images()
        self.pos_start = 0

    def read_images(self):
        if not os.path.exists(os.path.join(self.saveroot,self.dataclass+"data.hdf5")):
            print(self.dataclass+"picking ...It will take some minutes")
            modality_num = -1
            for modality in self.filelist.keys():
                if modality != self.modality[-1]:
                    ctlist=list(self.filelist[modality])
                    modality_num+=1
                    ct_num=-1
                    for ct in ctlist:
                        ct_num+=1
                        scanlist=list(self.filelist[modality][ct])
                        scan_num=-1
                        for scan in scanlist:
                            scan_num+=1
                            self.data[modality_num,:,:,scan_num,ct_num]=np.array(self.image_transform(scan,self.transformkey))
                else:
                    ctlist=list(self.filelist[modality])
                    ct_num=-1
                    for ct in ctlist:
                        ct_num+=1
                        labeladress=self.filelist[modality][ct]
                        self.label[0,:,:,ct_num]=np.array(self.image_transform(labeladress,0))
            f= h5py.File(os.path.join(self.saveroot,self.dataclass+"data.hdf5"), "w")
            f.create_dataset('data',data=self.data)
            f.create_dataset('label',data=self.label)
            f.close
        else:
            print("found pickle !!!")
            f = h5py.File(os.path.join(self.saveroot,self.dataclass+"data.hdf5"), "r")
            self.data = f['data']
            self.label = f['label']
            f.close


    def image_transform(self, filename, key):
        image = misc.imread(filename)
        if key:
            resize_image = misc.imresize(image,[self.blocksize[0], self.datasize[1]], interp='nearest')
        else:
            resize_image = image
        return np.array(resize_image)

    def read_batch_random_train(self):#vessel segmentation
        for batch in range(0,self.batch_size):
            nx=random.randint(self.blocksize[1]/2,self.datasize[1]-self.blocksize[1]/2)
            ny=random.randint(self.blocksize[2]/2,self.datasize[2]-self.blocksize[2]/2)
            startx = nx-int(self.blocksize[1]/2)
            endx = nx+int(self.blocksize[1]/2)
            starty= ny-int(self.blocksize[2]/2)
            endy = ny + int(self.blocksize[2]/2)
            ctnum = random.randint(0, self.cube_num - 1)
            self.images[batch, :,:, 0:self.blocksize[1], 0:self.blocksize[2]] = self.data[:,:, startx:endx,starty:endy, ctnum].astype(np.float32)
            self.annotations[batch,0,0:self.blocksize[1],0:self.blocksize[2]]=self.label[:,startx:endx,starty:endy, ctnum].astype(np.float32)
            image = torch.from_numpy(self.images)
            label = torch.from_numpy(self.annotations)
        return image, label

    def read_batch_normal_train(self):#FAZ segmentation
        sd=50 #Standard Deviation
        for batch in range(self.batch_size):
            nx=int(np.random.normal(self.datasize[1]/2,sd))
            ny=int(np.random.normal(self.datasize[2]/2,sd))
            startx = nx-int(self.blocksize[1]/2)
            endx = nx+int(self.blocksize[1]/2)
            starty= ny-int(self.blocksize[2]/2)
            endy = ny + int(self.blocksize[2]/2)
            while startx<0 or starty<0 or endx>self.datasize[1] or endy>self.datasize[2]:
                nx = int(np.random.normal(self.datasize[1] / 2, sd))
                ny = int(np.random.normal(self.datasize[2] / 2, sd))
                startx = nx - int(self.blocksize[1] / 2)
                endx = nx + int(self.blocksize[1] / 2)
                starty = ny - int(self.blocksize[2] / 2)
                endy = ny + int(self.blocksize[2] / 2)
            ctnum = random.randint(0, self.cube_num - 1)
            self.images[batch, :, :, 0:self.blocksize[1], 0:self.blocksize[2]] = self.data[:, :, startx:endx, starty:endy, ctnum].astype(np.float32)
            self.annotations[batch, 0, 0:self.blocksize[1], 0:self.blocksize[2]] = self.label[:, startx:endx, starty:endy, ctnum].astype(np.float32)
            image = torch.from_numpy(self.images)
            label = torch.from_numpy(self.annotations)
        return image, label




class BatchDatset_post:
    def __init__(self, records_list,datasize,channels,batch_size,cube_num,saveroot):
        self.saveroot = saveroot
        self.filelist = records_list
        self.datasize = datasize
        self.channels = channels
        self.batch_size = batch_size
        self.images = np.zeros((batch_size, channels,datasize[1], datasize[2]))
        self.annotations = np.zeros((batch_size, datasize[1], datasize[2]))
        self.top = 0
        self.left = 0
        self.cube_num = cube_num
        self.data = np.zeros((channels, datasize[1], datasize[2], self.cube_num), dtype=np.uint8)
        self.label = np.zeros((1, datasize[1], datasize[2], self.cube_num), dtype=np.uint8)
        self.read_images()
        self.pos_start = 0

    def read_images(self):
        if not os.path.exists(os.path.join(self.saveroot,"feature.hdf5")):
            print("picking ...It will take some minutes")
            ctlist = list(self.filelist['feature'])
            for i, ct_num in enumerate(ctlist):
                self.data[:,:,:,i] = np.array(np.load(self.filelist['feature'][ct_num]))
                self.label[0,:,:,i]=np.array(misc.imread(self.filelist['label'][ct_num]))
            f= h5py.File(os.path.join(self.saveroot,"feature.hdf5"), "w")
            f.create_dataset('data',data=self.data)
            f.create_dataset('label',data=self.label)
            f.close
        else:
            print("found pickle !!!")
            f = h5py.File(os.path.join(self.saveroot,"feature.hdf5"), "r")
            self.data = f['data']
            self.label = f['label']
            f.close
    def read_batch_feature(self):#vessel segmentation
        for batch in range(0,self.batch_size):
            ctnum=random.randint(0, self.cube_num - 1)
            self.images[batch, :,:,:] = self.data[:,:,:,ctnum].astype(np.float32)
            self.annotations[batch,:,:]=self.label[0,:,:,ctnum].astype(np.float32)
            image = torch.from_numpy(self.images)
            label = torch.from_numpy(self.annotations)
        return image, label