"""
Create A Data Dictionary
"""

import os
import natsort

def read_dataset(data_dir,trainids,valids,modality):
    datasetlist={'train':{},'val':{}}
    for modal in modality:#OCT/OCTA/Label
        datasetlist['train'].update({modal:{}})
        if modal!=modality[-1]:
            ctlist=os.listdir(os.path.join(data_dir, modal))
            ctlist=natsort.natsorted(ctlist)
            for ct in ctlist[trainids[0]:trainids[1]]:
                datasetlist['train'][modal].update({ct:{}})
                scanlist=os.listdir(os.path.join(data_dir, modal,ct))
                scanlist=natsort.natsorted(scanlist)
                for i in range(0,len(scanlist)):
                    scanlist[i]=os.path.join(data_dir, modal,ct,scanlist[i])
                datasetlist['train'][modal][ct]=scanlist
        else:
            ctlist=os.listdir(os.path.join(data_dir, modal))
            ctlist=natsort.natsorted(ctlist)
            for ct in ctlist[trainids[0]:trainids[1]]:
                datasetlist['train'][modal].update({ct: {}})
                labeladdress = os.path.join(data_dir, modal, ct)
                datasetlist['train'][modal][ct] = labeladdress
    for modal in modality:
        datasetlist['val'].update({modal:{}})
        if modal!=modality[-1]:
            ctlist=os.listdir(os.path.join(data_dir, modal))
            ctlist=natsort.natsorted(ctlist)
            for ct in ctlist[valids[0]:valids[1]]:#id1/id2/id3
                datasetlist['val'][modal].update({ct:{}})
                scanlist=os.listdir(os.path.join(data_dir, modal,ct))
                scanlist=natsort.natsorted(scanlist)
                for i in range(0,len(scanlist)):#1.bmp/2.bmp/.../n.bmp
                    scanlist[i]=os.path.join(data_dir, modal,ct,scanlist[i])
                datasetlist['val'][modal][ct]=scanlist
        else:
            ctlist=os.listdir(os.path.join(data_dir, modal))
            ctlist=natsort.natsorted(ctlist)
            for ct in ctlist[valids[0]:valids[1]]:
                datasetlist['val'][modal].update({ct: {}})
                labeladdress = os.path.join(data_dir, modal, ct)
                datasetlist['val'][modal][ct] = labeladdress
    train_records = datasetlist['train']
    validation_records = datasetlist['val']
    return train_records, validation_records

def read_dataset_post(data_dir,feature_dir,trainids,valids,modality):
    datasetlist = {'train': {}}
    ctlist = os.listdir(os.path.join(data_dir, modality[0]))
    ctlist = natsort.natsorted(ctlist)
    datasetlist['train'].update({'feature': {}})
    datasetlist['train'].update({'label': {}})
    for ct in ctlist[trainids[0]:trainids[1]]:
        datasetlist['train']['feature'].update({ct: {}})
        datasetlist['train']['feature'][ct] = os.path.join(feature_dir,ct+'.npy')
        datasetlist['train']['label'].update({ct: {}})
        datasetlist['train']['label'][ct] = os.path.join(data_dir,modality[-1],ct+'.bmp')
    train_records = datasetlist['train']
    return train_records