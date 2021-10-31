"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path
from PIL import Image
import torch.utils.data as data
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_filelist_reader(filelist):
    im_list = []
    with open(filelist, 'r') as rf:
        for line in rf.readlines():
            im_path = line.strip()
            im_list.append(im_path)
    return im_list

class SubSampler(Sampler):
    '''
    Customized sampler to subsample data
    '''
    def __init__(self, idlist):
        self.idlist = idlist

    def __iter__(self):
        return iter(self.idlist)

    def __len__(self):
        return len(self.idlist)

class ImageLabelFilelist(data.Dataset):
    def __init__(self, root,
                 filelist, transform=None,
                 filelist_reader=default_filelist_reader,
                 loader=default_loader,
                 return_paths=False):

        self.root = root
        self.im_list = filelist_reader(os.path.join(filelist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.im_list]))) # get the class labels for each files
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(im_path, self.class_to_idx[im_path.split('/')[0]]) for im_path in self.im_list]
        self.targets = [im[1] for im in self.imgs]
        self.return_paths = return_paths
        print('Data loader')
        print("\tRoot: %s" % root)
        print("\tList: %s" % filelist)
        print("\tNumber of classes: %d" % (len(self.classes)))

    def __getitem__(self, index):
        im_path, label = self.imgs[index]
        path = os.path.join(self.root, im_path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, label, path
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)

def prepare_data_list(root, data_name):
    train_root = os.path.join(root, 'train')
    # val_root = os.path.join(root, 'val')
    # test_root = os.path.join(root, 'test')

    train_txt = './datasets/{}_trainall.txt'.format(data_name)
    # val_txt = './datasets/{}_val.txt'.format(data_name)
    # test_txt = './datasets/{}_test.txt'.format(data_name)

    for folder in os.listdir(os.path.join(train_root)):
        for file in os.listdir(os.path.join(train_root, folder)):
            with open(train_txt, 'a+') as f:
                f.write(os.path.join(folder,
                                     file))
                f.write('\n')

    # for folder in os.listdir(os.path.join(val_root)):
    #     for file in os.listdir(os.path.join(val_root, folder)):
    #         with open(val_txt, 'a+') as f:
    #             f.write(os.path.join( folder,
    #                                   file))
    #             f.write('\n')
    #
    #
    # for folder in os.listdir(os.path.join(test_root)):
    #     for file in os.listdir(os.path.join(test_root, folder)):
    #         with open(test_txt, 'a+') as f:
    #             f.write(os.path.join(folder,
    #                                  file))
    #             f.write('\n')
    #
    #
if __name__ == '__main__':
    prepare_data_list(root='/home/ruofan/PycharmProjects/SoftTriple/datasets/logo2k',
                      data_name='logo2k')

    # root = '/home/ruofan/PycharmProjects/SoftTriple/datasets/cub'
    # train_root = os.path.join(root, 'train')
    # ct = 0
    # for folder in os.listdir(os.path.join(train_root)):
    #     for file in os.listdir(os.path.join(train_root, folder)):
    #         ct += 1
    #
    # print(ct)