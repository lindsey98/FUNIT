import matplotlib.pyplot as plt
import torch
import os
import argparse
import shutil
from utils import get_config, get_train_loaders, make_result_folders, dataset_from_list
from trainer import Trainer

import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data import SubSampler
import numpy as np
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="1, 0"


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/funit_cars.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path', type=str,
                    default='checkpoints/cars196', help="outputs path")
parser.add_argument('--test_batch_size', type=int, default=4)
parser.add_argument('--mix_path', type=str,
                    default='/home/ruofan/PycharmProjects/SoftTriple/datasets/cars196/train_val_mix',
                    help="directory to save mixed images")
# Specify the mixed folder

opts = parser.parse_args()

if __name__ == '__main__':
    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']

    trainer = Trainer(config)
    trainer.cuda()
    config['gpus'] = 1
    mix_folder = opts.mix_path

    # dataloader
    # loaders = get_train_loaders(config)
    # train_content_loader, train_class_loader = loaders[0], loaders[1]
    # test_content_loader, test_class_loader = loaders[2], loaders[3]
    train_content_dataset = dataset_from_list(root=config['data_folder_train'],
                                file_list=config['data_list_train'],
                                batch_size=config['batch_size'],
                                new_size=config['new_size'],
                                height=config['crop_image_height'],
                                width=config['crop_image_width'],
                                crop=True,
                                num_workers=1)
    test_class_dataset = dataset_from_list(root=config['data_folder_test'],
                                file_list=config['data_list_test'],
                                batch_size=config['batch_size'],
                                new_size=config['new_size'],
                                height=config['crop_image_height'],
                                width=config['crop_image_width'],
                                crop=True,
                                num_workers=1)

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')

    iterations = trainer.resume(checkpoint_directory,
                                hp=config, multigpus=False)
    trainer.eval()

    for it, (train_co_class, val_cl_class) in enumerate(zip(train_content_dataset.classes,
                                            test_class_dataset.classes)):

        train_cls_idx = train_content_dataset.class_to_idx[train_co_class]
        test_cls_idx = test_class_dataset.class_to_idx[val_cl_class]
        selected_train_indices = np.where(np.asarray(train_content_dataset.targets) == train_cls_idx)[0]  #
        selected_val_indices = np.where(np.asarray(test_class_dataset.targets) == test_cls_idx)[0]  #

        sampler = SubSampler(selected_train_indices)
        train_content_loader = DataLoader(train_content_dataset,
                                           batch_size=config['batch_size'],
                                           shuffle=False,
                                           drop_last=True,
                                           sampler=sampler)
        sampler_val = SubSampler(selected_val_indices)
        test_class_loader = DataLoader(test_class_dataset,
                                       batch_size=config['batch_size'],
                                       shuffle=False,
                                       drop_last=True,
                                       sampler=sampler_val)

        with torch.no_grad():
            for t, (train_co_data, val_cl_data) in enumerate(zip(train_content_loader, test_class_loader)):
                val_image_outputs = trainer.test(train_co_data, val_cl_data, False)
                '''Save'''
                for j, out in enumerate(val_image_outputs[-1]):
                    new_cls_name = '{}_{}'.format(train_co_class,
                                                  val_cl_class)
                    os.makedirs(os.path.join(mix_folder, new_cls_name), exist_ok=True)
                    file_name = '{}.jpg'.format(len(os.listdir(os.path.join(mix_folder, new_cls_name)))+1)
                    save_image(out, os.path.join(mix_folder, new_cls_name, file_name))

                '''Visualize'''
                # for i in range(len(train_co_data[0])):
                #     plt.subplot(len(train_co_data[0]), 3, 1+3*i)
                #     plt.imshow(val_image_outputs[0][i].permute(1,2,0).detach().cpu().numpy())
                #     plt.subplot(len(train_co_data[0]), 3, 2+3*i)
                #     plt.imshow(val_image_outputs[3][i].permute(1,2,0).detach().cpu().numpy())
                #     plt.subplot(len(train_co_data[0]), 3, 3 + 3 * i)
                #     plt.imshow(val_image_outputs[-1][i].permute(1,2,0).detach().cpu().numpy())
                # plt.show()

        if it >= len(train_content_dataset.classes): # only generate 50 new classes
            break


