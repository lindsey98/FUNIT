import matplotlib.pyplot as plt
import torch
import os
import argparse
import shutil

from tensorboardX import SummaryWriter

from utils import get_config, get_train_loaders, make_result_folders
from utils import write_loss, write_html, write_1images, Timer
from trainer import Trainer

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="1, 0"


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/funit_cub.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path', type=str,
                    default='checkpoints/cub200', help="outputs path")
parser.add_argument('--test_batch_size', type=int, default=4)
opts = parser.parse_args()

if __name__ == '__main__':
    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']

    trainer = Trainer(config)
    trainer.cuda()
    config['gpus'] = 1

    # dataloader
    loaders = get_train_loaders(config)
    train_content_loader, train_class_loader = loaders[0], loaders[1]
    test_content_loader, test_class_loader = loaders[2], loaders[3]

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')

    iterations = trainer.resume(checkpoint_directory,
                                hp=config, multigpus=False)
    trainer.eval()
    while True:
        with torch.no_grad():
            for t, (train_co_data, val_cl_data) in enumerate(zip(train_content_loader, test_class_loader)):
                if t >= opts.test_batch_size:
                    break
                val_image_outputs = trainer.test(train_co_data, val_cl_data, False)
                print(len(train_co_data))
                for i in range(len(train_co_data[0])):
                    plt.subplot(len(train_co_data[0]), 3, 1+3*i)
                    plt.imshow(val_image_outputs[0][i].permute(1,2,0).detach().cpu().numpy())
                    plt.subplot(len(train_co_data[0]), 3, 2+3*i)
                    plt.imshow(val_image_outputs[3][i].permute(1,2,0).detach().cpu().numpy())
                    plt.subplot(len(train_co_data[0]), 3, 3 + 3 * i)
                    plt.imshow(val_image_outputs[2][i].permute(1,2,0).detach().cpu().numpy())
                plt.show()
                exit()


