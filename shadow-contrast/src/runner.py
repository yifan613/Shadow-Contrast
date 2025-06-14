import os
from schedule.schedule import Schedule
from model.DocDiff import DocDiff, EMA
from schedule.diffusionSample import GaussianDiffusion
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

class Runner:
    def __init__(self, config):
        self.mode = config.MODE
        self.schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = config.CHANNEL_X + config.CHANNEL_Y
        out_channels = config.CHANNEL_Y
        self.out_channels = out_channels
        self.network = DocDiff(
            input_channels=in_channels,
            output_channels=out_channels,
            n_channels=config.MODEL_CHANNELS,
            ch_mults=config.CHANNEL_MULT,
            n_blocks=config.NUM_RESBLOCKS
        ).to(self.device)
        self.diffusion = GaussianDiffusion(self.network, config.TIMESTEPS, self.schedule).to(self.device)
        self.test_img_save_path = config.TEST_IMG_SAVE_PATH
        if not os.path.exists(self.test_img_save_path):
            os.makedirs(self.test_img_save_path)
        self.weight_path = config.WEIGHT_PATH
        self.path_train_img = config.PATH_IMG
        self.LR = config.LR
        self.num_timesteps = config.TIMESTEPS
        self.test_path_img = config.TEST_PATH_IMG
        self.pre_ori = config.PRE_ORI
        self.high_low_freq = config.HIGH_LOW_FREQ
        self.image_size = config.IMAGE_SIZE
        from data.docdata import DocData

        dataset_test = DocData(config.TEST_PATH_IMG, config.IMAGE_SIZE, self.mode)
        self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                            drop_last=False,
                                            num_workers=config.NUM_WORKERS)


    def test(self):
        def crop_concat(img, size=128):
            shape = img.shape
            correct_shape = (size*(shape[2]//size+1), size*(shape[3]//size+1))
            one = torch.ones((shape[0], shape[1], correct_shape[0], correct_shape[1]))
            one[:, :, :shape[2], :shape[3]] = img
            # crop
            for i in range(shape[2]//size+1):
                for j in range(shape[3]//size+1):
                    if i == 0 and j == 0:
                        crop = one[:, :, i*size:(i+1)*size, j*size:(j+1)*size]
                    else:
                        crop = torch.cat((crop, one[:, :, i*size:(i+1)*size, j*size:(j+1)*size]), dim=0)
            return crop
        def crop_concat_back(img, prediction, size=128):
            shape = img.shape
            for i in range(shape[2]//size+1):
                for j in range(shape[3]//size+1):
                    if j == 0:
                        crop = prediction[(i*(shape[3]//size+1)+j)*shape[0]:(i*(shape[3]//size+1)+j+1)*shape[0], :, :, :]
                    else:
                        crop = torch.cat((crop, prediction[(i*(shape[3]//size+1)+j)*shape[0]:(i*(shape[3]//size+1)+j+1)*shape[0], :, :, :]), dim=3)
                if i == 0:
                    crop_concat = crop
                else:
                    crop_concat = torch.cat((crop_concat, crop), dim=2)
            return crop_concat[:, :, :shape[2], :shape[3]]

        def min_max(array):
            return (array - array.min()) / (array.max() - array.min())
        with torch.no_grad():

            self.network.load_state_dict(torch.load(self.weight_path), strict=True)
            print('Test Model loaded')
            self.network.eval()
            tq = tqdm(self.dataloader_test)
            sampler = self.diffusion
            iteration = 0
            for img, name in tq:
                tq.set_description(f'Iteration {iteration} / {len(self.dataloader_test.dataset)}')
                iteration += 1
                noisyImage = torch.randn_like(img).to(self.device)
                init_predict = self.network.init_predictor(img.to(self.device), 0)
                sampledImgs = sampler(noisyImage.cuda(), init_predict, img.to(self.device), self.pre_ori)
                finalImgs = sampledImgs
                save_image(finalImgs.cpu(), os.path.join(
                    self.test_img_save_path, f"{name[0]}")  )
                
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop
def ImageTransform(loadSize):
    return {"train": Compose([
        # RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        # RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ])}
