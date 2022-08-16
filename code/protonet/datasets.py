import os.path as osp
import PIL
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.distributions import exponential
from torch.distributions import log_normal
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
import kornia
from tqdm.notebook import tqdm
import os

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))

class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class GammaCorrection():
        
    def __call__(self, tensor):
        gamma = torch.tensor(np.random.rayleigh(scale=0.5))
        return torch.pow(tensor, torch.exp(gamma))


class Blur():
        
    def __call__(self, tensor):
        kernel_size = 5
        sigma = exponential.Exponential(torch.tensor([2.0, 2.0])).sample().tolist()
        return F_t.gaussian_blur(tensor, kernel_size, sigma=sigma)


class BrightnessContrast():
    def __init__(self, sigma_b=None, sigma_c=None):
        self.sigma_b = sigma_b
        self.sigma_c = sigma_c
        
    def __call__(self, tensor):
        if self.sigma_b is None:
            contrast_factor = log_normal.LogNormal(0., self.sigma_c).sample()
            return F_t.adjust_contrast(tensor, contrast_factor)
        if self.sigma_c is None:
            brightness_factor = torch.randn(1) * self.sigma_b
            return kornia.enhance.adjust_brightness(tensor, brightness_factor.to(torch.int32), clip_output=False)

class Translation():
        
    def __call__(self, tensor):
        translation = torch.randn(1, 2) * tensor.shape[-1]
        return kornia.geometry.transform.translate(tensor, translation, padding_mode='reflection')
    
class CustomDataset(Dataset):
    
    def __init__(self, setname, args, augment=False):
        
        self.setname = setname
        self.augment = augment
        self.IMAGE_PATH =  args.dataset_root
        
        if self.augment:
            self.aug_sigma = args.sigma
    
        if args.splits_root is None:
            self.SPLIT_PATH = osp.join(args.dataset_root, 'splits')
        else:
            self.SPLIT_PATH = args.splits_root
            
        self.CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')
            
        im_size = args.orig_imsize
        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        txt_path = osp.join(self.SPLIT_PATH, setname + '.txt')
        cache_path = osp.join(self.CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )
        

        self.use_im_cache = ( im_size != -1 ) # not using cache
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(txt_path)
                self.data = [ resize_(Image.open(path).convert('RGB')) for path in data ]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label }, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data  = cache['data']
                self.label = cache['label']
        else:
            
            if os.path.exists(csv_path):
                lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
                self.data, self.label = self._parse_csv(csv_path)

            elif os.path.exists(txt_path):
                lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
                if os.path.exists(osp.join(self.IMAGE_PATH, setname)):
                    folder_path = osp.join(self.IMAGE_PATH, setname)
                else:
                    folder_path = self.IMAGE_PATH
                self.data, self.label = self._parse_folders(txt_path, folder_path)
        
        self.num_class = np.unique(np.array(self.label)).shape[0]
        
        # Transformation
        self._set_transform()
            
    def _set_transform(self):
        raise NotImplementedError

            
    def _parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                
            data.append(path)
            label.append(lb)

        return data, label
    
    def _parse_folders(self, txt_path, folder_path):
        data = []
        label = []
        lb = -1
        self.class_names = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()]

        for l in lines:
            class_name = l
            path = osp.join(folder_path, class_name)
            if class_name not in self.class_names:
                self.class_names.append(class_name)
                lb += 1
                
            #data.append(path)
            images = os.listdir(path)
            data += [osp.join(path, image) for image in images]
            label += [lb] * len(images)

        return data, label


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))
        return image, label 

"""
Few-shot datasets
"""

class CUB(CustomDataset):
    
    def _set_transform(self):
        image_size = 84
        transforms_list = [
                    transforms.Resize(84),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                  ]
        
        # Transformation
        
        normalization = transforms.Normalize(np.array([104/255.0, 117/255.0, 128/255.0]),
                                             np.array([1.0/255, 1.0/255, 1.0/255]))
        
        if self.setname == 'train' and self.augment: 
            self.transform = transforms.Compose(
                transforms_list + [transforms.RandomApply([AddGaussianNoise(0., self.aug_sigma)], p=0.2),
                                   normalization])
                
        else:
            self.transform = transforms.Compose(transforms_list + [normalization])
            

class MiniImageNet(CustomDataset):
    
    def _set_transform(self):
        image_size = 84
        transforms_list = [
            transforms.Resize(84),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
          ]
        
        # Transformation
        
        normalization = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                             np.array([0.229, 0.224, 0.225]))
        
        if self.setname == 'train' and self.augment: 
            self.transform = transforms.Compose(
                transforms_list + [transforms.RandomApply([AddGaussianNoise(0., self.aug_sigma)], p=0.2),
                                   normalization])
                
        else:
            self.transform = transforms.Compose(transforms_list + [normalization])
            
            
class CifarFS(CustomDataset):
    
    def _set_transform(self):
        image_size = 84
        transforms_list = [
            transforms.Resize(84),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
          ]
        
        # Transformation
        
        normalization = transforms.Normalize(np.array([0.4914, 0.4822, 0.4465]),
                                             np.array([0.2023, 0.1994, 0.2010]))
        
        if self.setname == 'train' and self.augment: 
            self.transform = transforms.Compose(
                transforms_list + [transforms.RandomApply([AddGaussianNoise(0., self.aug_sigma)], p=0.2),
                                   normalization])
                
        else:
            self.transform = transforms.Compose(transforms_list + [normalization])
            

def identity(x):
    return x
