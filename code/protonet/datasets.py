import os.path as osp
import PIL
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
import os

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CUB(Dataset):

    def __init__(self, setname, args, augment=False):
        
        self.IMAGE_PATH =  args.dataset_root
    
        if args.splits_root is None:
            self.SPLIT_PATH = osp.join(args.dataset_root, 'splits')
        else:
            self.SPLIT_PATH = args.splits_root
            
        self.CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')
            
        im_size = args.orig_imsize
        txt_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
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
            self.data, self.label = self.parse_csv(txt_path)
        
        self.num_class = np.unique(np.array(self.label)).shape[0]
        image_size = 84
    
        transforms_list = [
                    transforms.Resize(84),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                  ]
        
        # Transformation
        if setname == 'train' and augment: 
            self.transform = transforms.Compose(
                transforms_list + [transforms.RandomApply([AddGaussianNoise(0., 1.)], p=0.3),
                                   transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                                        np.array([0.229, 0.224, 0.225]))]
            )
        else:
            self.transform = transforms.Compose(transforms_list + [transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                                                                        np.array([0.229, 0.224, 0.225]))])
        
                     
    def parse_csv(self, txt_path):
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))
        return image, label 



class MiniImageNet(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False):
        
        self.IMAGE_PATH =  args.dataset_root
    
        if args.splits_root is None:
            self.SPLIT_PATH = osp.join(args.dataset_root, 'splits')
        else:
            self.SPLIT_PATH = args.splits_root
            
        self.CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')
            
        im_size = args.orig_imsize
        txt_path = osp.join(self.SPLIT_PATH, setname + '.txt')
        folder_path = osp.join(self.IMAGE_PATH)
        cache_path = osp.join(self.CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )

        self.use_im_cache = ( im_size != -1 ) # not using cache
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(csv_path, setname)
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
            self.data, self.label = self.parse_folders(txt_path, folder_path)

        self.num_class = len(set(self.label))

        image_size = 84
        
        transforms_list = [
            transforms.Resize(84),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
          ]
        
        # Transformation
        if setname == 'train' and augment: 
            self.transform = transforms.Compose(
                transforms_list + [transforms.RandomApply([AddGaussianNoise(0., 1.)], p=0.3),
                                   transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                                        np.array([0.229, 0.224, 0.225]))]
            )
        else:
            self.transform = transforms.Compose(transforms_list + [transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                                                                        np.array([0.229, 0.224, 0.225]))])

    def parse_folders(self, txt_path, folder_path):
        data = []
        label = []
        lb = -1
        self.class_names = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

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

class CifarFS(Dataset):

    def __init__(self, setname, args, augment=False):
        
#         if dataset_dir is None:
#             self.IMAGE_PATH =  '../../raid/data/datasets/cifar100/cifar-100-python/cifar-fs'
#             self.SPLIT_PATH = '../../raid/data/datasets/cifar100/cifar-100-python/cifar-fs-splits'
#             self.CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')
        self.IMAGE_PATH =  args.dataset_root
    
        if args.splits_root is None:
            self.SPLIT_PATH = osp.join(args.dataset_root, 'splits')
        else:
            self.SPLIT_PATH = args.splits_root
            
        self.CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')
            
        im_size = args.orig_imsize
        txt_path = osp.join(self.SPLIT_PATH, setname + '.txt')
        folder_path = osp.join(self.IMAGE_PATH, setname)
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
            self.data, self.label = self.parse_folders(txt_path, folder_path)
        
        self.num_class = np.unique(np.array(self.label)).shape[0]
        image_size = 84
    
        transforms_list = [
                    transforms.Resize(84),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                  ]
        
        # Transformation
        if setname == 'train' and augment: 
            self.transform = transforms.Compose(
                transforms_list + [transforms.RandomApply([AddGaussianNoise(0., 1.)], p=0.3),
                                   transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                                        np.array([0.229, 0.224, 0.225]))]
            )
        else:
            self.transform = transforms.Compose(transforms_list + [transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                                                                        np.array([0.229, 0.224, 0.225]))])
            #self.transform = transforms.Compose(transforms_list)
                     
    
    def parse_folders(self, txt_path, folder_path):
        data = []
        label = []
        lb = -1
        self.class_names = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

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

    
    
def identity(x):
    return x