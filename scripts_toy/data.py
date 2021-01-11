import nrrd
import numpy as np 
import torch
from torch.utils.data import Dataset
import glob
import nrrd
import os

import others.transform_converters as tc


def random_crop(img, crop_point, crop_sz):
    """
    Randomly crop image
    Args:
        img: image in shape (x, y, z)
        crop_point: point (x, y, z) to start cropping
        crop_sz: cropping size in (x, y, z)
    Return:
        cropped image
    """
    return img[crop_point[0]:crop_point[0]+crop_sz[0],
               crop_point[1]:crop_point[1]+crop_sz[1],
               crop_point[2]:crop_point[2]+crop_sz[2]]

def random_crop_tensor(tensor, crop_point, crop_sz):
    """
    Crop tensor
    Args:
        tensor: image in shape (1, x, y, z, c)
        crop_point: point (x, y, z) to start cropping
        crop_sz: cropping size in (x, y, z)
    Return:
        cropped image
    """
    return tensor[:,
                  crop_point[0]:crop_point[0]+crop_sz[0],
                  crop_point[1]:crop_point[1]+crop_sz[1],
                  crop_point[2]:crop_point[2]+crop_sz[2],
                  :]


def flip_img(img, opt):
    """
    Flip image
    Args:
        img: image in shape (x, y, z)
        opt: flip option, 0-no flip, 1-x flip, 2-z flip, 3-both x and z flip
    Return:
        flipped img
    """
    if opt == 1:
        img = np.flip(img, axis=0)
    elif opt == 2:
        img = np.flip(img, axis=2)
    elif opt == 3:
        img = np.flip(img, axis=0)
        img = np.flip(img, axis=2)
    else:
        img = img
    return img


def rot_img(img, k):
    """
    Rotate image
    Args:
        img: image in shape (x, y, z)
        k: 1, 2, or 3 times of 90 degrees
    """
    if k:
        img = np.rot90(img, k, axes=(0,1))
    return img

class GenerateDataDfieldRand(Dataset):
    """
    Generate training and validation data
    where the target is a displacement field
    """
    def __init__(self, sz, N, normalize=False, augment=False, preload=True):
        """
        Args:
            sz: the size of random patches
        """
        self.sz = sz

        self.grid_sz = sz.copy()
        self.grid_sz.insert(0,3)

        self.N = N
        self.normalize = normalize

        # TODO add augmentation at some point
        #self.augment = augment

        self.preload = preload

        self.img_loaded_list = []
        self.posgrid_loaded_list = []


    def __getitem__(self, idx):
        """
        Get specific data corresponding to the index
        Args:
            idx: data index
        Returns:
            tensor (img, dfield)
        """
        im = torch.randn(self.sz)

        #img2 = torch.randn(self.sz) + 3
        #return [img, img2]

        pos_grid = torch.randn(self.grid_sz)
        return [im, pos_grid]

    def __len__(self):
        return self.N

def grid_collate_fun( batch ):
    """
    :param batch: list of [ imgs, grid ]
    :return: list of tensors
    """

    for i,x in enumerate(batch):
        print( i )
        print( x[0].size())
        print( x[1].size())
        print( ' ')


    imgs = [x[0].unsqueeze(0) for x in batch]
    grids = [x[1] for x in batch]

    imt = torch.stack( imgs, dim=0 )
    gridt = torch.stack( grids, dim=0 )

    print(imt.size())
    print(gridt.size())

    return [imt, gridt]

def crop_pad_to( img, sz, is_grid=False ):
    """
    Zero-crops and pads the img about its center as necessary
    so its output is the given size. 3d only.

    :param img: the image
    :param sz: the desired size
    :return: the cropped and padded image
    """

    # find dimensions that need padding
    img_sz = np.array(img.shape)[:3]
    sz = np.array(sz)

    # make pad_img as at least as big as sz
    pad_amt = np.ceil(np.max((sz - img_sz)/2.)).astype(int)
    if is_grid:
        pad_list = [ np.pad(img[:,:,:,i], pad_amt) for i in range(img.shape[3])]
        pad_img = np.transpose( np.stack(pad_list), (1,2,3,0))

    else:
        pad_img = np.pad(img, pad_amt)

    pad_img_sz = np.array(pad_img.shape)[:3]

    lo = np.floor((pad_img_sz - sz)/ 2.0).astype(int)

    if is_grid:
        return pad_img[lo[0] : lo[0]+sz[0],
               lo[1] : lo[1]+sz[1],
               lo[2] : lo[2]+sz[2],
               :]
    else:
        return pad_img[lo[0] : lo[0]+sz[0],
                       lo[1] : lo[1]+sz[1],
                       lo[2] : lo[2]+sz[2]]


class GenerateDataDfield(Dataset):
    """
    Generate training and validation data
    where the target is a displacement field
    """
    def __init__(self, img_list, dfield_list, crop_sz=(128,64,64), normalize=False, augment=False, preload=True):
        """
        Args:
            img_list: a list of image files
            dfield_list: a list of displacement field files
            crop_sz: random cropping size
        """
        self.img_list = img_list
        self.dfield_list = dfield_list
        self.crop_sz = crop_sz
        self.normalize = normalize

        # TODO add augmentation at some point
        #self.augment = augment

        self.preload = preload

        self.img_loaded_list = []
        self.posgrid_loaded_list = []


    def __getitem__(self, idx):
        """
        Get specific data corresponding to the index
        Args:
            idx: data index
        Returns:
            tensor (img, dfield)
        """
        # Get image and template
        img_name = self.img_list[idx]
        img, head = nrrd.read(img_name)
        img = np.float32(img)
        img = crop_pad_to( img, self.crop_sz )

        dfield_name = self.dfield_list[idx]
        dfield, dfield_spacing = tc.load_dfield(dfield_name)
        pos_grid = tc.dfield_to_torch_position(dfield, dfield_spacing)

        # Normalize image
        # position grid does not need normalizing
        if self.normalize:
            img = (img-img.mean()) / img.std()

        # To tensor, shape (channel, x, y, z)
        img = torch.from_numpy(img.copy()).float()
        img = img.unsqueeze( 0 ).unsqueeze( 4 )
        # img is [1 x y z 1]

        # maybe skip crop, initially too?
        if self.crop_sz is not None:
            # Crop on image and template
            x = np.random.randint(0, img.shape[0]-self.crop_sz[0]+1)
            y = np.random.randint(0, img.shape[1]-self.crop_sz[1]+1)
            z = np.random.randint(0, img.shape[2]-self.crop_sz[2]+1)

            img = random_crop_tensor(img, (x,y,z), self.crop_sz)
            pos_grid = random_crop_tensor(pos_grid, (x,y,z), self.crop_sz)

        # # Augmentation to image and template
        # if self.augment:
        #     opt = np.random.randint(4)
        #     img = flip_img(img, opt)
        #     k = np.random.randint(4)
        #     img = rot_img(img, k)

        return [img, pos_grid]

    def __len__(self):
        return len(self.img_list)

class GenerateData(Dataset):
    """
    Generate training and validation data
    """
    def __init__(self, img_list, tmplt_name, crop_sz=(32,32,32), normalize=False, augment=False):
        """
        Args:
            img_list: a list of image names
            tmplt_name: template name
            crop_sz: random cropping size
        """
        self.img_list = img_list
        self.tmplt_name = tmplt_name
        self.crop_sz = crop_sz
        self.normalize = normalize
        self.augment = augment

    def __getitem__(self, idx):
        """
        Get specific data corresponding to the index
        Args:
            idx: data index
        Returns:
            tensor (img, tmplt)
        """
        # Get image and template
        img_name = self.img_list[idx]
        img, head = nrrd.read(img_name)
        img = np.float32(img)
        tmplt, head = nrrd.read(self.tmplt_name)
        tmplt = np.float32(tmplt)

        # Normalize image and template
        if self.normalize:
            img = (img-img.mean()) / img.std()
            tmplt = (tmplt-tmplt.mean()) / tmplt.std()

        # Crop on image and template
        x = np.random.randint(0, img.shape[0]-self.crop_sz[0]+1)
        y = np.random.randint(0, img.shape[1]-self.crop_sz[1]+1)
        z = np.random.randint(0, img.shape[2]-self.crop_sz[2]+1)
        img = random_crop(img, (x,y,z), self.crop_sz)
        tmplt = random_crop(tmplt, (x,y,z), self.crop_sz)

        # Augmentation to image and template
        if self.augment:
            opt = np.random.randint(4)
            img = flip_img(img, opt)
            tmplt = flip_img(tmplt, opt)
            k = np.random.randint(4)
            img = rot_img(img, k)
            tmplt = rot_img(tmplt, k)

        # To tensor, shape (channel, x, y, z)
        img = np.expand_dims(img, axis=0)
        tmplt = np.expand_dims(tmplt, axis=0)
        img = np.concatenate((img, tmplt), axis=0)  # input has two channels, channel0: img, channel1: tmplt
        img = torch.from_numpy(img.copy()).float()
        tmplt = torch.from_numpy(tmplt.copy()).float()

        return [img, tmplt]

    def __len__(self):
        return len(self.img_list)

def dfield2grid( dfield ):
    return dfield

def grid2dfield( grid ):
    dfield = grid[[2, 1, 0], ...] # permute axes
    return dfield

if __name__ == "__main__":

    data_path = '/groups/scicompsoft/home/fleishmang/exper/dldr/mini_training_test/warped_sphere/'
    img_list = [data_path+'warped.nrrd']
    tmplt_name = data_path+'sphere.nrrd'

    Data = GenerateData(img_list, tmplt_name, crop_sz=(64, 64, 64))
    img, tmplt = Data[0]
    print(img.shape)
    print(tmplt.shape)

    # View part of the data
    img = img.numpy()
    tmplt = tmplt.numpy()
    img_ch0 = np.zeros((img.shape[1], img.shape[2], img.shape[3]), dtype=img.dtype)
    img_ch0 = img[0,:,:,:]
    img_ch1 = np.zeros((img.shape[1], img.shape[2], img.shape[3]), dtype=img.dtype)
    img_ch1 = img[1,:,:,:]
    tmplt_ch0 = np.zeros((tmplt.shape[1], tmplt.shape[2], tmplt.shape[3]), dtype=tmplt.dtype)
    tmplt_ch0 = tmplt[0,:,:,:]

    curr_path = os.path.dirname(os.path.abspath(__file__))
    print(curr_path)
    if not os.path.exists(curr_path+'/data_view'):
        os.mkdir(curr_path+'/data_view')
    nrrd.write(curr_path+'/data_view/img_ch0.nrrd', img_ch0)
    nrrd.write(curr_path+'/data_view/img_ch1.nrrd', img_ch1)
    nrrd.write(curr_path+'/data_view/tmplt_ch0.nrrd', tmplt_ch0)
