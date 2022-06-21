from torchvision import transforms as tvtrans
import torch, random, numpy as np, albumentations as alb
from nvidia.dali import fn as nvfn
from nvidia.dali.pipeline import Pipeline as nvPipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
__all__ = ['data_augmentation', 'data_mixup', 'nvJPEGDataloader']
class data_augmentation():
    def __init__(self, method_list, Height = None, Width = None):
        """
        This class performs common data augmentations on images. Note that all images should
        be represented in numpy arrays.
        Parameters:
            method_list: list of methods utilized. Different methods include 'geometry',
                'noise', 'crop and erase', 'normalize'.
            Height: Height of images. If None, resize will not be executed.
            Width: Width of images. If None, resize will not be executed.
        """
        compose_list = []
        if Height is not None and Width is not None:
            compose_list.append(alb.Resize(Height, Width))
        if 'geometry' in method_list:
            compose_list.append(alb.Flip())
            compose_list.append(alb.ShiftScaleRotate())
        if 'crop and erase' in method_list:
            compose_list.append(alb.Cutout())
        if 'noise' in method_list:
            compose_list += [alb.HueSaturationValue(),
                            alb.OneOf([alb.GaussianBlur(), alb.MotionBlur()]),
                            alb.CLAHE(),
                            alb.RandomBrightnessContrast(p = 1)]
        self.transform = alb.Compose(compose_list)
        self.normalize = None
        if 'normalize' in method_list:
            self.normalize = tvtrans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def augment(self, img):
        """
        This function performs data augmentations and returns a torch tensor with shape
            (channel, height, width).
        Parameters:
            img: an image in form of numpy ndarray containing RGB float numbers
                with shape (height, width, channel).
        """
        r = tvtrans.ToTensor()(self.transform(image = img.astype(np.uint8))['image'])
        if self.normalize is not None:
            r = self.normalize(r)
        return r

class data_mixup():
    def __init__(self, alpha = 1):
        """
        This class performs the mixup operation on data.
        Parameters:
            alpha: Distribution parameter for a beta distribution, from which the mixup weight
                is chosen.
        Examples:
            mixup = mixup_class(1)
            x, y = load_data()
            x, y1, y2 = mixup.get_data(x, y)
            output = model(x)
            l1, l2 = criterion(output, y1), criterion(output, y2)
            loss = mixup.get_loss(l1, l2)
        """
        if alpha < 0:
            raise ValueError("alpha should be non-nagetive!")
        self.alpha = alpha
    def __call__(self, x):
        """
        This function mixups inputs and returns a tuple (mixed_x, permutation_ids, lambda),
        where a loss function should be applied to both targets y and y[permutation_ids] with
        the same input mixed_x and obtains two different loss values loss1 and loss2. The total
        loss is computed as loss1 * lambda + loss2 * (1 - lambda).
        Parameters:
            x: input data, batch first.
        """
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        #idxs = torch.randperm(x.shape[0], device = x.device)
        idxs = torch.flip(torch.arange(x.shape[0], device = x.device), [0])
        mixed_x = lam * x + (1 - lam) * x[idxs]
        return mixed_x, idxs, lam

class nvJPEGIterator():
    def __init__(self, dataset, batch_size, shuffle):
        """
        This module defines an iterator to fetch a batch of data.
        Parameters:
            dataset: pytorch-style dataset with a __getitem__(self, index), a __len__ method
                and a num_outs attribute. The getitem method should return a tuple of which an
                image filepath should be the first element. The attribute num_outs indicates
                the length of each tuple returned by the getitem method.
            batch_size: int, number of samples in a batch.
            shuffle: bool, whether to shuffle data at the beginning of an epoch.
        """
        self.batch_size, self.index = batch_size, list(range(len(dataset)))
        self.len, self.num_outs = len(self.index), dataset.num_outs
        self.dataset, self.shuffle, self.cur_index = dataset, shuffle, 0
    def __len__(self):
        return self.len
    def __iter__(self):
        self.cur_index = 0
        if self.shuffle:
            random.shuffle(self.index)
        return self
    def readfile(self, index):
        sample = self.dataset.__getitem__(self.index[index])
        if not isinstance(sample, tuple):
            sample = (sample,)
        image = np.fromfile(sample[0], dtype = np.uint8)
        return (image,) + sample[1:]
    def __next__(self):
        if self.cur_index >= self.len:
            self.__iter__()
            raise StopIteration
        ed = min(self.cur_index + self.batch_size, self.len)
        r = tuple(map(list, zip(*map(self.readfile, list(range(self.cur_index, ed))))))
        self.cur_index = ed
        return r

class nvJPEGDataloader():
    def __init__(self, dataset, batch_size, image_size, shuffle = False, num_threads = 1,
                return_ori_sizes = False):
        """
        This module defines a dataloader to decode images on GPU.
        Parameters:
            dataset: pytorch-style dataset with a __getitem__(self, index), a __len__ method
                and a num_outs attribute. The getitem method should return a tuple of which an
                image filepath should be the first element. The attribute num_outs indicates
                the length of each tuple returned by the getitem method.
            batch_size: int, number of samples in a batch.
            image_size: a tuple (H, W), which is the size that all images in a batch will be
                resized into.
            shuffle: bool, whether to shuffle data at the beginning of an epoch. Default False.
            num_threads: int greater than 0, threads used to load data.
            return_ori_sizes: Whether to return original sizes of images. If true, the element
                at the last index of the returned tuple is a tensor of original image sizes.
        """
        self.source = nvJPEGIterator(dataset, batch_size, shuffle)
        self.num_outs = self.source.num_outs + int(return_ori_sizes)
        self.len = 1 + (len(self.source) - 1) // batch_size
        pipe = nvPipeline(batch_size = batch_size, num_threads = num_threads, device_id = 0)
        with pipe:
            batch = nvfn.external_source(source = self.source, num_outputs = self.source.num_outs)
            batch[0] = nvfn.decoders.image(batch[0], device = "mixed")
            if return_ori_sizes:
                ori_sizes = nvfn.shapes(batch[0])
            batch[0] = nvfn.resize(batch[0], size = image_size)
            if return_ori_sizes:
                pipe.set_outputs(*batch, ori_sizes)
            else:
                pipe.set_outputs(*batch)
        self.pipe = DALIGenericIterator(pipe, ['outs' + str(i)
                                        for i in range(self.num_outs)],
                                        last_batch_padded = True, auto_reset = True,
                                        dynamic_shape = True,
                                        last_batch_policy = LastBatchPolicy.PARTIAL)
    def __iter__(self):
        return self
    def __len__(self):
        return self.len
    def __next__(self):
        res = next(self.pipe)[0]
        res = [res['outs' + str(i)] for i in range(self.num_outs)]
        if len(res) == 1:
            res = res[0]
        return res
