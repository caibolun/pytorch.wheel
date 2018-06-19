import torch.utils.data as data

from PIL import Image

import os
import os.path

def find_classes(listfile):
    classes = []
    file = open(listfile) 
    for line in file:
        fname, target = line.split()
        if target not in classes:
            classes.append(target)
    file.close()
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(listfile, class_to_idx, root=""):
    images = []
    file = open(listfile) 
    for line in file:
        fname, target = line.split()
        path = os.path.join(root, fname)
        item = (path, class_to_idx[target])
        images.append(item)
    file.close()
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
        
class ImageList(data.Dataset): 
    def __init__(self, listfile, transform=None, loader=default_loader, target_transform=None, root=""):
        classes, class_to_idx = find_classes(listfile)
        samples = make_dataset(listfile, class_to_idx, root)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in list file of: " + listfile + "\n"))
        self.listfile = listfile
        self.root = root
        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        #sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    List File: {}\n'.format(self.listfile)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

