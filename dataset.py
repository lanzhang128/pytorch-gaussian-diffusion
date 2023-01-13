import os
import PIL
import torchvision.transforms as trans
import torchvision.datasets as tvd


def get_trans(pipeline=None):
    if pipeline is None:
        return None
    transform = []
    for p in pipeline:
        name = p['name']
        p.pop('name')
        transform.append(trans.__getattribute__(name)(**p))
    transform = trans.Compose(transform)
    return transform


class CIFAR10(tvd.CIFAR10):
    def __init__(self, pipeline=None, **kwargs):
        transform = get_trans(pipeline)
        super().__init__(transform=transform, **kwargs)
    
    def __getitem__(self, index):
        return super().__getitem__(index)[0]


class CelebA_HQ_256(tvd.VisionDataset):
    def __init__(self, root, pipeline=None):
        self.transform = get_trans(pipeline)
        for _, _, files in os.walk(root):
            self.filename = [os.path.join(root, file) for file in files]
    
    def __getitem__(self, index):
        X = PIL.Image.open(self.filename[index])
        if self.transform is not None:
            X = self.transform(X)
        return X
    
    def __len__(self):
        return len(self.filename)
        