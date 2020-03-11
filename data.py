# Classes to handle train, test datasets.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import mesh
from utils import so3
from utils import se3
from utils import globset
import math

class Mesh2Points:
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh = mesh.clone()
        v = mesh.vertex_array
        return torch.from_numpy(v).type(dtype=torch.float)

class OnUnitSphere:
    def __init__(self, zero_mean=False):
        self.zero_mean = zero_mean

    def __call__(self, tensor):
        if self.zero_mean:
            m = tensor.mean(dim=0, keepdim=True) # [N, D] -> [1, D]
            v = tensor - m
        else:
            v = tensor
        nn = v.norm(p=2, dim=1) # [N, D] -> [N]
        nmax = torch.max(nn)
        return v / nmax

class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True) # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0] # [N, D] -> [D]
        s = torch.max(c) # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        #return self.method1(tensor)
        return self.method2(tensor)


class Resampler:
    """ [N, D] -> [M, D] """
    def __init__(self, num):
        self.num = num

    def __call__(self, tensor):
        num_points, dim_p = tensor.size()
        out = torch.zeros(self.num, dim_p).to(tensor)

        selected = 0
        while selected < self.num:
            remainder = self.num - selected
            idx = torch.randperm(num_points)
            sel = min(remainder, num_points)
            val = tensor[idx[:sel]]
            out[selected:(selected + sel)] = val
            selected += sel
        return out

class RandomTranslate:
    def __init__(self, mag=None, randomly=True):
        self.mag = 1.0 if mag is None else mag
        self.randomly = randomly
        self.igt = None

    def __call__(self, tensor):
        # tensor: [N, 3]
        amp = torch.rand(1) if self.randomly else 1.0
        t = torch.randn(1, 3).to(tensor)
        t = t / t.norm(p=2, dim=1, keepdim=True) * amp * self.mag

        g = torch.eye(4).to(tensor)
        g[0:3, 3] = t[0, :]
        self.igt = g # [4, 4]

        p1 = tensor + t
        return p1

class RandomRotator:
    def __init__(self, mag=None, randomly=True):
        self.mag = math.pi if mag is None else mag
        self.randomly = randomly
        self.igt = None

    def __call__(self, tensor):
        # tensor: [N, 3]
        amp = torch.rand(1) if self.randomly else 1.0
        w = torch.randn(1, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp * self.mag

        g = so3.exp(w).to(tensor) # [1, 3, 3]
        self.igt = g.squeeze(0) # [3, 3]

        p1 = so3.transform(g, tensor) # [1, 3, 3] x [N, 3] -> [N, 3]
        return p1

class RandomRotatorZ:
    def __init__(self):
        self.mag = 2 * math.pi

    def __call__(self, tensor):
        # tensor: [N, 3]
        w = torch.Tensor([0, 0, 1]).view(1, 3) * torch.rand(1) * self.mag

        g = so3.exp(w).to(tensor) # [1, 3, 3]

        p1 = so3.transform(g, tensor)
        return p1

class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip
        self.e = None

    def jitter(self, tensor):
        noise = torch.zeros_like(tensor).to(tensor) # [N, 3]
        noise.normal_(0, self.scale)
        noise.clamp_(-self.clip, self.clip)
        self.e = noise
        return tensor.add(noise)

    def __call__(self, tensor):
        return self.jitter(tensor)


class RandomTransformSE3:
    """ rigid motion """
    def __init__(self, mag=1, mag_randomly=False):
        self.mag = mag
        self.randomly = mag_randomly

        self.gt = None
        self.igt = None

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]

        p1 = se3.transform(g, p0)
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)

class ModelNet(globset.Globset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.offread
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)

class ShapeNet2(globset.Globset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """
    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = mesh.objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class CADset4tracking(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(pm)
        igt = self.rigid_transform.igt

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class CADset4tracking_fixed_perturbation(torch.utils.data.Dataset):
    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None,
                 fmt_trans=False):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation) # twist (len(dataset), 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans # twist or (rotation and translation)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0) # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        twist = torch.from_numpy(numpy.array(self.perturbation[index])).contiguous().view(1, 6)
        pm, _ = self.dataset[index]
        x = twist.to(pm)
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


def get_datasets(args):
    cinfo = None
    if args.categoryfile:
        #categories = numpy.loadtxt(args.categoryfile, dtype=str, delimiter="\n").tolist()
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                Mesh2Points(),\
                OnUnitCube(),\
                Resampler(args.num_points),\
            ])

        traindata = ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testdata = ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        mag_randomly = True
        trainset = CADset4tracking(traindata,\
                        RandomTransformSE3(args.mag, mag_randomly))
        testset = CADset4tracking(testdata,\
                        RandomTransformSE3(args.mag, mag_randomly))

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                ShapeNet2_transform_coordinate(),\
                Mesh2Points(),\
                OnUnitCube(),\
                Resampler(args.num_points),\
            ])

        dataset = ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        traindata, testdata = dataset.split(0.8)

        mag_randomly = True
        trainset = CADset4tracking(traindata,\
                        RandomTransformSE3(args.mag, mag_randomly))
        testset = CADset4tracking(testdata,\
                        RandomTransformSE3(args.mag, mag_randomly))


    return trainset, testset

def get_classification_datasets(args):
    cinfo = None
    if args.categoryfile:
        #categories = numpy.loadtxt(args.categoryfile, dtype=str, delimiter="\n").tolist()
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                Mesh2Points(),\
                OnUnitCube(),\
                Resampler(args.num_points),\
                RandomRotatorZ(),\
                RandomJitter()\
            ])

        trainset = ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testset = ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                ShapeNet2_transform_coordinate(),\
                Mesh2Points(),\
                OnUnitCube(),\
                Resampler(args.num_points),\
                RandomRotatorZ(),\
                RandomJitter()\
            ])

        dataset = ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        trainset, testset = dataset.split(0.8)

    return trainset, testset