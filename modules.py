"""
Modulo
Neural Modular Networks in PyTorch

modules.py
"""

import torch
import numpy as np

from torch import nn, Tensor
from torch.nn import Module
from torch.autograd import Variable
import torch.nn.functional as F

from schema import VQAModuloTask


class SHAPESModuloTask(VQAModuloTask):

    def __init__(self):
        super(SHAPESModuloTask, self).__init__(
            task_type='vqa',
            text_modules=[FindModule, TransformModule, AnswerModule, AndModule],
            image_modules=[ConvolutionLayers],
            param_dict={
                'batch_size': 64,
                'channels': 64,
                'height': 64,
                'width': 64,
                'x_txt_dim': 16,
                'map_dim': 16,
                'output_channel_size': 64,
                'elt_dim': 1,
                'kernel_size': 3,
                'num_choices': 1,
                'attn_reduced': 3,
                'use_dict': {'use_case': 'images'}
            },
            module_dict={'_Find': FindModule, '_Transform': TransformModule,
                         '_And': AndModule, '_Answer': AnswerModule})

    def check_modules(self):
        super(SHAPESModuloTask, self).check_modules()

    def assemble(self, rev_polish, xvis, xtxt):

        _module_input_num = {'_Find': 0,
                             '_Transform': 1,
                             '_And': 2,
                             '_Answer': 1}

        rpn = rev_polish.split()
        stack = []
        ctr = 0
        while rpn:
            mod_str = rpn.pop(0)
            mod = self.module_dict[mod_str]
            if _module_input_num[mod_str] == 1:
                # Answer
                if mod_str == '_Answer':
                    mod.update_params(stack.pop())
                else:
                    # Transform
                    mod.update_params(xtxt[:, ctr, :], stack.pop())
            elif _module_input_num[mod_str] == 2:
                # And
                mod.update_params(stack.pop(), stack.pop())
            else:
                # Find
                mod.update_params(xvis, xtxt[:, ctr, :])
            att_grid = mod.forward()
            stack.append(att_grid)
            ctr += 1
        return stack.pop()


class ConvolutionLayers(Module):
    def __init__(self, param_dict=None, input_dim=3, hidden_dim=64,
                 output_dim=64, kernel_1=10, kernel_2=1, stride_1=10,
                 stride_2=1):

        super(ConvolutionLayers, self).__init__()
        self.name = '_Img'
        use_dict = param_dict['use_dict']
        if not use_dict:
            raise ValueError('Must supply a use dictionary')

        use_case = use_dict['use_case']

        if use_case == 'images':
            self.conv_1 = nn.Conv2d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_1,
                stride=stride_1,
                padding=0
            )

            self.conv_2 = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=kernel_2,
                stride=stride_2,
                padding=0
            )
            nn.init.xavier_uniform(self.conv_1.weight)
            nn.init.xavier_uniform(self.conv_2.weight)

            self.xvis_layer_1 = nn.Sequential(self.conv_1, nn.ReLU())
            self.xvis_layer_2 = nn.Sequential(self.conv_2, nn.ReLU())

        elif use_case == 'find':
            self.conv_linear = nn.Linear(*use_dict['conv_linear_1'])
            self.conv_linear_2 = nn.Linear(*use_dict['conv_linear_2'])
            self.fc_linear = torch.nn.Linear(*use_dict['fc_linear'])
            nn.init.xavier_uniform(self.conv_linear.weight)
            nn.init.xavier_uniform(self.conv_linear_2.weight)
            nn.init.xavier_uniform(self.fc_linear.weight)

        elif use_case == 'transform':
            self.conv_linear = nn.Linear(*use_dict['conv_linear'])
            self.fc_linear = nn.Linear(*use_dict['fc_linear'])
            self.conv_2d = nn.Conv2d(
                in_channels=use_dict['conv_2d_input_dim'],
                out_channels=use_dict['conv_2d_output_dim'],
                kernel_size=use_dict['conv_2d_kernel_size'],
                stride=use_dict['conv_2d_stride_size'],
                padding=use_dict['conv_2d_kernel_size'] // 2
            )
            nn.init.xavier_uniform(self.conv_linear.weight)
            nn.init.xavier_uniform(self.fc_linear.weight)
            nn.init.xavier_uniform(self.conv_2d.weight)

        elif use_case == 'answer':
            self.fc_linear = nn.Linear(*use_dict['fc_linear'])
            nn.init.xavier_uniform(self.fc_linear.weight)

        else:
            raise ValueError('{} is not a valid use_case'.format(use_case))

    def forward(self, input_batch):
        return self.xvis_layer_2(self.xvis_layer_1(input_batch))

    def conv_1x1(self, input_batch, output_dim, find_extra=False):
        N, C, H, W = input_batch.size()
        flattened = input_batch.view(N, -1, C)
        if not find_extra:
            conv_flat = self.conv_linear(flattened)
        else:
            conv_flat = self.conv_linear_2(flattened)

        reshaped_conv = conv_flat.view(N, H, W, output_dim)
        return reshaped_conv

    def conv(self, input_batch):
        input_batch = input_batch.permute(0, 3, 1, 2).contiguous()
        return self.conv_2d(input_batch)

    def fc_layer(self, input_batch, with_relu=False):
        if len(list(input_batch.size())) > 2:
            input_batch = input_batch.permute(1, 0, 2).contiguous()
        shape = list(input_batch.size())
        input_dim = int(np.prod(shape[1:]))
        batch_size = shape[0]
        flattened = input_batch.view(batch_size, -1, input_dim)

        if with_relu:
            return torch.nn.ReLU(self.fc_linear(flattened))
        else:
            return self.fc_linear(flattened)


class FindModule(Module):
    def __init__(self, param_dict):
        super(FindModule, self).__init__()
        self.name = "_Find"
        self.txt_dim = param_dict['x_txt_dim']
        self.batch_size = param_dict['batch_size']
        self.channels = param_dict['channels']
        self.h = param_dict['height']
        self.w = param_dict['width']
        self.map_dim = param_dict['map_dim']
        self.img_dim = param_dict['output_channel_size']
        self.elt_dim = param_dict['elt_dim']
        self.gen_nn = ConvolutionLayers(param_dict={
            'use_dict': {
                'use_case': 'find',
                'conv_linear_1': (self.img_dim, self.map_dim),
                'conv_linear_2': (self.map_dim, self.elt_dim),
                'fc_linear': (self.txt_dim, self.map_dim)
            }})
        self.xvis = None
        self.xtxt = None

    def update_params(self, xvis, xtxt):
        self.xvis = xvis
        self.xtxt = xtxt.contiguous()

    def forward(self):
        """
        image_feat_grid = _slice_image_feat_grid(self.batch_idx)
        text_param = _slice_word_vecs(self.time_idx, self.batch_idx)
        # Mapping: image_feat_grid x text_param -> att_grid
        # Input:
        #   image_feat_grid: [N, H, W, D_im]
        #   text_param: [N, D_txt]
        # Output:
        #   att_grid: [N, H, W, 1]
        #
        # Implementation:
        #   1. Element-wise multiplication between image_feat_grid
               and text_param
        #   2. L2-normalization
        #   3. Linear classification
        """

        img_mapped = self.gen_nn.conv_1x1(self.xvis, self.map_dim)
        txt_mapped = self.gen_nn.fc_layer(self.xtxt)
        txt_mapped = txt_mapped.unsqueeze(2)
        prod = img_mapped * txt_mapped
        prod = nn.functional.normalize(prod, p=2, dim=3).permute(0, 3, 1, 2).contiguous()
        att_grid = self.gen_nn.conv_1x1(prod, output_dim=1, find_extra=True)
        return att_grid


class TransformModule(Module):
    def __init__(self, param_dict):

        super(TransformModule, self).__init__()
        self.name = "_Transform"
        self.batch_size = param_dict['batch_size']
        self.elt_dim = param_dict['elt_dim']
        self.kernel_size = param_dict['kernel_size']
        self.map_dim = param_dict['map_dim']
        self.txt_dim = param_dict['x_txt_dim']
        self.gen_nn = ConvolutionLayers(
            param_dict={
                'use_dict': {
                    'use_case': 'transform',
                    'conv_linear': (self.map_dim, self.elt_dim),
                    'fc_linear': (self.txt_dim, self.map_dim),
                    'conv_2d_kernel_size': self.kernel_size,
                    'conv_2d_output_dim': self.map_dim,
                    'conv_2d_input_dim': 1,
                    'conv_2d_stride_size': 1
                }})
        self.xtxt = None
        self.submodule = None

    def update_params(self, xtxt, submodule):
        self.xtxt = xtxt.contiguous()
        self.submodule = submodule

    def forward(self):
        """
        Mapping: att_grid x text_param -> att_grid
        Input:
          att_grid: [N, H, W, 1]
          text_param: [N, D_txt]
        Output:
          att_grid_transformed: [N, H, W, 1]
                    Implementation:
          Convolutional layer that also involve text_param
          A 'soft' convolutional kernel that is modulated by text_param
        """
        att_grid = self.submodule
        att_mapped = self.gen_nn.conv(att_grid)
        att_mapped.permute(0, 2, 3, 1).contiguous()
        txt_mapped = self.gen_nn.fc_layer(self.xtxt)
        txt_mapped = txt_mapped.view(self.batch_size, self.map_dim, 1, 1)
        prod = att_mapped * txt_mapped
        prod = nn.functional.normalize(prod, p=2, dim=3)
        att_grid = self.gen_nn.conv_1x1(prod, output_dim=1)
        return att_grid


class AndModule(Module):
    def __init__(self, param_dict):
        super(AndModule, self).__init__()
        self.name = "_And"
        self.submodule1 = None
        self.submodule2 = None

    def update_params(self, submodule1, submodule2):
        self.submodule1 = submodule1
        self.submodule2 = submodule2

    def forward(self):
        """
        Mapping: att_grid x att_grid -> att_grid
        Input:
          att_grid_1: [N, H, W, 1]
          att_grid_2: [N, H, W, 1]
        Output:
          att_grid_and: [N, H, W, 1]

        Implementation:
          Take the element-wise min
        """
        att_grid_1 = self.submodule1
        att_grid_2 = self.submodule2
        att_grid_and = torch.min(att_grid_1, att_grid_2)
        return att_grid_and


class AnswerModule(Module):
    def __init__(self, param_dict):
        super(AnswerModule, self).__init__()
        self.name = "_Answer"
        self.num_choices = param_dict['num_choices']
        self.attn_reduced = param_dict['attn_reduced']
        self.gen_nn = ConvolutionLayers({
            'use_dict': {
                'use_case': 'answer',
                'fc_linear': (self.attn_reduced, self.num_choices)
            }})
        self.submodule = None

    def update_params(self, submodule):
        self.submodule = submodule

    def forward(self):
        """
        Mapping: att_grid -> answer probs
        Input:
          att_grid: [N, H, W, 1]
        Output:
          answer_scores: [N, self.num_choices]
                    Implementation:
          1. Max-pool over att_grid
          2. a linear mapping layer (without ReLU)
        """
        att_grid = self.submodule
        # reduce_min, mean, max implementation brute?
        att_min = torch.min(torch.min(att_grid, 1)[0], 1)[0]
        att_avg = torch.mean(torch.mean(att_grid, 1), 1)
        att_max = torch.max(torch.max(att_grid, 1)[0], 1)[0]
        # att_reduced has shape [N, 3]
        att_reduced = torch.cat((att_min, att_avg, att_max), 1)
        # print(att_reduced.shape)
        scores = self.gen_nn.fc_layer(att_reduced)
        return scores
