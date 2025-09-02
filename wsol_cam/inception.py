"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from .util import initialize_weights
from .util import remove_layer

__all__ = ['inception_v3']

model_urls = {
    'inception_v3_google':
        'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, 1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, 1)
        self.branch5x5_2 = BasicConv2d(48, 64, 5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size,
                                     stride=stride, padding=padding)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, 3,
                                          stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride,
                                   padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, 1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7_2 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, (7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, 1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, (1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionCam(nn.Module):
    def __init__(self, num_classes=1000, large_feature_map=False, **kwargs):
        super(InceptionCam, self).__init__()

        self.large_feature_map = large_feature_map
        self.unfreeze_layer = kwargs['unfreeze_layer']
        self.model_structure = kwargs['model_structure'] 

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, 3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, 3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, 3, stride=1, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, 1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, 3, stride=1, padding=0)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.SPG_A3_1b = nn.Sequential(
            nn.Conv2d(768, 1024, 3, padding=1),
            nn.ReLU(True),
        )
        self.SPG_A3_2b = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(True),
        )

        # last layer
        self.SPG_A4 = nn.Conv2d(1024, num_classes, 1, padding=0)

        if self.model_structure == 'b2': 
            self.SPG_A4_2 = nn.Conv2d(1024, num_classes, 1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if kwargs['init_weights']:
            initialize_weights(self.modules(), init_mode='xavier')
        if self.unfreeze_layer != 'all':
            self.freeze_layers() 
        
        if kwargs['debug']:
            self.check_params() 

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]
        
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        if not self.large_feature_map:
            x = F.max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)

        x = F.dropout(feat, 0.5, self.training)
        x = self.SPG_A3_1b(x)
        x = F.dropout(x, 0.5, self.training)
        x = self.SPG_A3_2b(x)
        x = F.dropout(x, 0.5, self.training)
        feat_map = self.SPG_A4(x) 

        logits = self.avgpool(feat_map)
        logits = logits.view(logits.shape[0:2])
        results = {'logits': logits}

        if return_cam:
            feature_map = feat_map.clone().detach()  
            cams = feature_map[range(batch_size), labels]
            feature_map = x.detach().clone()

            if labels is None:
                print('none!!')
                labels = logits.argmax(dim=1)
                results['labels'] = labels

            cam_weights = self.SPG_A4.weight[labels].squeeze(-1).squeeze(-1)  # [B, 1024]
            cams = (cam_weights.unsqueeze(-1).unsqueeze(-1) * feature_map).sum(dim=1)
            results['cams'] = cams
                
        if self.model_structure == 'b2':
            feat_map_2 = self.SPG_A4_2(x) 
            logits_2 = self.avgpool(feat_map_2) 
            logits_2 = logits_2.view(logits_2.shape[0:2])
            results['logits2'] = logits_2 
            if return_cam:
                feature_map_2 = x.detach().clone()
                cam_weights_2 = self.SPG_A4_2.weight[labels].squeeze(-1).squeeze(-1)  # [B, 1024]
                
                cam_weights_2 = torch.where(
                        cam_weights_2 > 0,
                        cam_weights_2,
                        torch.zeros_like(cam_weights_2)
                    )
                
                cams = (cam_weights_2.unsqueeze(-1).unsqueeze(-1) * feature_map_2).sum(dim=1)
                results['cams2'] = cams 
                
        return results

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def check_params(self):
        trainable_params_lst = []
        print(f'[Sanity Check] Trainable Parameters are following ...')
        for name, param in self.named_parameters():
            print(f"{name} >> {param.requires_grad}")
            if param.requires_grad:
                trainable_params_lst.append(name)
        print('----------------------------------------------------------')
        print(f"Trainable Parameters: {trainable_params_lst}")
        print(f"Total # of parameters: {self.count_parameters()}")
        print('----------------------------------------------------------') 

    def freeze_layers(self):
        for name, param in self.named_parameters():
            param.requires_grad = False  
            if self.unfreeze_layer in ['SPG_A4','SPG_A4_2']: 
                types = [self.unfreeze_layer + ".weight", self.unfreeze_layer + ".bias"] 
                if name in types:
                    param.requires_grad = True

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()  
                module.track_running_stats = False
                module.affine = False 


def load_pretrained_model(model, path=None):
    if path:
        state_dict = torch.load(
            os.path.join(path, 'inception_v3.pth'))
    else:
        state_dict = load_url(model_urls['inception_v3_google'],
                              progress=True)

    remove_layer(state_dict, 'Mixed_7')
    remove_layer(state_dict, 'AuxLogits')
    remove_layer(state_dict, 'fc.')

    model.load_state_dict(state_dict, strict=False)
    return model

def _inception_v3(architecture_type, pretrained=False, pretrained_path=None,**kwargs):
    unfreeze_layer = kwargs['unfreeze_layer']
    model = {'cam': InceptionCam}[architecture_type](**kwargs) 
    
    if kwargs['ft_ckpt'] is not None:
        print(f'Loading Fine-tuned Checkpoint: {kwargs["ft_ckpt"]}')
        print(f'Load checkpoint to head1') 
        
        state_dict = torch.load(kwargs['ft_ckpt'])['state_dict'] 
        new_state_dict = {} 

        for key, value in state_dict.items():
            if key.startswith('module.'): 
                new_key = key.replace('module.', '') 
            else:
                new_key = key
            new_state_dict[new_key] = value
            
        model.load_state_dict(new_state_dict, strict=False) 
    else:
        if pretrained:
            model = load_pretrained_model(model, pretrained_path)
    return model 
            
def inception_v3(architecture_type, pretrained=False, pretrained_path=None,
                 **kwargs):
    return _inception_v3(architecture_type, pretrained, pretrained_path, **kwargs)
