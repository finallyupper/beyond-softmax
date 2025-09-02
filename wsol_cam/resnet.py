"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url

from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights

__all__ = ['resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

_ADL_POSITION = [[], [], [], [0], [0, 2]]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCam(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 large_feature_map=False, **kwargs):
        super(ResNetCam, self).__init__()

        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.last_layer = kwargs['last_layer'] 
        self.unfreeze_layer = kwargs['unfreeze_layer']
        self.model_structure = kwargs['model_structure'] 

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_l3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.last_layer == 'fc':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        if self.model_structure == 'b2':
            if self.last_layer == 'fc':
                self.fc2 = nn.Linear(512 * block.expansion, num_classes)

        if kwargs['init_weights']:
            initialize_weights(self.modules(), init_mode='xavier')

        if self.unfreeze_layer != 'all':
            self.freeze_layers() 

        if kwargs['debug']:
            self.check_params() 

    def forward(self, x, labels=None, return_cam=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        
        if self.last_layer == 'fc':
            logits = self.fc(pre_logit) 
        else:
            logits = self.conv(pre_logit) 
        results = {'logits': logits} 
        

        if return_cam:
            feature_map = x.detach().clone()

            if labels is None: 
                labels = logits.argmax(dim=1)
                results['labels'] = labels

            cam_weights = self.fc.weight[labels] 
            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            
            results['cams'] = cams 

        if self.model_structure == 'b2':
            if self.last_layer == 'fc': 
                logits2 = self.fc2(pre_logit)
                results['logits2'] = logits2 

            if return_cam:
                feature_map = x.detach().clone()

                cam_weights = self.fc2.weight[labels] 

                cam_weights = torch.where(torch.gt(cam_weights, 0.), cam_weights, torch.zeros_like(cam_weights))

                cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *feature_map).mean(1, keepdim=False)
                results['cams2'] = cams
        return results

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

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
            if self.unfreeze_layer in ['fc', 'fc2', 'conv', 'conv2']:
                types = [self.unfreeze_layer + ".weight", self.unfreeze_layer + ".bias"] # fc.weight, fc.bias, fc2.weight, fc2.bias, ...
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
  

def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


def align_layer(state_dict):
    keys = [key for key in sorted(state_dict.keys())]
    for key in reversed(keys):
        move = 0
        if 'layer' not in key:
            continue
        key_sp = key.split('.')
        layer_idx = int(key_sp[0][-1])
        block_idx = key_sp[1]
        if not _ADL_POSITION[layer_idx]:
            continue

        for pos in reversed(_ADL_POSITION[layer_idx]):
            if pos < int(block_idx):
                move += 1

        key_sp[1] = str(int(block_idx) + move)
        new_key = '.'.join(key_sp)
        state_dict[new_key] = state_dict.pop(key)
    return state_dict


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'layer3.0.', 'SPG_A1.0.')
    state_dict = replace_layer(state_dict, 'layer3.1.', 'SPG_A2.0.')
    state_dict = replace_layer(state_dict, 'layer3.2.', 'SPG_A2.1.')
    state_dict = replace_layer(state_dict, 'layer3.3.', 'SPG_A2.2.')
    state_dict = replace_layer(state_dict, 'layer3.4.', 'SPG_A2.3.')
    state_dict = replace_layer(state_dict, 'layer3.5.', 'SPG_A2.4.')
    return state_dict


def load_pretrained_model(model, wsol_method, path=None, **kwargs):
    strict_rule = True

    if path:
        state_dict = torch.load(os.path.join(path, 'resnet50.pth'))
    else:
        state_dict = load_url(model_urls['resnet50'], progress=True)

    if wsol_method == 'adl':
        state_dict = align_layer(state_dict)
    elif wsol_method == 'spg':
        state_dict = batch_replace_layer(state_dict)

    if kwargs['dataset_name'] != 'ILSVRC' or wsol_method in ('acol', 'spg'):
        state_dict = remove_layer(state_dict, 'fc')
        strict_rule = False

    try:
        model.load_state_dict(state_dict, strict=strict_rule)
    except:
        model.load_state_dict(state_dict, strict=False)
    return model

def _resnet50(architecture_type, pretrained, pretrained_path, **kwargs):
    unfreeze_layer = kwargs['unfreeze_layer']  
    model = {'cam': ResNetCam}[architecture_type](Bottleneck, [3, 4, 6, 3], **kwargs)
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
            if unfreeze_layer in ['fc', 'conv']:
                if new_key.startswith('fc.'):
                    new_key = new_key.replace('fc.', 'fc2.')
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False) 
    else:   
        if pretrained:
            print(f' Loading Pretrained Checkpoint')
            model = load_pretrained_model(model, architecture_type, path=pretrained_path, **kwargs)
    return model
def resnet50(architecture_type, pretrained=False, pretrained_path=None,
             **kwargs):
    return _resnet50(architecture_type, pretrained, pretrained_path, **kwargs)
