"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

configs_dict = {
    'cam': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 512, 512, 512],
    }
}


class VggCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggCam, self).__init__()
        self.features = features 
        self.target_layers = [self.features[-1]] 

        dropout = kwargs.get('dropout', 0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.model_structure = kwargs['model_structure'] 
        self.unfreeze_layer = kwargs['unfreeze_layer']
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        if self.model_structure == 'b2':
            self.classifier_2 = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, num_classes),
            )

        if kwargs['init_weights']: 
            initialize_weights(self.modules(), init_mode='he')
        if self.unfreeze_layer != 'all':
            print(f'Freeze layers: {self.unfreeze_layer}')
            self.freeze_layers()

        self.check_params() 

    def forward(self, x, labels=None,):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        results = {'logits': logits} 

        if self.model_structure == 'b2':
            logits_2 = self.classifier_2(x) 
            results['logits2'] = logits_2 

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
            if self.unfreeze_layer in ['classifier', 'classifier_2']:
                types = [self.unfreeze_layer + f".{n}.weight" for n in [0, 3, 6]]  
                types += [self.unfreeze_layer + f".{n}.bias" for n in [0, 3, 6]]
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


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'features.17', 'SPG_A_1.0')
    state_dict = replace_layer(state_dict, 'features.19', 'SPG_A_1.2')
    state_dict = replace_layer(state_dict, 'features.21', 'SPG_A_1.4')
    state_dict = replace_layer(state_dict, 'features.24', 'SPG_A_2.0')
    state_dict = replace_layer(state_dict, 'features.26', 'SPG_A_2.2')
    state_dict = replace_layer(state_dict, 'features.28', 'SPG_A_2.4')
    return state_dict


def load_pretrained_model(model, architecture_type, path=None, **kwargs):
    strict_rule = True
    if path is not None:
        state_dict = torch.load(os.path.join(path, 'vgg16.pth'))
    else:
        state_dict = load_url(model_urls['vgg16'], progress=True)

    if architecture_type == 'spg':
        state_dict = batch_replace_layer(state_dict)

    if kwargs['dataset_name'] != 'ILSVRC': 
        state_dict = remove_layer(state_dict, 'classifier.')
        strict_rule = False
    if kwargs['model_structure'] == 'b2':
        strict_rule = False
    state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def make_layers(cfg, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # elif v == 'A':
        #     layers += [
        #         ADL(kwargs['adl_drop_rate'], kwargs['adl_drop_threshold'])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg16(architecture_type, pretrained=False, pretrained_path=None,
          **kwargs):
    config_key = '28x28' if kwargs['large_feature_map'] else '14x14'
    layers = make_layers(configs_dict[architecture_type][config_key], **kwargs)

    model = {'cam': VggCam}[architecture_type](layers, **kwargs)
    
    if kwargs['ft_ckpt'] is not None and kwargs['dataset_name'] != 'ILSVRC':
        unfreeze_layer = kwargs['unfreeze_layer']
        print(f'Load checkpoint from {kwargs["ft_ckpt"]}')
        print(f'Load checkpoint to head1')

        state_dict = torch.load(kwargs['ft_ckpt'])['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items(): 
            if key.startswith('module.'): 
                new_key = key.replace('module.', '') 
            else:
                new_key = key
                
            # if unfreeze_layer in ['classifier', 'classifier_2']:
            #     if new_key.startswith('classifier.'):
            #         new_key = new_key.replace('classifier.', 'classifier_2.')
            
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False) 
    else:
        if pretrained:
            print(f'Load pretrained model from {architecture_type}') 
            model = load_pretrained_model(model, architecture_type,path=pretrained_path, **kwargs)
    return model
