"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import os
from os.path import dirname as ospd
import numpy as np
from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from util import t2n
from tqdm import tqdm 
from pytorch_grad_cam import *
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import scale_cam_image, get_target_width_height

def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float32)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float32) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam

def reshape_transform(tensor, height=14, width=14):
    """ the first element represents the class token, and the rest represent the 14x14 patches in the image. """
    result = tensor[:, 1:, :].reshape(tensor.size(0), # BATCH x 197 x 192 -> BATCH x 196 x 14 x 14
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, log_folder=None,
                 model_structure='vanilla', unfreeze_layer='all', method='gradcam', **kwargs):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.model_structure = model_structure 
        self.unfreeze_layer = unfreeze_layer
        self.scoremap_path = kwargs['scoremap_path'] 
        self.architecture_type = kwargs['architecture_type']
        self.alpha = kwargs['alpha'] 
        self.architecture = kwargs['architecture'] 
        methods = {
            "gradcam": GradCAM,
            "gradcam++": GradCAMPlusPlus,
            "xgradcam": XGradCAM,
            "layercam": LayerCAM,
        }
        self.method = method
        self.cam_algorithm = methods[method] if method != 'cam' else None

        if self.model_structure == 'b2' and self.method != 'scorecam':
            self.truncate_weight = True
            print("[Branch 2] Truncate weight is set to True.")
        else: 
            self.truncate_weight = False

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = [round(x * cam_curve_interval, 3) for x in range(int(1 / cam_curve_interval) + 1)]

        self.evaluator = {"OpenImages": MaskEvaluator,
                        "CUB": BoxEvaluator,
                        "ILSVRC": BoxEvaluator,
                        'CARS' : BoxEvaluator,
                        }[dataset_name](metadata=metadata,
                                        dataset_name=dataset_name,
                                        split=split,
                                        cam_threshold_list=cam_threshold_list,
                                        iou_threshold_list=iou_threshold_list,
                                        mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_gradcams(self, device, pred_correct_list=None):
        
        print("Computing and evaluating gradcams.")
        
        for batch_idx, (images, labels, image_ids) in tqdm(enumerate(self.loader), desc='Computing GradCAMs', total=len(self.loader)):
            batch_pred_correct = pred_correct_list[batch_idx]
            image_size = images.shape[2:]
            images = images.to(device)
            targets = [ClassifierOutputTarget(category) for category in labels] 

            """ 
            Define CAM algorithm and get CAMs.
            """
            if self.method == 'cam':
                pred_outputs = self.model(images, return_cam=True) # predicted logits from softmax head
                       
                if self.model_structure == 'vanilla':
                    cams = t2n(self.model(images, labels, return_cam=True)['cams'])   
                    pred_based_cams = t2n(pred_outputs['cams']) # get cam from single head
                else: 
                    cams = t2n(self.model(images, labels, return_cam=True)['cams2'])
                    pred_based_cams = t2n(pred_outputs['cams2']) # get cam from head2
                
                target_size = get_target_width_height(images)
                pred_based_cams = scale_cam_image(pred_based_cams, target_size) 

                pred_targets = pred_outputs['labels'].cpu().numpy() # get predicted labels from head1
                pred_targets = [ClassifierOutputTarget(category) for category in pred_targets]


            else:
                cam_algorithm = self.cam_algorithm(model=self.model, 
                                    target_layers=self.model.target_layers,
                                    device=device,)
                cam_algorithm.batch_size = len(images) 
                cams = cam_algorithm(input_tensor=images, targets=targets, truncate_weight=self.truncate_weight,) 

  
            for idx, (cam, image_id, image) in enumerate(zip(cams, image_ids,images)):
                # image : 3 x 224 x 224
                # cam : 14 x 14
                cam_resized = cv2.resize(cam, image_size, interpolation=cv2.INTER_CUBIC) 
                cam_normalized = normalize_scoremap(cam_resized) 

                if self.split in ('val', 'test'):            
                    cam_path = os.path.join(os.path.dirname(self.scoremap_path), image_id) 
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(os.path.join(cam_path), cam_normalized) 

                self.evaluator.accumulate(cam_normalized, image_id, batch_pred_correct[idx], image, self.method)
        return self.evaluator.compute()