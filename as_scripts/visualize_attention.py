import os
from PIL import Image
import shlex

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, resnet18
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

from trainer import parser
from novas.models.wavenet import WaveNet

arg_string = "--version wavenet_synthetic2_pose_mag_1spk_remove_delay2 --model wavenet --dataset synthetic --use-rgb --num-channel 2 --n-gpus 8 --num-node 4 --batch-size 8 --gpu-mem32 --decode-wav --slurm --max-epochs 60 --use-tgt-pose --mag --one-speaker --remove-delay --auto-resume --use-intermediate-loss"
args = parser.parse_args(shlex.split(arg_string))
args.from_pretrained = os.path.join(args.model_dir, args.version, f'best_val.ckpt')
wavenet = WaveNet(args)
wavenet.load_weights(torch.load(args.from_pretrained, map_location='cpu'))
model = nn.Sequential(*wavenet.visual_encoder, nn.Flatten())
target_layers = [model[7][-1]]
rgb_img = np.array(Image.open('/private/home/changanvr/Projects/audio_synthesis/data/models/novas/wavenet_synthetic_v5/images_test/0-src-rgb.png'))[..., :-1] / 255.0
input_tensor = torch.tensor(rgb_img)
input_tensor = input_tensor.permute(-1, 0, 1).float().unsqueeze(0)
print(input_tensor.shape)

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

targets = [0]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
