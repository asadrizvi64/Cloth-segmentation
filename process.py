from network import U2NET

import os
from PIL import Image, ImageChops
import cv2
import gdown
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from options import opt


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"




def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)


def generate_mask(input_image, styled_layer, net, palette, device='cpu'):
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    alpha_out_dir = os.path.join(opt.output, 'alpha')
    cloth_seg_out_dir = os.path.join(opt.output, 'cloth_seg')

    os.makedirs(alpha_out_dir, exist_ok=True)
    os.makedirs(cloth_seg_out_dir, exist_ok=True)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    classes_to_save = []

    # Check which classes are present in the image
    for cls in range(1, 4):  # Exclude background class (0)
        if np.any(output_arr == cls):
            classes_to_save.append(cls)

    # Save alpha masks
    for cls in classes_to_save:
        alpha_mask = (output_arr == cls).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
        alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
        alpha_mask_path = os.path.join(alpha_out_dir, f'{cls}.png')
        alpha_mask_img.save(alpha_mask_path)

    # Save final cloth segmentations
    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg.putpalette(palette)
    cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
    cloth_seg_path = os.path.join(cloth_seg_out_dir, 'final_seg.png')
    cloth_seg.save(cloth_seg_path)

    # Load styled layer
    styled_layer = styled_layer.convert('RGBA')

    # Resize styled_layer to match img_size
    styled_layer_resized = styled_layer.resize(img_size, Image.BICUBIC)

    # Apply styled layer over the cloth segmentation
    cloth_seg_with_style = Image.alpha_composite(Image.new('RGBA', img_size, (0, 0, 0, 255)), styled_layer_resized)

    # Ensure both images have an alpha channel
    if cloth_seg_with_style.mode != 'RGBA':
        cloth_seg_with_style = cloth_seg_with_style.convert('RGBA')

    # Create a mask from the cloth segmentation
    mask = Image.new("L", img_size, 0)
    for cls in classes_to_save:
        alpha_mask_path = os.path.join(alpha_out_dir, f'{cls}.png')
        mask = ImageChops.add(mask, Image.open(alpha_mask_path).convert('L'))

    # Apply the mask to the cloth segmentation with style
    cloth_seg_with_style = Image.alpha_composite(Image.new('RGBA', img_size, (0, 0, 0, 0)), cloth_seg_with_style)
    cloth_seg_with_style = Image.alpha_composite(cloth_seg_with_style, styled_layer_resized)
    cloth_seg_with_style.putalpha(mask)

    # Overlay the cloth segmentation with style over the input image
    cloth_seg_with_style = cloth_seg_with_style.resize(img_size, Image.BICUBIC)  # Resize to match input image size
    result = Image.alpha_composite(input_image.convert('RGBA'), cloth_seg_with_style)

    return result



def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY"
        gdown.download(url, file_path, quiet=False)
        print("Model downloaded successfully.")
    else:
        print("Model already exists.")


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def main(args):
    device = 'cuda:0' if args.cuda else 'cpu'

    # Create an instance of your model
    model = load_seg_model(args.checkpoint_path, device=device)

    palette = get_palette(4)

    img = Image.open(args.image).convert('RGB')

    # Add code to load the styled layer image
    styled_layer_path = args.styled_layer
    styled_layer = Image.open(styled_layer_path).convert('RGBA')

    # Generate the cloth segmentation with style
    cloth_seg_with_style = generate_mask(img, styled_layer, net=model, palette=palette, device=device)
    
    # Save the final output
    output_path = os.path.join(opt.output, 'final_output.png')
    cloth_seg_with_style.save(output_path)
    print(f"Final output saved at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--styled_layer', type=str, help='Path to the styled layer image')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='model/cloth_segm.pth', help='Path to the checkpoint file')
    args = parser.parse_args()

    main(args)
