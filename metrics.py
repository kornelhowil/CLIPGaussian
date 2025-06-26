import os
import sys
import random
from argparse import ArgumentParser
from PIL import Image
import torch
import clip
from CLIP import compose_text_with_templates
from torchvision import transforms
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

cropper = transforms.Compose([
    transforms.RandomCrop(128)
])

augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
    transforms.Resize(224)
])


def encode_image(model,
                 preprocess,
                 image_path,
                 patch=False,
                 device="cuda"
                 ):
    """ Encode image using CLIP model
    Args:
        model: CLIP model
        preprocess: Preprocessing function
        image_path: Path to the image
        device: Device to use for encoding
    Returns:
        image_feat: Encoded image feature
    """

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_proc = []
        if patch:
            for _ in range(128):
                target_crop = cropper(image)
                target_crop = augment(target_crop)
                img_proc.append(target_crop)
        else:
            img_proc.append(image)
        img_proc = torch.cat(img_proc, dim=0)
        image_feat = model.encode_image(img_proc)
        image_feat /= (image_feat.clone().norm(dim=-1, keepdim=True))
    return image_feat


def encode_text(model,
                text,
                device
                ):
    """ Encode text using CLIP model
    Args:
        model: CLIP model
        text: Text to encode
        device: Device to use for encoding
    Returns:
        text_feat: Encoded text feature
    """
    composed_text = compose_text_with_templates(text)
    tokens = clip.tokenize(composed_text).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(tokens)
        text_feat = text_feat.mean(axis=0, keepdim=True)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    return text_feat


def get_direction(emb1, emb2):
    """ Get direction vector between two embeddings
    Args:
        emb1: First embedding
        emb2: Second embedding
    Returns:
        direction: Direction vector
    """
    direction = emb1 - emb2
    return direction


class CLIPDirSim():
    def __init__(self,
                 model,
                 preprocess,
                 style_prompt=None,
                 style_image=None,
                 object_prompt="a Photo",
                 device="cuda"):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device
        with torch.no_grad():
            if style_prompt is not None:
                self.style_feat = encode_text(model, style_prompt, device)
            if style_image is not None:
                self.style_feat = encode_image(
                    model, preprocess, style_image, device=device)
            obj_feat = encode_text(model, object_prompt, device=device)
            self.style_dir = get_direction(self.style_feat, obj_feat)

    def __call__(self, gt_image_path, render_image_path, patch=False):
        files = os.listdir(render_image_path)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        scores_sum = 0
        images = 0
        with torch.no_grad():
            for filename in files:
                gt_path = os.path.join(gt_image_path, filename)
                render_path = os.path.join(render_image_path, filename)
                gt_feat = encode_image(
                    self.model, self.preprocess, gt_path, device=self.device)
                render_feat = encode_image(
                    self.model, self.preprocess, render_path, patch=patch, device=self.device)
                img_dir = get_direction(render_feat, gt_feat)
                style_dir = self.style_dir
                scores_sum += torch.cosine_similarity(
                    img_dir, style_dir, dim=1).cpu().numpy()[0]
                images += 1
        return 100 * scores_sum / images


class CLIPDirCons():
    def __init__(self,
                 model,
                 preprocess,
                 device="cuda"):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device

    def __call__(self, gt_image_path, render_image_path, k=1, patch=False):
        files = os.listdir(render_image_path)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        scores_sum = 0
        images = 0
        with torch.no_grad():
            for i, _ in enumerate(files):
                if i < len(files) - k:
                    gt_path = os.path.join(gt_image_path, files[i])
                    gt_path_next = os.path.join(gt_image_path, files[i+k])
                    render_path = os.path.join(render_image_path, files[i])
                    render_path_next = os.path.join(
                        render_image_path, files[i+k])
                    render_feat = encode_image(
                        self.model, self.preprocess, render_path, patch, device=self.device)
                    render_feat_next = encode_image(
                        self.model, self.preprocess, render_path_next, patch, device=self.device)
                    gt_feat = encode_image(
                        self.model, self.preprocess, gt_path, device=self.device)
                    gt_feat_next = encode_image(
                        self.model, self.preprocess, gt_path_next, device=self.device)
                    gt_dir = get_direction(gt_feat_next, gt_feat)
                    render_dir = get_direction(render_feat_next, render_feat)
                    scores_sum += torch.cosine_similarity(
                        gt_dir, render_dir, dim=1).cpu().numpy()[0]
                    images += 1
        return 100 * scores_sum / images


class CLIPScore():
    def __init__(self,
                 model,
                 preprocess,
                 style_prompt=None,
                 style_image=None,
                 device="cuda"):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device
        with torch.no_grad():
            if style_prompt is not None:
                self.style_feat = encode_text(
                    model, style_prompt, device)
            if style_image is not None:
                self.style_feat = encode_image(
                    model, preprocess, style_image, device=device)

    def __call__(self, render_image_path, patch=False):
        files = os.listdir(render_image_path)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        scores_sum = 0
        images = 0
        with torch.no_grad():
            for filename in files:
                render_path = os.path.join(render_image_path, filename)
                render_feat = encode_image(
                    self.model, self.preprocess, render_path, patch, device=self.device)
                scores_sum += torch.cosine_similarity(
                    render_feat, self.style_feat, dim=1).cpu().numpy()[0]
                images += 1
        return 100 * scores_sum / images


class CLIPF():
    def __init__(self,
                 model,
                 preprocess,
                 device="cuda"):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device

    def __call__(self, gt_image_path, render_image_path, patch=False):
        files = os.listdir(render_image_path)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        scores_sum_render = 0
        scores_sum_gt = 0
        images = 0
        with torch.no_grad():
            for i, _ in enumerate(files):
                if i < len(files) - 1:
                    gt_path = os.path.join(gt_image_path, files[i])
                    gt_path_next = os.path.join(gt_image_path, files[i+1])
                    render_path = os.path.join(render_image_path, files[i])
                    render_path_next = os.path.join(
                        render_image_path, files[i+1])
                    render_feat = encode_image(
                        self.model, self.preprocess, render_path, patch, device=self.device)
                    render_feat_next = encode_image(
                        self.model, self.preprocess, render_path_next, patch, device=self.device)
                    gt_feat = encode_image(self.model, self.preprocess,
                                           gt_path, device=self.device)
                    gt_feat_next = encode_image(
                        self.model, self.preprocess, gt_path_next, device=self.device)
                    scores_sum_render += torch.cosine_similarity(
                        render_feat, render_feat_next, dim=1).cpu().numpy()[0]
                    scores_sum_gt += torch.cosine_similarity(
                        gt_feat, gt_feat_next, dim=1).cpu().numpy()[0]
                    images += 1
        clip_f_gt = scores_sum_gt / images
        cli_f_render = scores_sum_render / images
        return 100 * cli_f_render / clip_f_gt


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--render', type=str, default=None)
    parser.add_argument('--style_image', type=str, default=None)
    parser.add_argument('--style_prompt', type=str, default=None)
    parser.add_argument("--object_prompt", type=str, default="a Photo")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--interval", type=int, default=1)
    args = parser.parse_args(sys.argv[1:])

    clip_model, clip_preprocess = clip.load("ViT-L/14", device=args.device)

    clip_similarity = CLIPDirSim(clip_model, clip_preprocess,
                                   style_prompt=args.style_prompt,
                                   style_image=args.style_image,
                                   object_prompt=args.object_prompt,
                                   device=args.device)
    clip_consistency = CLIPDirCons(clip_model, clip_preprocess, device=args.device)
    clip_f = CLIPF(clip_model, clip_preprocess, device=args.device)
    clip_score = CLIPScore(clip_model, clip_preprocess,
                            style_prompt=args.style_prompt,
                            style_image=args.style_image,
                            device=args.device)
    print(
        f"CLIP directional consistency: {clip_consistency(args.gt, args.render, k=args.interval)}")
    print(
        f"CLIP_F (scaled): {clip_f(args.gt, args.render)}")
    print(
        f"CLIP Score: {clip_score(args.render)}")
    print(
        f"CLIP directional similarity: {clip_similarity(args.gt, args.render)}")
