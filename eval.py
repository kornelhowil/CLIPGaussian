import os
import sys
from argparse import ArgumentParser
from PIL import Image
import torch
import clip
from template import imagenet_templates


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


class CLIP_similarity():
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
                style_prompt = compose_text_with_templates(
                    style_prompt, templates=imagenet_templates)
                tokens = clip.tokenize(style_prompt).to(device)
                style_feat = self.model.encode_text(tokens)
                style_feat = style_feat.mean(axis=0, keepdim=True)
                style_feat /= style_feat.norm(dim=-1, keepdim=True)
            if style_image is not None:
                style_image = self.preprocess(
                    Image.open(style_image)).unsqueeze(0).to(device)
                style_feat = model.encode_image(style_image)
                style_feat /= style_feat.norm(dim=-1, keepdim=True)

            obj_prompt = compose_text_with_templates(object_prompt,
                                                     templates=imagenet_templates)
            tokens = clip.tokenize(obj_prompt).to(device)
            obj_feat = self.model.encode_text(tokens)
            obj_feat = obj_feat.mean(axis=0, keepdim=True)
            obj_feat /= obj_feat.norm(dim=-1, keepdim=True)
            self.style_dir = (style_feat-obj_feat)
            self.style_dir /= self.style_dir.norm(dim=-1, keepdim=True)

    def __call__(self, gt_image_path, render_image_path):
        files = os.listdir(gt_image_path)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        scores_sum = 0
        images = 0
        with torch.no_grad():
            for filename in files:
                gt_path = os.path.join(gt_image_path, filename)
                render_path = os.path.join(render_image_path, filename)
                render = self.preprocess(Image.open(render_path)
                                         ).unsqueeze(0).to(args.device)
                gt = self.preprocess(Image.open(gt_path)
                                     ).unsqueeze(0).to(args.device)
                render_feat = model.encode_image(render)
                render_feat /= (render_feat.norm(dim=-1, keepdim=True))
                gt_feat = model.encode_image(gt)
                gt_feat /= (gt_feat.norm(dim=-1, keepdim=True))
                img_dir = (render_feat-gt_feat)
                img_dir /= img_dir.norm(dim=-1, keepdim=True)
                scores_sum += torch.cosine_similarity(
                    img_dir, self.style_dir, dim=1).cpu().numpy()[0]
                images += 1
        return scores_sum / images


class CLIP_consistency():
    def __init__(self,
                 model,
                 preprocess,
                 device="cuda"):
        self.model = model.to(device)
        self.preprocess = preprocess
        self.device = device

    def __call__(self, gt_image_path, render_image_path, k=1):
        files = os.listdir(gt_image_path)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        scores_sum = 0
        images = 0
        with torch.no_grad():
            for i in range(len(files)):
                if i < len(files) - k:
                    gt_path = os.path.join(gt_image_path, files[i])
                    gt_path_next = os.path.join(gt_image_path, files[i+k])
                    render_path = os.path.join(render_image_path, files[i])
                    render_path_next = os.path.join(
                        render_image_path, files[i+k])
                    render = self.preprocess(Image.open(render_path)
                                             ).unsqueeze(0).to(self.device)
                    render_next = self.preprocess(Image.open(render_path_next)
                                                  ).unsqueeze(0).to(self.device)
                    gt = self.preprocess(Image.open(gt_path)
                                         ).unsqueeze(0).to(self.device)
                    gt_next = self.preprocess(Image.open(gt_path_next)
                                              ).unsqueeze(0).to(self.device)
                    render_feat = model.encode_image(render)
                    render_feat /= (render_feat.norm(dim=-1, keepdim=True))
                    render_feat_next = model.encode_image(render_next)
                    render_feat_next /= (render_feat_next.norm(
                        dim=-1, keepdim=True))
                    gt_feat = model.encode_image(gt)
                    gt_feat /= (gt_feat.norm(dim=-1, keepdim=True))
                    gt_feat_next = model.encode_image(gt_next)
                    gt_feat_next /= (gt_feat_next.norm(dim=-1, keepdim=True))
                    gt_dir = (gt_feat_next-gt_feat)
                    gt_dir /= gt_dir.norm(dim=-1, keepdim=True)
                    render_dir = (render_feat_next-render_feat)
                    render_dir /= render_dir.norm(dim=-1, keepdim=True)
                    scores_sum += torch.cosine_similarity(
                        gt_dir, render_dir, dim=1).cpu().numpy()[0]
                    images += 1
        return scores_sum / images


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

    model, preprocess = clip.load("ViT-B/32", device=args.device)

    clip_similarity = CLIP_similarity(model, preprocess,
                                      style_prompt=args.style_prompt,
                                      style_image=args.style_image,
                                      object_prompt=args.object_prompt,
                                      device=args.device)
    clip_consistency = CLIP_consistency(model, preprocess, device=args.device)
    print(
        f"Consistency score: {clip_consistency(args.gt, args.render, args.interval)}")
    print(
        f"Similarity score: {clip_similarity(args.gt, args.render)}")
