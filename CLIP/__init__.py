import clip
import torch
from torchvision import models, transforms

from CLIP.template import imagenet_templates

from CLIP.utils.image_utils import img_normalize, clip_normalize, load_image


def load_model():
    VGG = models.vgg19(weights='DEFAULT').features
    VGG.to("cuda")

    clip_model, _ = clip.load('ViT-B/32', "cuda", jit=False)
    return clip_model


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]


def get_style_embedding(
        clip_model,
        style_prompt,
        style_image,
        object_prompt
):
    with torch.no_grad():
        if style_image is None:
            print(style_prompt)
            template_text = compose_text_with_templates(style_prompt, imagenet_templates)
            tokens = clip.tokenize(template_text).to("cuda")
            style_features = clip_model.encode_text(tokens).detach()
            style_features = style_features.mean(axis=0, keepdim=True)
            style_features /= style_features.norm(dim=-1, keepdim=True)
        else:
            style_image = load_image(style_image).to("cuda")
            style_features = clip_model.encode_image(clip_normalize(style_image))
            style_features /= (style_features.clone().norm(dim=-1, keepdim=True))

        template_source = compose_text_with_templates(object_prompt, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to("cuda")
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)

        style_direction = (style_features - text_source)
        style_direction /= style_direction.norm(dim=-1, keepdim=True)
        return style_direction