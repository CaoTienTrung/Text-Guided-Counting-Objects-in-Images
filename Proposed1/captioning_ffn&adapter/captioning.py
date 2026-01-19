import torch
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image
import cv2


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Transform:
    def __init__(self, image_size=448):
        self.image_size = image_size

    def transform_from_PIL(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])


class ImageCaptioner:
    def __init__(self, model_path, device):
        self.device = device
        self.transform = Transform(image_size=448)
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

    def convert_image_type(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[..., ::-1])  # BGR → RGB
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        return image

    def tokenize_image(self, image):
        image = self.convert_image_type(image).convert("RGB")

        transform = self.transform.transform_from_PIL()
        pixel_values = transform(image).unsqueeze(0)  # shape (1, 3, 448, 448)

        return pixel_values.to(self.device).to(torch.float16)

    def caption_single(self, image, prompt: str):
        pixel_values = self.tokenize_image(image)
        generation_config = dict(max_new_tokens=64, do_sample=False)

        prefix = "<image>\n"
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prefix + prompt,
            generation_config
        )
        return response


# PROMPT_TEMPLATE_EN = (
#     "Describe the {category} in this image in one English sentence."
#     "The sentence must start with the {category}."
#     "Mention its color, approximate shape, and where it is located in the scene."
#     '''If the {category} is not visible, respond with "The {category} is not visible in the image."'''

# )

PROMPT_TEMPLATE_EN = (
    "Describe the {category} in this image in one English sentence."
    "The sentence must start with the {category}."
    "Mention its color, approximate shape, and where it is located in the scene."
)



def describe_image_with_category(
    image_path: str,
    category: str,
    model_path: str = "OpenGVLab/InternVL2_5-1B",
    device: str = "cpu"
) -> str:

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image at: {image_path}")

    captioner = ImageCaptioner(model_path=model_path, device=device)
    prompt = PROMPT_TEMPLATE_EN.format(category=category)
    description = captioner.caption_single(img, prompt)
    return description

if __name__ == "__main__":
    img_path = "/content/Screenshot 2025-12-18 082715.png"
    category = "stawberries"
    desc = describe_image_with_category(img_path, category)
    print("Rich description:", desc)