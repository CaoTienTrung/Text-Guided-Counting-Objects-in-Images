import torch
import re
import os
import numpy as np
import cv2
import json

from tqdm import tqdm
from icecream import  ic
from transformers import AutoTokenizer, AutoModel
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from PIL import Image
from collections import Counter
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


#===================================== TRANSFORM ===================================
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

class Transform:
    def __init__(self, image_size=None):
        self.image_size = image_size 


    def transform_from_PIL(self):
        transform = transforms.Compose([
            # transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def transform_from_ndarray(self):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        return transform


#===================================== INTERNVL25 ===================================
class ImageCaptioner:
    def __init__(self, config, device):

        self.config = config
        self.device = device
        self.transform = Transform(image_size=448)
        #~ Load model
        self.load_model()

    def convert_image_type(self, image):
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        elif type(image) == torch.Tensor:
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        return image

    def load_model(self):
        model_path = self.config["model_path"]
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            local_files_only=True  # Force to use local files only
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=False,
            local_files_only=True  # Force to use local files only
        )

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=10, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def tokenize_image(self, image, max_num=16):
        image = self.convert_image_type(image).convert("RGB")
        images = self.dynamic_preprocess(
            image=image, 
            use_thumbnail=True, 
            max_num=max_num,
            image_size=448
        )
        transform = self.transform.transform_from_PIL()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(self.device)
        return pixel_values
    
    def caption(self, image, prompt):
        pixel_values = self.tokenize_image(image, max_num=16).to(torch.float16).to(self.device)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        # question = "<image>\nPlease describe the image shortly."
        prefix = "<image>\n"
        response = self.model.chat(self.tokenizer, pixel_values, prefix + prompt, generation_config)
        return response
    
    def batch_caption(self, images, prompt):
        batch_size = len(images)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        
        pixel_values = [self.tokenize_image(image, max_num=12) for image in images]
        image_counts = [pixel_value.size(0) for pixel_value in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0).to(torch.float16).to(self.device)
        
        questions = [prompt] * len(image_counts)
        responses = self.model.batch_chat(
            self.tokenizer, 
            pixel_values, 
            num_patches_list=image_counts,
            questions=questions,
            generation_config=generation_config
        )
        return responses


#---- Load json
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        json_content = json.load(file)
        return json_content
    
def save_description(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(content))

def check_valid_description(ocr_description):
    all_items = ocr_description.split() 
    counter = Counter(all_items)
    top_element, top_freq = counter.most_common(1)[0]
    top_element, top_freq
    if top_freq / len(all_items) > 0.75:
        return False
    else:
        return True

#=====================================INTERNVL3===================================
PROMPT = "<image>\nMô tả thông tin tổng quát nội dung của infographic dựa trên các thông tin văn bản và hình ảnh xuất hiện trong infographic thành đoạn văn dài từ 1 đến 3 câu."

if __name__=="__main__":
    # device = "cuda:7"
    device = "cuda:0"
    config = {"model_path": "/datastore/npl/ViInfographicCaps/model/InternVL2_5-1B"}
    
    # Init Model
    captioner = ImageCaptioner(
        config=config,
        device=device
    )

    # Load Image
    train_path = "/datastore/npl/ViInfographicCaps/data/data/new_split_format/train_new.json"
    val_path = "/datastore/npl/ViInfographicCaps/data/data/new_split_format/val_new.json"
    test_path = "/datastore/npl/ViInfographicCaps/data/data/new_split_format/test_new.json"

    train_json = load_json(train_path)
    val_json = load_json(val_path)
    test_json = load_json(test_path)

    # Extract OCR
    # names = ["train", "val", "test"]
    names = ["test"]
    IMAGE_DIR = "/datastore/npl/ViInfographicCaps/data/images"
    SAVE_DIR = "/datastore/npl/ViInfographicCaps/save_inference/internvl2_5_1B_captioning"
    BATCH_SIZE = 4

    for name in names:
        if name=="train":
            path = "/datastore/npl/ViInfographicCaps/data/data/new_split_format/train_new.json"
        elif name=="val":
            path = "/datastore/npl/ViInfographicCaps/data/data/new_split_format/val_new.json"
        else:
            path = "/datastore/npl/ViInfographicCaps/data/data/new_split_format/test_new.json"

        json_file = load_json(path)
        img_ids = json_file.keys()
        img_names = [f"{img_id}.png" for img_id in img_ids]
        ic(len(img_names))

        all_extracted_image_ids = [image_name.split(".")[0] for image_name in os.listdir(SAVE_DIR)]
        img_ids = [img_id for img_id in img_ids if img_id not in all_extracted_image_ids]
        for start_index in tqdm(range(0, len(img_ids), BATCH_SIZE), desc="Captioning ..."):
            batch_image_ids = list(img_ids)[start_index:start_index + BATCH_SIZE]
            ic(batch_image_ids)
            batch_image_names = [f"{img_id}.png" for img_id in batch_image_ids]
            batch_image_paths = [os.path.join(IMAGE_DIR, img_name) for img_name in batch_image_names]
            batch_ids = [
                idx
                for idx, image_id in enumerate(batch_image_ids) 
                if image_id not in all_extracted_image_ids
            ]

            batch_image_paths = np.array(batch_image_paths)[batch_ids]
            batch_image_names = np.array(batch_image_names)[batch_ids]

            if len(batch_image_paths) == 0:
                continue
            
            batch_images = [np.array(cv2.imread(image_path)) for image_path in batch_image_paths]
            
            #~ Extract OCR
            ocr_descriptions = captioner.batch_caption(batch_images, PROMPT)
            for name, description in zip(batch_image_names, ocr_descriptions):
                id = name.split(".")[0]
                save_path = os.path.join(SAVE_DIR, f"{id}.txt")  
                if check_valid_description(description):
                    save_description(save_path, description)
                else:
                    ic(f"ID {id} failed to extract")