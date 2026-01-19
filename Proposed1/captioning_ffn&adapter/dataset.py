import os
import json
import random
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset

class FSC147Stage1Dataset(Dataset):
    """
    Dataset FSC-147 cho Stage 1:
    - Input: root folder chứa:
        root/
          images_384_VarV2/
          FSC_147/Train_Test_Val_FSC_147.json
          FSC_147/ImageClasses_FSC147.txt
    - Trả ra: (PIL.Image, pos_text, neg_text)
      pos_text  = class name của ảnh
      neg_text  = class name của ảnh khác (cố gắng khác class)
    """

    def __init__(self, root: str, split: str = "train", text_type: str = "class",):
        super().__init__()
        assert split in ["train", "val", "test"]
        assert text_type in ["class", "text"]
        self.root = root
        self.split = split
        self.text_type = text_type

        # Load split file
        split_path = os.path.join(root, "Train_Test_Val_FSC_147.json")
        with open(split_path, "r") as f:
            split_data = json.load(f)[split]
        self.im_list = [os.path.join(root, "images_384_VarV2", x) for x in split_data]

        # Load map: image_name -> class_name
        class_file = os.path.join(root, "ImageClasses_FSC147.txt")
        self.cls_dict = {}
        with open(class_file, "r", encoding="utf-8") as f:
            for line in f:
                name, cls = line.strip().split("\t")
                self.cls_dict[name] = cls

        # Load map: image_name -> describe_name
        describe_file = os.path.join(root, "DescribeObject_FSC147.txt")
        self.describe_dict = {}
        with open(describe_file, "r", encoding="utf-8") as f:
            for line in f:
                name, des = line.strip().split("\t")
                self.describe_dict[name] = des

        # Precompute class for each image & list of all (basename, cls)
        self.im_info = []
        for p in self.im_list:
            basename = os.path.basename(p)
            cls_name = self.cls_dict[basename]
            des_name = self.describe_dict[basename]
            self.im_info.append((p, basename, cls_name, des_name))

        print(f"[FSC147Stage1Dataset] Loaded {len(self.im_info)} images for split={split}")

    def __len__(self):
        return len(self.im_info)

    def __getitem__(self, idx: int):
        img_path, basename, cls_name, des_name = self.im_info[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        if self.text_type == "class":
            pos_text = cls_name
        else:
            pos_text = des_name

        # Negative text = class của ảnh khác (cố gắng khác class)
        neg_text = pos_text
        trial = 0
        while neg_text == pos_text and trial < 10:
            j = random.randint(0, len(self.im_info) - 1)
            _, _, cls_j, des_j = self.im_info[j]
            neg_text = cls_j if self.text_type == "class" else des_j
            trial += 1
        # nếu sau 10 lần vẫn trùng, đành chấp nhận (hiếm)

        return img, pos_text, neg_text


def collate_stage1(batch) -> Tuple[List[Image.Image], List[str], List[str]]:
    """
    batch: list of (img, pos_text, neg_text)
    """
    images = [b[0] for b in batch]
    pos_texts = [b[1] for b in batch]
    neg_texts = [b[2] for b in batch]
    return images, pos_texts, neg_texts
