import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
import clip
from torch.utils.data import Dataset
from PIL import Image

# =========================
#  FFN & Adapter
# =========================

class ImageFFN(nn.Module):
    """
    FFN đặt trên image embedding 512-d của CLIP.
    (MLP nhỏ có BN + Dropout như trong paper)
    """
    def __init__(self, dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class TextAdapter(nn.Module):
    """
    Adapter cho text embedding 512-d của CLIP.
    """
    def __init__(self, dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, D)
        h = self.act(self.fc1(x))
        h = self.dropout(h)
        h = self.act(self.fc2(h))
        h = self.bn(h)
        h = self.dropout(h)
        out = self.fc3(h)
        return out
    
@dataclass
class Stage1Config:
    # dùng openai/clip ViT-B/16
    clip_name: str = "ViT-B/16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    margin: float = 1.0
    batch_size: int = 32
    num_workers: int = 4
    lr_ffn: float = 1e-4
    lr_adapter: float = 1e-4
    epochs_ffn: int = 200
    epochs_adapter: int = 200

    # Chỗ lưu checkpoint
    checkpoint_path: str = "/content/drive/MyDrive/CS331/checkpoint/richcount_stage1_fsc147.pt"
    
# =========================
#  Stage 1 model
# =========================

class RichCountStage1(nn.Module):
    """
    Stage 1: Visual–Text Alignment trên CLIP.
    - Dùng clip.load("ViT-B/16")
    - Thêm FFN cho image
    - Thêm Adapter cho text
    - Train bằng contrastive loss
    """

    def __init__(self, cfg: Stage1Config):
        super().__init__()
        self.cfg = cfg

        # Load CLIP
        self.clip_model, self.clip_preprocess = clip.load(
            cfg.clip_name,
            device=cfg.device,
            jit=False
        )

        self.clip_model = self.clip_model.float()

        dim = self.clip_model.visual.output_dim  # thường = 512

        self.ffn = ImageFFN(dim, dim)
        self.adapter = TextAdapter(dim, dim)

        self.to(cfg.device)

    # ---------- freeze / unfreeze ----------

    def freeze_clip(self):
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def freeze_ffn(self):
        for p in self.ffn.parameters():
            p.requires_grad = False

    def freeze_adapter(self):
        for p in self.adapter.parameters():
            p.requires_grad = False

    def unfreeze_ffn(self):
        for p in self.ffn.parameters():
            p.requires_grad = True

    def unfreeze_adapter(self):
        for p in self.adapter.parameters():
            p.requires_grad = True

    # ---------- encoding ----------

    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        """
        images: list[PIL.Image]
        -> tensor (B, 3, 224, 224) đã normalize theo CLIP
        """
        tensors = [self.clip_preprocess(im) for im in images]
        pixel_values = torch.stack(tensors, dim=0).to(self.cfg.device)
        return pixel_values

    def preprocess_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts).to(self.cfg.device)
        return tokens

    def get_image_emb_base(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (B, D)
        img_emb = self.clip_model.encode_image(pixel_values)
        img_emb = F.normalize(img_emb, dim=-1)
        return img_emb

    def get_text_emb_base(self, tokens: torch.Tensor) -> torch.Tensor:
        txt_emb = self.clip_model.encode_text(tokens)
        txt_emb = F.normalize(txt_emb, dim=-1)
        return txt_emb

    def encode_image_with_ffn(self, pixel_values: torch.Tensor) -> torch.Tensor:
        base = self.get_image_emb_base(pixel_values)
        out = self.ffn(base)
        return out

    def encode_text_raw(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.get_text_emb_base(tokens)

    def encode_text_with_adapter(self, tokens: torch.Tensor) -> torch.Tensor:
        base = self.get_text_emb_base(tokens)
        out = self.adapter(base)
        return out

    # ---------- contrastive loss ----------

    def contrastive_loss(
        self,
        img_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        margin: float
    ) -> torch.Tensor:
        """
        img_emb, pos_emb, neg_emb: (B, D)
        Loss giống Eq.(5)(6) trong paper.
        """
        pos_dist = torch.norm(img_emb - pos_emb, dim=-1)  # (B,)
        neg_dist = torch.norm(img_emb - neg_emb, dim=-1)  # (B,)

        loss_pos = pos_dist.pow(2)
        loss_neg = F.relu(margin - neg_dist).pow(2)

        loss = 0.5 * (loss_pos + loss_neg).mean()
        return loss
# =========================
#  Eval
# =========================

@torch.no_grad()
def evaluate_stage1(
    cfg: Stage1Config,
    model: RichCountStage1,
    loader: DataLoader,
    split: str = "val",
    use_adapter: bool = True
):
    model.eval()
    total = correct = 0
    dpos = dneg = 0.0

    for images, pos, neg in tqdm(loader, desc=f"Eval {split}"):
        pv = model.preprocess_images(images)
        pt = model.preprocess_texts(pos)
        nt = model.preprocess_texts(neg)

        img = model.encode_image_with_ffn(pv)

        if use_adapter:
            pos_e = model.encode_text_with_adapter(pt)
            neg_e = model.encode_text_with_adapter(nt)
        else:
            pos_e = model.encode_text_raw(pt)
            neg_e = model.encode_text_raw(nt)

        dp = torch.norm(img - pos_e, dim=-1)
        dn = torch.norm(img - neg_e, dim=-1)

        correct += (dp < dn).sum().item()
        total += dp.size(0)
        dpos += dp.sum().item()
        dneg += dn.sum().item()

    acc = correct / total
    avg_pos = dpos / total
    avg_neg = dneg / total
    gap = avg_neg - avg_pos

    print(f"[{split}] acc={acc:.4f}  d_pos={avg_pos:.4f}  d_neg={avg_neg:.4f}  gap={gap:.4f}")

    return {
        "acc": acc,
        "avg_pos": avg_pos,
        "avg_neg": avg_neg,
        "gap": gap,
    }

# =========================
#  Save / load checkpoint
# =========================

def save_checkpoint(model: RichCountStage1, cfg: Stage1Config, path: str):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
        },
        path
    )
    print(f"💾 Saved checkpoint: {path}")


def load_checkpoint(model: RichCountStage1, path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"✅ Loaded checkpoint from: {path}")
    return ckpt


# =========================
#  Training FFN & Adapter
# =========================

def train_ffn(
    cfg: Stage1Config,
    model: RichCountStage1,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ckpt_dir: str="ckpts_ffn"
    ):
    """
    Phase 1: Train FFN.
    - Freeze CLIP
    - Freeze Adapter
    - Text dùng embedding CLIP gốc
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    model.freeze_clip()
    model.unfreeze_ffn()
    model.freeze_adapter()

    optimizer = torch.optim.Adam(model.ffn.parameters(), lr=cfg.lr_ffn)

    best_gap = -1e9
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(cfg.epochs_ffn):
        model.train()
        total_loss = 0.0

        for step, (images, pos_texts, neg_texts) in enumerate(tqdm(train_loader, desc=f"FFN Epoch {epoch+1}")):
            pixel_values = model.preprocess_images(images)
            pos_tokens = model.preprocess_texts(pos_texts)
            neg_tokens = model.preprocess_texts(neg_texts)

            img_emb = model.encode_image_with_ffn(pixel_values)      # FFN có grad
            pos_emb = model.encode_text_raw(pos_tokens)              # text gốc CLIP
            neg_emb = model.encode_text_raw(neg_tokens)

            loss = model.contrastive_loss(img_emb, pos_emb, neg_emb, cfg.margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[FFN] Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")
        # 🔹 EVAL DEV
        metrics = evaluate_stage1(cfg, model, val_loader, split="val", use_adapter=False)

        # # 🔹 SAVE CHECKPOINT (EVERY EPOCH)
        # ckpt_path = os.path.join(ckpt_dir, f"ffn_epoch_{epoch+1}.pt")
        # save_checkpoint(model, cfg, ckpt_path)
        save_checkpoint(model, cfg, os.path.join(ckpt_dir, "last.pt"))

        # 🔹 TRACK BEST
        if metrics["gap"] > best_gap:
            best_gap = metrics["gap"]
            # best_ckpt_path = ckpt_path
            # print(f"New BEST FFN @ epoch {epoch+1} (gap={best_gap:.4f})")
            save_checkpoint(model, cfg, best_ckpt_path)
            print(f"New BEST saved -> {best_ckpt_path} (gap={best_gap:.4f})")

    print(f"\nBest FFN checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def train_adapter(
    cfg: Stage1Config,
    model: RichCountStage1,
    train_loader: DataLoader,
    val_loader: DataLoader,
    best_ffn_ckpt: str,
    ckpt_dir: str = "ckpts_adapter"):
    """
    Phase 2: Train Adapter.
    - Freeze CLIP
    - Freeze FFN
    - Only Adapter trainable
    """

    os.makedirs(ckpt_dir, exist_ok=True)

    # 🔹 LOAD BEST FFN
    load_checkpoint(model, best_ffn_ckpt, cfg.device)

    model.freeze_clip()
    model.freeze_ffn()
    model.unfreeze_adapter()

    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=cfg.lr_adapter)

    best_gap = -1e9
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(cfg.epochs_adapter):
        model.train()
        total_loss = 0.0

        for step, (images, pos_texts, neg_texts) in enumerate(tqdm(train_loader, desc=f"Adapter Epoch {epoch+1}")):
            pixel_values = model.preprocess_images(images)
            pos_tokens = model.preprocess_texts(pos_texts)
            neg_tokens = model.preprocess_texts(neg_texts)

            with torch.no_grad():
                img_emb = model.encode_image_with_ffn(pixel_values)  # fixed

            pos_emb = model.encode_text_with_adapter(pos_tokens)     # adapter train
            neg_emb = model.encode_text_with_adapter(neg_tokens)

            loss = model.contrastive_loss(img_emb, pos_emb, neg_emb, cfg.margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Adapter] Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")

        # 🔹 eval dev
        metrics = evaluate_stage1(cfg, model, val_loader, split="val", use_adapter=True)

        # # 🔹 save ckpt
        # ckpt_path = os.path.join(ckpt_dir, f"adapter_epoch_{epoch+1}.pt")
        # save_checkpoint(model, cfg, ckpt_path)

        # 🔹 track best
        if metrics["gap"] > best_gap:
            best_gap = metrics["gap"]
            # best_ckpt_path = ckpt_path
            # print(f"New BEST Adapter @ epoch {epoch+1} (gap={best_gap:.4f})")
            save_checkpoint(model, cfg, best_ckpt_path)
            print(f"New BEST saved -> {best_ckpt_path} (gap={best_gap:.4f})")

    print(f"\nBest Adapter checkpoint: {best_ckpt_path}")
    return best_ckpt_path

# =========================
#  Main
# =========================

def main():
    cfg = Stage1Config()
    print("Using device:", cfg.device)

    root = "FSC_147"

    # train / val datasets
    train_dataset = FSC147Stage1Dataset(root=root, split="train", text_type="text")
    val_dataset = FSC147Stage1Dataset(root=root, split="val", text_type="text")
    test_dataset = FSC147Stage1Dataset(root=root, split="test", text_type="text")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_stage1,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_stage1,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_stage1,
        drop_last=False,
    )

    # Model
    model = RichCountStage1(cfg)

    # Phase 1: train FFN
    print("========== TRAINING FFN ==========")
    best_ffn_ckpt = train_ffn(cfg, model, train_loader, val_loader, "checkpoint/ckpts_ffn")

    # Phase 2: train Adapter
    print("========== TRAINING ADAPTER ==========")
    best_adapter_ckpt = train_adapter(cfg, model, train_loader, val_loader, best_ffn_ckpt, "checkpoint/ckpts_adapter")

    # Load best adapter
    load_checkpoint(model, best_adapter_ckpt, cfg.device)

    save_checkpoint(model, cfg, cfg.checkpoint_path)
    print(f"✅ Saved final Stage 1 checkpoint to {cfg.checkpoint_path}")

    model_infer = RichCountStage1(cfg)
    load_checkpoint(model_infer, cfg.checkpoint_path, cfg.device)

    evaluate_stage1(cfg, model_infer, test_loader, split="test", use_adapter=True)


if __name__ == "__main__":
    cfg = Stage1Config()
    model_infer = RichCountStage1(cfg)
    model_infer = load_checkpoint(model_infer, cfg.checkpoint_path, cfg.device)
    main()
