"""Shared train/inference preprocessing. Crops are letterboxed to a square (no aspect distortion, no edge cropping) so training and serving see identical inputs."""
from PIL import Image
MEAN, STD, FILL = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], (124, 116, 104)
def pad_square(img, fill=FILL):
    w, h = img.size; s = max(w, h); c = Image.new("RGB", (s, s), fill); c.paste(img.convert("RGB"), ((s - w) // 2, (s - h) // 2)); return c
def eval_tf(size=224):
    from torchvision import transforms as T
    return T.Compose([T.Lambda(pad_square), T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
def train_tf(size=224):
    from torchvision import transforms as T
    return T.Compose([T.Lambda(pad_square), T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)), T.RandomHorizontalFlip(), T.RandomRotation(20, fill=FILL),
                      T.ColorJitter(0.25, 0.25, 0.25, 0.03), T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2), T.ToTensor(), T.Normalize(MEAN, STD), T.RandomErasing(p=0.25, scale=(0.02, 0.15))])
