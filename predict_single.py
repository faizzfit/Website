import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance
from pathlib import Path
from datetime import datetime
import hashlib
import exifread

# === PATHS ===
base_dir = Path(__file__).resolve().parent.parent
input_path = base_dir / 'TestSingle' / '002.jpg'
model_path = base_dir / 'model.pth'

# === DEVICE ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Using device: {device}")

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === ELA FUNCTION ===
def apply_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    temp_path = str(image_path).replace('.jpg', '_temp.jpg')
    original.save(temp_path, 'JPEG', quality=quality)
    compressed = Image.open(temp_path)
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    os.remove(temp_path)
    return ela_image

# === MODEL ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        for idx, layer in enumerate(self.network):
            x = layer(x)
            if idx == 2:
                self.activation1 = x
            elif idx == 5:
                self.activation2 = x
            elif idx == 8:
                self.activation3 = x
        return x

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === LOAD IMAGE ===
ela_image = apply_ela(input_path)
ela_tensor = transform(ela_image).unsqueeze(0).to(device)

# === PREDICT ===
with torch.no_grad():
    output = model(ela_tensor)
    raw_score = output.item()
    label = 'Fake' if raw_score > 0.5 else 'Real'
    confidence = raw_score * 100 if label == 'Fake' else (1 - raw_score) * 100

print(f"\n✅ Prediction: {label} ({confidence:.2f}%)")

# === SAVE RESULT IMAGE ===
result_img = Image.open(input_path).convert('RGB')
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(result_img)
axs[0].set_title('Original')
axs[0].axis('off')
axs[1].imshow(ela_image)
axs[1].set_title('ELA Preview')
axs[1].axis('off')
plt.suptitle(f"Prediction: {label} ({confidence:.2f}%)")
plt.savefig(input_path.parent / f"{input_path.stem}_result_summary.png")
plt.close()

# === SAVE ELA IMAGE ===
ela_image.save(input_path.parent / f"{input_path.stem}_ela_preview.png")

# === SAVE METADATA ===
def extract_and_save_metadata(image_path):
    meta_path = image_path.parent / f"{image_path.stem}_metadata.txt"
    try:
        if not image_path.exists():
            print(f"❌ File not found: {image_path}")
            return

        pil_img = Image.open(image_path)
        stat = os.stat(image_path)

        def get_hash(file_path, algo='sha256'):
            h = hashlib.new(algo)
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    h.update(chunk)
            return h.hexdigest()

        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(f"Filename        : {image_path.name}\n")
            f.write(f"Format          : {pil_img.format}\n")
            f.write(f"Mode            : {pil_img.mode}\n")
            f.write(f"Dimensions      : {pil_img.width} x {pil_img.height} pixels\n")
            f.write(f"File Size       : {stat.st_size:,} bytes\n")
            f.write(f"Created Time    : {datetime.fromtimestamp(stat.st_ctime)}\n")
            f.write(f"Modified Time   : {datetime.fromtimestamp(stat.st_mtime)}\n")
            f.write(f"SHA-256 Hash    : {get_hash(image_path, 'sha256')}\n")
            f.write(f"MD5 Hash        : {get_hash(image_path, 'md5')}\n\n")
            f.write("=== EXIF Metadata ===\n")
            try:
                with open(image_path, 'rb') as img_file:
                    tags = exifread.process_file(img_file, details=False)
                if tags:
                    for tag in sorted(tags.keys()):
                        f.write(f"{tag}: {tags[tag]}\n")
                else:
                    f.write("No EXIF metadata found in this image.\n")
            except Exception as e:
                f.write(f"Error reading EXIF metadata: {e}\n")

        print(f"\n📁 Metadata saved to: {meta_path}\n")

    except Exception as e:
        print(f"❌ Failed to extract metadata: {e}")

extract_and_save_metadata(input_path)

# === SAVE CONV LAYER ACTIVATIONS ===
def save_activation_grid(activation, title, out_name):
    act = activation.squeeze(0).cpu().detach().numpy()
    fig, axes = plt.subplots(1, min(16, act.shape[0]), figsize=(15, 2))
    for i, ax in enumerate(axes):
        ax.imshow(act[i], cmap='viridis')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(input_path.parent / f"{input_path.stem}_{out_name}.png")
    plt.close()

save_activation_grid(model.activation1, "Layer Activations (Conv1)", "activation_conv1")
save_activation_grid(model.activation2, "Layer Activations (Conv2)", "activation_conv2")
save_activation_grid(model.activation3, "Layer Activations (Conv3)", "activation_conv3")

# === GRAD-CAM ===
def generate_gradcam(input_tensor, model, target_layer, output_name):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output.backward()

    grads = gradients[0].squeeze(0).cpu().detach().numpy()
    acts = activations[0].squeeze(0).cpu().detach().numpy()
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts[0].shape)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (result_img.width, result_img.height))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(result_img), 0.6, heatmap, 0.4, 0)

    plt.figure()
    plt.imshow(overlay)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    plt.savefig(input_path.parent / f"{input_path.stem}_gradcam_heatmap.png")
    plt.close()

    handle_fw.remove()
    handle_bw.remove()

generate_gradcam(ela_tensor, model, model.network[6], 'gradcam')
print(f"\n✅ All results saved to: {input_path.parent}\n")
