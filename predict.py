import os, sys, json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH   = r"D:\boxart_cnn\models\best_model.pth"
CLASSES_PATH = r"D:\boxart_cnn\results\classes.json"
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image_path):
    if not os.path.exists(image_path):
        print(f"image not found: {image_path}")
        return

    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        top_k = torch.topk(probs, NUM_CLASSES)

    print(f"\nimage: {os.path.basename(image_path)}")
    print(f"prediction: {CLASS_NAMES[top_k.indices[0].item()]}")
    print(f"confidence: {top_k.values[0].item()*100:.2f}%\n")
    print("all classes:")
    for i in range(NUM_CLASSES):
        idx  = top_k.indices[i].item()
        conf = top_k.values[i].item() * 100
        bar  = "█" * int(conf / 5)
        print(f"  {CLASS_NAMES[idx]:<25} {conf:>6.2f}%  {bar}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python predict.py <path_to_image>")
        print("example: python predict.py C:\\Users\\Francisc\\Desktop\\game.jpg")
    else:
        predict(sys.argv[1])