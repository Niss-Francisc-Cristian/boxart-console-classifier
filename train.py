import os, json, random, time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR    = r"D:\boxart_cnn\data"
MODELS_DIR  = r"D:\boxart_cnn\models"
PLOTS_DIR   = r"D:\boxart_cnn\plots"
RESULTS_DIR = r"D:\boxart_cnn\results"

IMG_SIZE      = 224
BATCH_SIZE    = 32
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 25
SEED          = 42

random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {DEVICE}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")

for d in [MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=eval_tf)

CLASS_NAMES = train_ds.classes
NUM_CLASSES = len(CLASS_NAMES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

ga_idx    = random.sample(range(len(train_ds)), int(len(train_ds) * 0.1))
ga_loader = DataLoader(Subset(train_ds, ga_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"\nfound {NUM_CLASSES} classes: {CLASS_NAMES}")
print(f"train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}\n")

with open(os.path.join(RESULTS_DIR, "classes.json"), "w") as f:
    json.dump(CLASS_NAMES, f)

def get_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(DEVICE)

criterion = nn.CrossEntropyLoss()

print("running genetic algorithm to find best hyperparams...")
print("testing 4 individuals per generation, 3 generations total\n")

population = [
    {"lr": 10 ** random.uniform(-4, -2),
     "wd": 10 ** random.uniform(-5, -3)}
    for _ in range(4)
]

ga_fits     = []
best_params = None

for gen in range(3):
    results = []
    for ind in population:
        m = get_model()
        for p in m.parameters():
            p.requires_grad = False
        for p in m.fc.parameters():
            p.requires_grad = True

        opt = optim.Adam(m.fc.parameters(), lr=ind["lr"], weight_decay=ind["wd"])
        m.train()
        for i, (x, y) in enumerate(ga_loader):
            if i > 15: break
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            criterion(m(x), y).backward()
            opt.step()

        m.eval()
        correct = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i > 5: break
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += torch.sum(torch.max(m(x), 1)[1] == y)
        fit = (correct.double() / (6 * BATCH_SIZE)).item()
        results.append((fit, ind))
        print(f"gen {gen+1} - lr={ind['lr']:.4f}, wd={ind['wd']:.6f} -> accuracy: {fit*100:.1f}%")

    results.sort(key=lambda x: x[0], reverse=True)
    ga_fits.append(results[0][0])
    best_params = results[0][1]
    print(f"best of gen {gen+1}: {results[0][0]*100:.1f}%\n")

    p1, p2 = results[0][1], results[1][1]
    population = [p1, p2,
                  {"lr": p1["lr"] * random.uniform(0.8, 1.2), "wd": p1["wd"]},
                  {"lr": p2["lr"] * random.uniform(0.8, 1.2), "wd": p2["wd"]}]

print(f"best params found: lr={best_params['lr']:.4f}, wd={best_params['wd']:.6f}\n")

plt.figure()
plt.plot(range(1, 4), [f * 100 for f in ga_fits], marker="o", color="darkorange")
plt.title("GA Fitness per Generation")
plt.xlabel("Generation")
plt.ylabel("Fitness (%)")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "ga_fitness.png"), dpi=150)
plt.close()

print("starting training...")
print(f"stage 1: {STAGE1_EPOCHS} epochs with frozen backbone")
print(f"stage 2: {STAGE2_EPOCHS} epochs fine tuning everything\n")

model        = get_model()
hist         = {"ta": [], "va": [], "tl": [], "vl": []}
best_val_acc = 0.0

def run_epoch(model, loader, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += y.size(0)
    return total_loss / total, correct / total

print("stage 1 - training only the last layer for now")
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

opt1 = optim.Adam(model.fc.parameters(), lr=best_params["lr"], weight_decay=best_params["wd"])

for epoch in range(1, STAGE1_EPOCHS + 1):
    t0 = time.time()
    tl, ta = run_epoch(model, train_loader, opt1, train=True)
    vl, va = run_epoch(model, val_loader, train=False)
    elapsed = time.time() - t0
    hist["ta"].append(ta); hist["va"].append(va)
    hist["tl"].append(tl); hist["vl"].append(vl)
    print(f"  ep {epoch}/{STAGE1_EPOCHS} | train={ta*100:.2f}%  val={va*100:.2f}%  loss={vl:.4f}  {elapsed:.1f}s")

print("\nstage 2 - unfreezing everything and fine tuning")
for p in model.parameters():
    p.requires_grad = True

opt2 = optim.Adam(model.parameters(), lr=best_params["lr"] * 0.1, weight_decay=best_params["wd"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=STAGE2_EPOCHS, eta_min=1e-7)

for epoch in range(1, STAGE2_EPOCHS + 1):
    t0 = time.time()
    tl, ta = run_epoch(model, train_loader, opt2, train=True)
    vl, va = run_epoch(model, val_loader, train=False)
    scheduler.step()
    elapsed = time.time() - t0
    hist["ta"].append(ta); hist["va"].append(va)
    hist["tl"].append(tl); hist["vl"].append(vl)

    marker = ""
    if va > best_val_acc:
        best_val_acc = va
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
        marker = " (saved)"

    print(f"  ep {epoch}/{STAGE2_EPOCHS} | train={ta*100:.2f}%  val={va*100:.2f}%  loss={vl:.4f}  {elapsed:.1f}s{marker}")

print(f"\nbest validation accuracy was {best_val_acc*100:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(hist["ta"], label="Train"); ax1.plot(hist["va"], label="Validation")
ax1.axvline(x=STAGE1_EPOCHS - 0.5, color="gray", linestyle="--", alpha=0.7, label="stage 1->2")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.grid(True)

ax2.plot(hist["tl"], label="Train"); ax2.plot(hist["vl"], label="Validation")
ax2.axvline(x=STAGE1_EPOCHS - 0.5, color="gray", linestyle="--", alpha=0.7)
ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "training_curves.png"), dpi=150)
plt.show()

model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth")))

def evaluate_split(loader, split_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc        = (all_preds == all_labels).mean() * 100

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    with open(os.path.join(RESULTS_DIR, f"{split_name}_report.txt"), "w") as f:
        f.write(f"{split_name} results\naccuracy: {acc:.2f}%\n\n{report}")

    print(f"\n{split_name} accuracy: {acc:.2f}%")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{split_name} confusion matrix - accuracy: {acc:.2f}%", fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{split_name}_confusion_matrix.png"), dpi=150)
    plt.show()

    fig2, axes = plt.subplots(3, 3, figsize=(12, 12))
    indices = random.sample(range(len(loader.dataset)), 9)
    for i, idx in enumerate(indices):
        img, lbl = loader.dataset[idx]
        with torch.no_grad():
            out        = torch.softmax(model(img.unsqueeze(0).to(DEVICE)), dim=1)
            conf, pred = torch.max(out, 1)
        img_show = img.numpy().transpose((1, 2, 0))
        img_show = img_show * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_show = np.clip(img_show, 0, 1)
        ax       = axes[i // 3][i % 3]
        ax.imshow(img_show); ax.axis("off")
        color = "green" if pred.item() == lbl else "red"
        ax.set_title(
            f"pred: {CLASS_NAMES[pred.item()]} ({conf.item()*100:.1f}%)\nactual: {CLASS_NAMES[lbl]}",
            color=color, fontsize=9)
    plt.suptitle(f"{split_name} sample predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{split_name}_confidence_grid.png"), dpi=150)
    plt.show()

evaluate_split(val_loader,  "validation")
evaluate_split(test_loader, "test")

print("\ndone! check the plots and results folders")    