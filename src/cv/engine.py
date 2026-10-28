import copy
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import models


def top_k_mean(s, k=5):
    if len(s) <= k:
        return s.mean()
    return s.nlargest(k).mean()


def build_model(dropout, device):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.classifier[1].in_features, 1),
    )
    return m.to(device)


def video_metrics(m, loader, criterion, device, k=5):
    m.eval()
    recs = []
    loss_sum = 0.0
    n = 0
    correct = 0

    with torch.no_grad():
        for images, labels, vids in loader:
            images, labels = images.to(device), labels.to(device).float()

            # استخراج ویژگی برای تطابق با معماری بهینه‌شده
            features = m.features(images)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)

            out = m.classifier(features).squeeze(1)
            loss_sum += criterion(out, labels).item() * images.size(0)

            p = torch.sigmoid(out)
            correct += ((p >= 0.5).float() == labels).sum().item()
            n += images.size(0)

            recs += list(zip(vids, labels.cpu().numpy(), p.cpu().numpy()))

    d = pd.DataFrame(recs, columns=["video_id", "y", "p"])
    v = d.groupby("video_id").agg(y=("y", "first"), s=("p", lambda s: top_k_mean(s, k)))

    f1 = f1_score(
        v["y"].astype(int),
        (v["s"] >= 0.5).astype(int),
        average="macro",
        zero_division=0,
    )

    vacc = ((v["s"] >= 0.5).astype(int) == v["y"].astype(int)).mean()
    return loss_sum / n, correct / n, f1, vacc


def train_run(
    params,
    epochs,
    train_dataset,
    val_loader,
    criterion,
    device,
    trial=None,
    patience=3,
    label_smoothing=0.1,
):
    train_ld = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    m = build_model(params["dropout"], device)

    opt_cls = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[
        params["optimizer"]
    ]

    opt = opt_cls(
        m.classifier.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = {
        k: []
        for k in [
            "lr",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "val_video_f1",
            "val_video_acc",
        ]
    }

    best_f1 = -1.0
    best_state = None
    patience_counter = 0

    for ep in range(epochs):
        m.train()
        m.features.eval()  

        rl = 0.0
        c = 0
        t = 0
        hist["lr"].append(opt.param_groups[0]["lr"])

        for images, labels, _ in train_ld:
            images, labels = images.to(device), labels.to(device).float()

            if label_smoothing > 0.0:
                smooth_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                smooth_labels = labels

            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                features = m.features(images)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)

            out = m.classifier(features).squeeze(1)
            loss = criterion(out, smooth_labels)

            loss.backward()
            opt.step()

            rl += loss.item() * images.size(0)
            c += ((torch.sigmoid(out) >= 0.5).float() == labels).sum().item()
            t += labels.size(0)

        sched.step()

        vl, va, f1, vacc = video_metrics(m, val_loader, criterion, device)

        hist["train_loss"].append(rl / t)
        hist["train_acc"].append(c / t)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["val_video_f1"].append(f1)
        hist["val_video_acc"].append(vacc)

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(m.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if trial is not None:
            trial.report(f1, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if trial is None and patience_counter >= patience:
            print(
                f"Early stopping triggered at epoch {ep + 1}! Best Val Video F1: {best_f1:.4f}"
            )
            break

    if best_state is not None:
        m.load_state_dict(best_state)

    return m, hist, best_f1, best_state
