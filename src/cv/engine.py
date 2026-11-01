import copy
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from torchvision import models


def top_k_mean(s, k=5):
    """
    Compute the mean probability of the top-k frames with the highest accident scores for each video.
    """
    if len(s) <= k:
        return s.mean()
    return s.nlargest(k).mean()


def build_model(dropout=0.2, device="cuda", backbone="mobilenet_v2"):
    """
    Build the model with a pretrained frozen backbone (MobileNetV2 or EfficientNet-B0)
    and a custom binary classification head.
    """
    if backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = 1280
    else:
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = m.classifier[1].in_features

    # Freeze all backbone parameters
    for p in m.parameters():
        p.requires_grad = False

    # Define custom classifier head for binary classification
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return m.to(device)


def _mean_group_ap(y_true, scores, groups):
    """
    Competition metric: mean of per-group average precision.
    Groups with a single class are skipped (AP undefined). If groups is None,
    a single AP over all videos is returned.
    """
    d = pd.DataFrame({"y": y_true, "s": scores, "g": groups})
    aps = []
    for _, sub in d.groupby("g"):
        if sub["y"].nunique() < 2:
            continue
        aps.append(average_precision_score(sub["y"], sub["s"]))
    return float(sum(aps) / len(aps)) if aps else 0.0


def video_metrics(model, loader, device, threshold=0.5, metric="f1", group_map=None):
    """
    Evaluate model performance at the video level using top-k frame aggregation.

    metric="f1"  -> returns (macro_f1, acc)   (default, MobileNet behaviour)
    metric="map" -> returns (mean per-group AP, acc). group_map is a dict
                    {5-digit video id: group}; if None, one global AP is used.
    """
    model.eval()
    recs = []

    with torch.no_grad():
        for images, labels, vids in loader:
            images = images.to(device)
            features = model.features(images)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            out = model.classifier(features).squeeze(1)
            p = torch.sigmoid(out)

            for vid, label, prob in zip(vids, labels.cpu().numpy(), p.cpu().numpy()):
                recs.append({"video_id": vid, "y": label, "p": prob})

    df = pd.DataFrame(recs)
    if df.empty:
        return 0.0, 0.0

    video_df = df.groupby("video_id").agg(
        y=("y", "first"), s=("p", lambda s: top_k_mean(s, k=5))
    )

    y_true = video_df["y"].astype(int).values
    y_pred = (video_df["s"] >= threshold).astype(int)

    acc = (y_pred == y_true).mean()

    if metric == "map":
        if group_map is None:
            groups = [0] * len(video_df)
        else:
            groups = [group_map.get(str(v).zfill(5), -1) for v in video_df.index]
        return _mean_group_ap(y_true, video_df["s"].values, groups), acc

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return macro_f1, acc


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
    metric="f1",
    group_map=None,
):
    """
    Full training loop with early stopping, Optuna pruning, and validation monitoring.
    """
    train_loader = DataLoader(
        train_dataset, batch_size=int(params["batch_size"]), shuffle=True, num_workers=0
    )

    # Support backward compatibility if backbone is not specified in params
    backbone = params.get("backbone", "mobilenet_v2")
    model = build_model(dropout=params["dropout"], device=device, backbone=backbone)

    optimizer_name = params.get("optimizer", "Adam")
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )

    best_val_f1 = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_f1": [], "val_score": []}

    for epoch in range(epochs):
        model.train()
        model.features.eval()  # Keep frozen backbone in eval mode
        total_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features = model.features(images)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            outputs = model.classifier(features).squeeze(1)

            if label_smoothing > 0:
                smoothed_labels = (
                    labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
                )
                loss = criterion(outputs, smoothed_labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        epoch_loss = total_loss / len(train_dataset)
        history["train_loss"].append(epoch_loss)

        # Validation phase
        val_f1, val_acc = video_metrics(
            model, val_loader, device, metric=metric, group_map=group_map
        )
        # val_f1 holds the selected metric (F1 or mAP) for backward compatibility
        history["val_f1"].append(val_f1)
        history["val_score"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return model, history, best_val_f1, best_model_state
