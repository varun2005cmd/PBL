# =============================================================================
# backend/tests/evaluate_dataset.py
#
# Standalone graphical evaluation harness for the face-recognition pipeline.
#
# What it does:
#   1. Runs MTCNN face detection on every image → draws bounding boxes.
#   2. Generates FaceNet embeddings for the train set → builds user prototypes.
#   3. Trains an SVM on the embeddings.
#   4. Tests on the test set → collects predictions, distances, confidences.
#   5. Produces 6 saved plots:
#       plot_face_detection.png   – face crops + bbox overlay for every image
#       plot_embeddings_pca.png   – 2-D PCA of all embeddings (train vs test)
#       plot_confusion_matrix.png – sklearn-style confusion matrix
#       plot_confidence_dist.png  – confidence score violin per user
#       plot_distance_dist.png    – Euclidean distance distribution
#       plot_metrics_summary.png  – bar chart: accuracy, precision, recall, F1
#
# Run from the backend/ directory:
#   python -m tests.evaluate_dataset
# or directly:
#   python tests/evaluate_dataset.py
# =============================================================================

import os
import sys
import glob
import time
import textwrap
from pathlib import Path

# ── ensure backend/ is on the path so `app` package is importable ─────────
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — writes PNG files, no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support, accuracy_score,
)
from sklearn.decomposition import PCA

# ── dataset paths ─────────────────────────────────────────────────────────
DATASET_DIR = BACKEND_DIR / "test_dataset"
TRAIN_DIR   = DATASET_DIR / "train"
TEST_DIR    = DATASET_DIR / "test"
OUTPUT_DIR  = BACKEND_DIR / "tests" / "eval_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


# =============================================================================
# Utilities
# =============================================================================

def load_images(base_dir: Path) -> dict:
    """
    Returns {username: [(path, bgr_image), ...]} for all sub-folders.
    """
    data = {}
    for user_dir in sorted(base_dir.iterdir()):
        if not user_dir.is_dir():
            continue
        imgs = []
        for ext in IMG_EXTS:
            for p in sorted(user_dir.glob(ext)):
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    imgs.append((p, bgr))
        if imgs:
            data[user_dir.name] = imgs
    return data


def save_fig(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# =============================================================================
# 1. Face Detection
# =============================================================================

def run_detection(all_images: dict):
    """
    Detect faces in every image using MTCNN.
    Returns {username: [(path, bgr, detection_or_None), ...]}
    """
    from app.ml.face_detector import detect_face

    results = {}
    for user, imgs in all_images.items():
        user_results = []
        for path, bgr in imgs:
            det = detect_face(bgr)
            user_results.append((path, bgr, det))
        results[user] = user_results
        detected = sum(1 for _, _, d in user_results if d is not None)
        print(f"    {user}: {detected}/{len(user_results)} faces detected")
    return results


def plot_face_detection(train_det: dict, test_det: dict):
    """
    Grid showing each image with bbox + crop overlay.
    Rows = users, Columns = train images then test images.
    """
    users = sorted(set(list(train_det.keys()) + list(test_det.keys())))
    n_train_max = max(len(v) for v in train_det.values()) if train_det else 0
    n_test_max  = max(len(v) for v in test_det.values())  if test_det  else 0
    n_cols = n_train_max + n_test_max + 1   # +1 for separator

    fig_w = max(16, n_cols * 2.5)
    fig_h = max(4,  len(users) * 3)
    fig, axes = plt.subplots(len(users), n_cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.suptitle("Face Detection Results\n(green box = detected | red border = miss)",
                 fontsize=13, fontweight="bold")

    for row, user in enumerate(users):
        col = 0
        # ── train images ──────────────────────────────────────────────────
        for i in range(n_train_max):
            ax = axes[row][col]
            ax.axis("off")
            items = train_det.get(user, [])
            if i < len(items):
                path, bgr, det = items[i]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
                if det:
                    x1, y1, x2, y2 = det["bbox"]
                    rect = mpatches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor="lime", facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.set_title(
                        f"Train {i+1}\nconf={det['confidence']:.2f}",
                        fontsize=7, color="green"
                    )
                else:
                    for spine in ax.spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(3)
                    ax.set_title(f"Train {i+1}\nNO FACE", fontsize=7, color="red")
            col += 1

        # ── separator ─────────────────────────────────────────────────────
        axes[row][col].axis("off")
        axes[row][col].text(0.5, 0.5, "│", ha="center", va="center",
                            fontsize=20, color="gray",
                            transform=axes[row][col].transAxes)
        col += 1

        # ── test images ───────────────────────────────────────────────────
        for i in range(n_test_max):
            ax = axes[row][col]
            ax.axis("off")
            items = test_det.get(user, [])
            if i < len(items):
                path, bgr, det = items[i]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(rgb)
                if det:
                    x1, y1, x2, y2 = det["bbox"]
                    rect = mpatches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor="cyan", facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.set_title(
                        f"Test {i+1}\nconf={det['confidence']:.2f}",
                        fontsize=7, color="steelblue"
                    )
                else:
                    ax.set_title(f"Test {i+1}\nNO FACE", fontsize=7, color="red")
            col += 1

        axes[row][0].set_ylabel(user, fontsize=11, fontweight="bold", rotation=90, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "plot_face_detection.png")


# =============================================================================
# 2. Embeddings
# =============================================================================

def build_embeddings(detection_results: dict):
    """
    Returns {username: [np.ndarray(512), ...]} for all successfully detected faces.
    """
    from app.ml.embedder import generate_embedding

    emb_map = {}
    for user, items in detection_results.items():
        embs = []
        for _, _, det in items:
            if det is None:
                continue
            emb = generate_embedding(det["face_crop"])
            if emb is not None:
                embs.append(emb)
        emb_map[user] = embs
        print(f"    {user}: {len(embs)} embeddings generated")
    return emb_map


def plot_embeddings_pca(train_emb: dict, test_emb: dict):
    """
    PCA to 2-D; scatter train=filled, test=star marker.
    """
    X, labels, split = [], [], []
    for user, embs in train_emb.items():
        for e in embs:
            X.append(e); labels.append(user); split.append("train")
    for user, embs in test_emb.items():
        for e in embs:
            X.append(e); labels.append(user); split.append("test")

    if len(X) < 2:
        print("  Not enough embeddings for PCA — skipping.")
        return

    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.array(X))

    users = sorted(set(labels))
    color_map = {u: COLORS[i % len(COLORS)] for i, u in enumerate(users)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y) in enumerate(coords):
        marker = "o" if split[i] == "train" else "*"
        sz     = 80  if split[i] == "train" else 160
        ax.scatter(x, y, c=color_map[labels[i]], marker=marker, s=sz,
                   edgecolors="black", linewidths=0.5, zorder=3)

    # legend: users by color
    user_patches = [mpatches.Patch(color=color_map[u], label=u) for u in users]
    train_pt = plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="gray", markersize=8, label="Train")
    test_pt  = plt.Line2D([0], [0], marker="*", color="w",
                          markerfacecolor="gray", markersize=12, label="Test")
    ax.legend(handles=user_patches + [train_pt, test_pt], loc="best", fontsize=9)

    ax.set_title(
        f"Embedding Space (PCA 2-D)\n"
        f"Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}  "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}",
        fontsize=11
    )
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    save_fig(fig, "plot_embeddings_pca.png")


# =============================================================================
# 3. Recognition (SVM + Euclidean)
# =============================================================================

def run_recognition(train_emb: dict, test_emb: dict):
    """
    Train SVM, then predict every test embedding.
    Returns list of dicts: {true, pred, confidence, distance, user}.
    """
    from app.ml.recognizer import train_classifier, recognize_user

    # train SVM
    train_result = train_classifier(train_emb)
    print(f"    SVM trained: {train_result}")

    results = []
    for user, embs in test_emb.items():
        for emb in embs:
            rec = recognize_user(emb, train_emb)
            results.append({
                "true":       user,
                "pred":       rec["user"],
                "confidence": rec["confidence"],
                "distance":   rec["distance"],
                "method":     rec["method"],
            })
    return results


# =============================================================================
# 4. Metrics Plots
# =============================================================================

def plot_confusion_matrix(results: list):
    users = sorted(set(r["true"] for r in results))
    y_true = [r["true"] for r in results]
    y_pred = [r["pred"] for r in results]

    cm = confusion_matrix(y_true, y_pred, labels=users)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=users)

    fig, ax = plt.subplots(figsize=(max(5, len(users) * 2), max(4, len(users) * 1.8)))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix\n(rows = true, cols = predicted)", fontsize=12)
    plt.tight_layout()
    save_fig(fig, "plot_confusion_matrix.png")


def plot_confidence_distribution(results: list):
    users = sorted(set(r["true"] for r in results))

    fig, ax = plt.subplots(figsize=(max(6, len(users) * 2.5), 5))
    ax.axhline(0.55, color="orange", linestyle="--", linewidth=1.2,
               label="SVM threshold (0.55)")

    for i, user in enumerate(users):
        vals = [r["confidence"] for r in results if r["true"] == user]
        if not vals:
            continue
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
        ax.scatter([i + j for j in jitter], vals,
                   color=COLORS[i % len(COLORS)], s=80,
                   edgecolors="black", linewidths=0.5, zorder=3,
                   label=user)
        ax.bar(i, np.mean(vals), width=0.4, alpha=0.3,
               color=COLORS[i % len(COLORS)])

    ax.set_xticks(range(len(users)))
    ax.set_xticklabels(users, fontsize=11)
    ax.set_ylabel("Confidence Score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recognition Confidence per User\n(bar = mean, dots = individual samples)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    save_fig(fig, "plot_confidence_dist.png")


def plot_distance_distribution(results: list):
    users = sorted(set(r["true"] for r in results))

    fig, ax = plt.subplots(figsize=(max(6, len(users) * 2.5), 5))
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1.5,
               label="Distance threshold (0.9)")

    for i, user in enumerate(users):
        vals = [r["distance"] for r in results if r["true"] == user]
        if not vals:
            continue
        jitter = np.random.default_rng(0).uniform(-0.1, 0.1, len(vals))
        ax.scatter([i + j for j in jitter], vals,
                   color=COLORS[i % len(COLORS)], s=80,
                   edgecolors="black", linewidths=0.5, zorder=3,
                   label=user)
        ax.bar(i, np.mean(vals), width=0.4, alpha=0.3,
               color=COLORS[i % len(COLORS)])

    ax.set_xticks(range(len(users)))
    ax.set_xticklabels(users, fontsize=11)
    ax.set_ylabel("Euclidean Distance to Nearest Prototype")
    ax.set_title("Embedding Distance per User\n(lower = better match)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    save_fig(fig, "plot_distance_dist.png")


def plot_metrics_summary(results: list):
    users = sorted(set(r["true"] for r in results))
    y_true = [r["true"] for r in results]
    y_pred = [r["pred"] for r in results]
    acc = accuracy_score(y_true, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=users, zero_division=0, average=None
    )

    x = np.arange(len(users))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Recognition Metrics   (Overall Accuracy = {acc:.1%})",
                 fontsize=13, fontweight="bold")

    # Per-user bars
    ax = axes[0]
    ax.bar(x - width,   prec, width, label="Precision", color="#4C72B0")
    ax.bar(x,           rec,  width, label="Recall",    color="#55A868")
    ax.bar(x + width,   f1,   width, label="F1 Score",  color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(users, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-User Precision / Recall / F1")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in ax.patches:
        ax.annotate(f"{bar.get_height():.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                    ha="center", fontsize=8, color="black")

    # Overall summary donut
    ax2 = axes[1]
    mean_metrics = [acc, float(np.mean(prec)), float(np.mean(rec)), float(np.mean(f1))]
    metric_names  = ["Accuracy", "Macro Prec.", "Macro Recall", "Macro F1"]
    bar_colors    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax2.barh(metric_names, mean_metrics, color=bar_colors, edgecolor="black")
    ax2.set_xlim(0, 1.15)
    ax2.set_xlabel("Score")
    ax2.set_title("Overall Model Summary")
    for bar, val in zip(bars, mean_metrics):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "plot_metrics_summary.png")


def plot_per_sample_results(test_det: dict, results: list):
    """
    For each test image show the crop with predicted label + confidence.
    """
    users = sorted(test_det.keys())
    n_cols = max(len(v) for v in test_det.values())
    n_rows = len(users)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3, n_rows * 3.5),
                             squeeze=False)
    fig.suptitle("Per-Sample Recognition Results\n(green = correct | red = wrong)",
                 fontsize=13, fontweight="bold")

    result_iter = iter(results)
    for row, user in enumerate(users):
        items = test_det[user]      # list of (path, bgr, det)
        for col in range(n_cols):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(items):
                continue
            path, bgr, det = items[col]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # pick the matching result for this user/image
            try:
                res = next(r for r in results
                           if r["true"] == user and "_iter" not in r)
                # mark consumed
                res["_iter"] = True
            except StopIteration:
                res = {"pred": "?", "confidence": 0.0}

            correct = res["pred"].lower() == user.lower()
            border_color = "#27AE60" if correct else "#E74C3C"

            if det:
                face_rgb = cv2.cvtColor(det["face_crop"], cv2.COLOR_RGB2BGR)
                face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
                ax.imshow(face_rgb)
            else:
                ax.imshow(rgb)

            title = (
                f"True: {user}\n"
                f"Pred: {res['pred']}\n"
                f"Conf: {res['confidence']:.2f}"
            )
            ax.set_title(title, fontsize=8,
                         color=border_color, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "plot_per_sample_results.png")


# =============================================================================
# 5. Console report
# =============================================================================

def print_report(results: list):
    from sklearn.metrics import classification_report
    y_true = [r["true"] for r in results]
    y_pred = [r["pred"] for r in results]
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, zero_division=0))
    total   = len(results)
    correct = sum(1 for r in results if r["pred"] == r["true"])
    print(f"Overall accuracy : {correct}/{total} = {correct/total:.1%}")
    print("=" * 60)
    print("\nPer-sample breakdown:")
    for r in results:
        status = "✓" if r["pred"] == r["true"] else "✗"
        print(f"  {status}  true={r['true']:10s}  pred={r['pred']:10s}  "
              f"conf={r['confidence']:.3f}  dist={r['distance']:.3f}  ({r['method']})")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n══════════════════════════════════════════════════")
    print("  Face Recognition Evaluation Harness")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Outputs : {OUTPUT_DIR}")
    print("══════════════════════════════════════════════════\n")

    # ── 1. Load images ────────────────────────────────────────────────────
    print("[1/6] Loading images …")
    train_imgs = load_images(TRAIN_DIR)
    test_imgs  = load_images(TEST_DIR)
    print(f"  Train users: {list(train_imgs.keys())}  "
          f"(total {sum(len(v) for v in train_imgs.values())} images)")
    print(f"  Test  users: {list(test_imgs.keys())}  "
          f"(total {sum(len(v) for v in test_imgs.values())} images)")

    # ── 2. Face detection ────────────────────────────────────────────────
    print("\n[2/6] Running face detection (MTCNN) …")
    train_det = run_detection(train_imgs)
    test_det  = run_detection(test_imgs)

    print("  Plotting face detection grid …")
    plot_face_detection(train_det, test_det)

    # ── 3. Generate embeddings ───────────────────────────────────────────
    print("\n[3/6] Generating FaceNet embeddings …")
    train_emb = build_embeddings(train_det)
    test_emb  = build_embeddings(test_det)

    flat_train = {u: e for u, e in train_emb.items() if e}
    flat_test  = {u: e for u, e in test_emb.items()  if e}

    if not flat_train:
        print("\n  ERROR: No embeddings generated for training set.")
        print("  Ensure MTCNN detected at least one face in train images.")
        sys.exit(1)

    print("  Plotting 2-D PCA of embedding space …")
    plot_embeddings_pca(flat_train, flat_test)

    # ── 4. Recognition ───────────────────────────────────────────────────
    print("\n[4/6] Training SVM + evaluating test set …")
    results = run_recognition(flat_train, flat_test)

    if not results:
        print("  ERROR: No test samples evaluated. "
              "Check that test images have detectable faces.")
        sys.exit(1)

    # ── 5. Metric plots ───────────────────────────────────────────────────
    print("\n[5/6] Generating metric plots …")
    plot_confusion_matrix(results)
    plot_confidence_distribution(results)
    plot_distance_distribution(results)
    plot_metrics_summary(results)
    plot_per_sample_results(test_det, [dict(r) for r in results])

    # ── 6. Console report ─────────────────────────────────────────────────
    print("\n[6/6] Summary")
    print_report(results)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
