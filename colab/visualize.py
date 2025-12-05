import matplotlib.pyplot as plt

models = df_all_nano["Model"].tolist()
map50_95 = df_all_nano["mAP50-95"].tolist()
fps = df_all_nano["FPS"].tolist()

x = np.arange(len(models))

fig, ax1 = plt.subplots(figsize=(12, 6))

bars = ax1.bar(x, map50_95, width=0.6, edgecolor="black", alpha=0.85, label="mAP50-95")
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
ax1.set_ylabel("mAP50-95", fontweight="bold")
ax1.set_xlabel("Nano Models (CNN vs Transformer)", fontweight="bold")
ax1.set_title("Nano Models on VisDrone: Accuracy vs Speed", fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

for i, v in enumerate(map50_95):
    ax1.text(i, v + max(map50_95)*0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

ax2 = ax1.twinx()
ax2.plot(x, fps, marker="o", linewidth=2, label="FPS")
ax2.set_ylabel("FPS", fontweight="bold")

for i, v in enumerate(fps):
    ax2.text(i, v + max(fps)*0.03, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")

fig.tight_layout()
plt.show()

fig.savefig("nano_cnn_vs_transformer.png", dpi=300, bbox_inches="tight")
print("Saved nano_cnn_vs_transformer.png")
