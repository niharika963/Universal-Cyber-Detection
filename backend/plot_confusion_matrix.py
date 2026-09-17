import numpy as np
import matplotlib.pyplot as plt

labels = ["Normal", "DoS", "Probe", "R2L", "U2R"]

cm = np.array([
    [19427, 1,     6,   16, 6],
    [11,    78279, 2,    0, 0],
    [1,     0,   821,    0, 0],
    [2,     0,     0,  221, 2],
    [1,     0,     0,    0, 9]
])

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix - IALP + IFF + XGBoost")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Confusion matrix saved as confusion_matrix.png")