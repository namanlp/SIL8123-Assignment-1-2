from os import environ, makedirs, path
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove TF XLA warnings
import tensorflow as tf

MODEL_PATH = '../cifar10_cnn_initial_model.keras'
USE_SUBSET = 20
NUM_CLASSES = 10

model = tf.keras.models.load_model(MODEL_PATH)

from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import NewtonFool
loss_object = tf.keras.losses.CategoricalCrossentropy()

import numpy as np
from tensorflow.keras.datasets import cifar10
(x_test, y_test_raw), (_,_) = cifar10.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = x_test[:USE_SUBSET]

y_test = tf.keras.utils.to_categorical(y_test_raw, NUM_CLASSES)
y_test = y_test[:USE_SUBSET]
y_test_raw = y_test_raw[:USE_SUBSET]

print("Using test set shape:", x_test.shape)

# === Wrap model with ART classifier ===
clf = TensorFlowV2Classifier(
    model=model,
    loss_object=loss_object,
    nb_classes=NUM_CLASSES,
    input_shape=x_test.shape[1:],
    clip_values=(0.0, 1.0),
)

# === Evaluate clean accuracy ===
preds_clean = clf.predict(x_test)
pred_labels_clean = np.argmax(preds_clean, axis=1)
true_labels = np.squeeze(y_test_raw)
clean_acc = np.mean(pred_labels_clean == true_labels)
print(f"Clean accuracy on {len(x_test)} samples: {clean_acc*100:.2f}%")

newton = NewtonFool(
    classifier=clf,
    max_iter=5,
    eta=0.05,
)

# === Generate adversarial examples ===
x_adv = newton.generate(x=x_test)

# === Evaluate adversarial accuracy ===
preds_adv = clf.predict(x_adv)
pred_labels_adv = np.argmax(preds_adv, axis=1)
adv_acc = np.mean(pred_labels_adv == true_labels)
print(f"Adversarial accuracy after NewtonFool: {adv_acc*100:.2f}%")

# === Compute perturbation norms ===
perturbations = (x_adv - x_test).reshape(len(x_adv), -1)
l2_norms = np.linalg.norm(perturbations, axis=1)
linf_norms = np.max(np.abs(perturbations), axis=1)
print(f"L2 norms: mean={l2_norms.mean():.4f}, max={l2_norms.max():.4f}")
print(f"L-inf norms: mean={linf_norms.mean():.4f}, max={linf_norms.max():.4f}")

import matplotlib.pyplot as plt
# === Save images ===
SAVE_DIR = "newton_fool_results"
makedirs(SAVE_DIR, exist_ok=True)

for i in range(USE_SUBSET):
    orig = np.clip((x_test[i] * 255.0), 0, 255).astype(np.uint8)
    adv = np.clip((x_adv[i] * 255.0), 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(6,2))
    axes[0].imshow(orig); axes[0].axis('off'); axes[0].set_title('orig')
    axes[1].imshow(adv); axes[1].axis('off'); axes[1].set_title('adv')
    diff = (adv.astype(np.int16) - orig.astype(np.int16))
    # scale diff for visualization
    disp = ((diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255).astype(np.uint8)
    axes[2].imshow(disp); axes[2].axis('off'); axes[2].set_title('scaled diff')
    plt.tight_layout()
    plt.savefig(path.join(SAVE_DIR, f"newtonfool_idx{i}.png"))
    plt.close(fig)
