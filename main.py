# main.py

import os
import numpy as np
import matplotlib.pyplot as plt

from core.xor import DataGen
from core.symmetry import Symmetry
from core.slp import SLP
from core.train import Trainer


RESULTS_DIR = "results"


def ensure_dir():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def save_plot(fig, name):
    fig.savefig(os.path.join(RESULTS_DIR, name))
    plt.close(fig)


def visualize_original(inputs, outputs, center):
    fig = plt.figure()

    for i in range(len(inputs)):
        plt.scatter(inputs[i, 0], inputs[i, 1])
        plt.text(inputs[i, 0] + 0.02,
                 inputs[i, 1] + 0.02,
                 str(int(outputs[i])))

    plt.scatter(center[0], center[1])
    plt.text(center[0] + 0.02, center[1] + 0.02, "Center")

    plt.title("Original XOR Dataset + Symmetry Center")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.axis("equal")

    save_plot(fig, "01_original_with_center.png")


def visualize_centered(centered):
    fig = plt.figure()

    for i in range(len(centered)):
        plt.arrow(0, 0,
                  centered[i, 0],
                  centered[i, 1],
                  head_width=0.05,
                  length_includes_head=True)

    plt.title("Centered Vectors")
    plt.xlabel("x1'")
    plt.ylabel("x2'")
    plt.grid(True)
    plt.axis("equal")

    save_plot(fig, "02_centered_vectors.png")


def visualize_unit_circle(z):
    fig = plt.figure()

    unit_vectors = z / np.abs(z)

    for i in range(len(unit_vectors)):
        plt.arrow(0, 0,
                  np.real(unit_vectors[i]),
                  np.imag(unit_vectors[i]),
                  head_width=0.05,
                  length_includes_head=True)

    circle = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(circle), np.sin(circle))

    plt.title("Unit Circle Projection")
    plt.axis("equal")
    plt.grid(True)

    save_plot(fig, "03_unit_circle.png")


def visualize_training(loss_history):
    fig = plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Hinge Loss")
    plt.title("Training Loss")
    plt.grid(True)

    save_plot(fig, "04_training_loss.png")


def visualize_harmonic_feature(f, outputs):
    fig = plt.figure()
    plt.scatter(f, outputs)
    plt.xlabel("Re((wz)^2)")
    plt.ylabel("Class")
    plt.title("Harmonic Feature Projection")
    plt.grid(True)

    save_plot(fig, "05_feature_projection.png")


def save_model(model):
    np.save(os.path.join(RESULTS_DIR, "model_theta.npy"), model.theta)


def load_model(model):
    model.theta = np.load(os.path.join(RESULTS_DIR, "model_theta.npy"))
    return model


def main():

    ensure_dir()

    datagen = DataGen()
    inputs, outputs = datagen.generate_xor_truth_table()

    symmetry = Symmetry(inputs, outputs)
    center = symmetry.estimate_rotational_center()

    print("Estimated Symmetry Center:", center)

    visualize_original(inputs, outputs, center)

    centered = inputs - center
    visualize_centered(centered)

    z = centered[:, 0] + 1j * centered[:, 1]
    visualize_unit_circle(z)

    model = SLP()
    trainer = Trainer(model, z, outputs)

    model, loss_history = trainer.train_with_history()

    visualize_training(loss_history)

    f = model.forward(z)
    preds = (f > 0).astype(int)

    print("Predictions:", preds)
    print("Accuracy:", np.mean(preds == outputs))

    visualize_harmonic_feature(f, outputs)

    save_model(model)

    print("Model and plots saved to 'results/' directory.")


if __name__ == "__main__":
    main()
