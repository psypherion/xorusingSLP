# main.py

import numpy as np
import matplotlib.pyplot as plt

from core.xor import DataGen
from core.symmetry import Symmetry
from core.slp import SLP
from core.train import Trainer


def visualize_original(inputs, outputs):

    plt.figure()

    for i in range(len(inputs)):
        plt.scatter(inputs[i, 0], inputs[i, 1])
        plt.text(inputs[i, 0] + 0.02,
                 inputs[i, 1] + 0.02,
                 str(int(outputs[i])))

    plt.title("Original XOR Dataset")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()


def main():

    # ---- Generate Data ----
    datagen = DataGen()
    inputs, outputs = datagen.generate_xor_truth_table()

    print("Inputs:\n", inputs)
    print("Outputs:", outputs)

    visualize_original(inputs, outputs)

    # ---- Estimate Symmetry Center ----
    symmetry = Symmetry(inputs, outputs)
    center = symmetry.estimate_rotational_center()

    print("\nEstimated Rotational Symmetry Center:", center)

    # ---- Center Data ----
    centered = inputs - center

    # Convert to complex
    z = centered[:, 0] + 1j * centered[:, 1]

    # ---- Train Model ----
    model = SLP()
    trainer = Trainer(model, z, outputs)
    model = trainer.train()

    # ---- Evaluate ----
    f = model.forward(z)
    preds = (f > 0).astype(int)

    print("\nPredictions:", preds)
    print("Accuracy:", np.mean(preds == outputs))

    # ---- Feature Visualization ----
    plt.figure()
    plt.scatter(f, outputs)
    plt.xlabel("Re((wz)^2)")
    plt.ylabel("Class")
    plt.title("Harmonic Feature Space")
    plt.show()


if __name__ == "__main__":
    main()
