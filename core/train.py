# train.py

import numpy as np


class Trainer:

    def __init__(self, model, z, y, epochs=2000, lr=0.1):
        self.model = model
        self.z = z
        self.y = y
        self.epochs = epochs
        self.lr = lr

    def train(self):

        y_pm = 2 * self.y - 1  # convert to {-1,1}

        for _ in range(self.epochs):

            w = np.exp(1j * self.model.theta)
            u = w * self.z
            f = np.real(u ** 2)

            margins = y_pm * f
            grad = 0.0

            for i in range(len(self.z)):
                if margins[i] < 1:
                    du_dtheta = 1j * w * self.z[i]
                    df_dtheta = 2 * np.real(u[i] * du_dtheta)
                    grad += -y_pm[i] * df_dtheta

            grad /= len(self.z)

            self.model.update(grad, self.lr)

        return self.model


if __name__ == "__main__":
    datagen = DataGen()
    inputs, outputs = datagen.generate_xor_truth_table()
    model = SLP()
    model = train(model, inputs, outputs)
    f = model.forward(inputs)
    preds = (f > 0).astype(int)
    print("Predictions:", preds)
    print("Accuracy:", np.mean(preds == outputs))
