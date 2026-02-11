import numpy as np

class SLP:
    """
    Harmonic Single Layer Perceptron
    f(x) = Re((wz)^2)
    """

    def __init__(self):
        self.theta = np.random.randn()

    def forward(self, z):
        w = np.exp(1j * self.theta)
        u = w * z
        return np.real(u ** 2)

    def update(self, grad, lr):
        self.theta -= lr * grad

if __name__ == "__main__":
    datagen = DataGen()
    inputs, outputs = datagen.generate_xor_truth_table()
    model = SLP()
    model = train(model, inputs, outputs)
    f = model.forward(inputs)
    preds = (f > 0).astype(int)
    print("Predictions:", preds)
    print("Accuracy:", np.mean(preds == outputs))
