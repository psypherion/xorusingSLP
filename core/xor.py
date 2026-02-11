import numpy as np

class DataGen:
    def __init__(self):
        self.inputs = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]], dtype=float)
        self.outputs = np.array([0, 1, 1, 0], dtype=float)

    def generate_xor_truth_table(self):
        return self.inputs, self.outputs

if __name__ == "__main__":
    datagen = DataGen()
    inputs, outputs = datagen.generate_xor_truth_table()
    print("XOR Truth Table:")
    print("Inputs | Output")
    for i in range(len(inputs)):
        print(f"{inputs[i]} | {outputs[i]}")
