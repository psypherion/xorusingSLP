import numpy as np

class Symmetry:

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def estimate_rotational_center(self):
        """
        Estimate 180° rotational symmetry center
        using label-preserving midpoints.
        """
        midpoints = []

        n = len(self.inputs)

        for i in range(n):
            for j in range(i + 1, n):
                if self.outputs[i] == self.outputs[j]:
                    midpoint = (self.inputs[i] + self.inputs[j]) / 2.0
                    midpoints.append(midpoint)

        midpoints = np.array(midpoints)

        if len(midpoints) == 0:
            raise ValueError("No symmetric label-preserving pairs found.")

        return np.mean(midpoints, axis=0)
