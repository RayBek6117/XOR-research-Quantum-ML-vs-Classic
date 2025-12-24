import pennylane as qml
from pennylane import numpy as pnp


def build_vqc_circuit(n_qubits: int = 2, n_layers: int = 1, shots=None):

    """
    Build a PennyLane QNode for a simple VQC.

    Encoding:
      RY(x0) on qubit 0, RY(x1) on qubit 1

    Variational block per layer:
      RZ-RY-RZ on each qubit + one CNOT entangler

    Measurement:
      <Z> on qubit 0

    shots:
      None -> analytic expectation value
      int  -> finite-shot sampling
    """

    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit(params, x_angles):
        # Angle encoding (2D XOR -> 2 qubits)
        qml.RY(x_angles[0], wires=0)
        qml.RY(x_angles[1], wires=1)

        # Variational layers
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RZ(params[l, w, 0], wires=w)
                qml.RY(params[l, w, 1], wires=w)
                qml.RZ(params[l, w, 2], wires=w)
            qml.CNOT(wires=[0, 1])

        return qml.expval(qml.PauliZ(0))

    return circuit


def prob_class1(circuit, params, x_angles):

    """
    Convert expectation value <Z> in [-1,1] to P(class=1) in [0,1].

    We interpret:
      p1 = (1 - <Z>) / 2
    """

    expval = circuit(params, x_angles)
    return (1.0 - expval) / 2.0


def mse_loss(circuit, params, X_angles, y):

    """Mean squared error between predicted probability and labels (0/1)."""

    preds = []
    for i in range(len(X_angles)):
        preds.append(prob_class1(circuit, params, X_angles[i]))
    preds = pnp.array(preds)
    y_ = pnp.array(y)
    return pnp.mean((preds - y_) ** 2)
