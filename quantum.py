"""
conda create -n quantum python=3.10
conda activate quantum
pip install pennylane --upgrade
pip install pennylane-lightning pennylane-lightning[gpu]

https://github.com/PennyLaneAI/pennylane
https://pennylane.ai/qml/demos/tutorial_qubit_rotation
https://pennylane.ai/qml/demos/tutorial_qft_arithmetics
"""

import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as np

dev1 = qml.device("lightning.qubit", wires=1)


@qml.qnode(dev1, interface="autograd")
def circuit(phi1, phi2):
    qml.RX(phi1, wires=0)
    qml.RY(phi2, wires=0)
    return qml.expval(qml.PauliZ(0))


print(circuit(0.54, 0.12))
