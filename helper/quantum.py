import torch
import pennylane as qml
import pennylane.numpy as np


def create_qcnn_model(model_parameters, draw_model = False):
    n_qubits = model_parameters.num_features

    dev = qml.device("lightning.qubit", wires=n_qubits)

    weight_shapes = {"weights_c1": 16, "weights_c2": 18, "weights_c3": 2}


    @qml.qnode(dev)
    def qnode(inputs, weights_c1, weights_c2, weights_c3):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        conv_layer(weights_c1, 5, 8)

        qml.Barrier()
        pool_layer(8)

        qml.Barrier()
        conv_layer(weights_c2, 3, 4)

        qml.Barrier()
        pool_layer(4)

        qml.Barrier()
        qml.RY(weights_c3[0], wires=0)
        qml.RY(weights_c3[1], wires=1)
        qml.CNOT([0,1])

        qml.Barrier()
        pool_layer(2)

        return qml.expval(qml.PauliZ(0))
    
    if draw_model:
        random_input = np.random.randn(8)
        weights_c1 = np.random.randn(16)
        weights_c2 = np.random.randn(8)
        weights_c3 = np.random.randn(2)
        qml.draw_mpl(qnode)(random_input, weights_c1, weights_c2, weights_c3)

    class QuantumCNN(torch.nn.Module):
        def __init__(self):
            super(QuantumCNN, self).__init__()
            self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        def forward(self, x):   
            x = self.qlayer(x)
            x = torch.sigmoid(x)
            return x
    
    return QuantumCNN()

def conv(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    
def conv_layer(params, stride, n_qubits):
    wires = list(range(n_qubits))
    for i in range(n_qubits):
        conv(params[2*i:2*i+2], [wires[i], wires[(i+stride)%n_qubits]])

def pool_layer(n_qubits):
    for i in range(int(n_qubits/2)):
        qml.CNOT([n_qubits-i-1, i])