# Pulsar classification using various classical and quantum models

This project has the goal to replicate the results of the paper "Pulsar Classification: Comparing Quantum Convolutional Neural Networks and Quantum Support Vector Machines" by D Slabbert, M Lourens and F Petruccione. The paper can be found [here](https://arxiv.org/abs/2309.15592).

## Datasets

The HTRU2 dataset is composed of statistical value of the integrated profile and the DM SNR curve. The dataset has the mean, standard deviation, excess kurtosis and skewness of the two informations meaning that it's composed of 8 float features. The target (class) is a binary classification of pulsar or not pulsar.


## Implementation

For this replication, we created all model discuss in the paper. This includes a QCNN, QSVM, QNN and their classical counterparts. The models were implemented using the Pennylane library and the classical models were implemented using the scikit-learn library. Both classical and quantum models used the same pipeline including a preprocessing and balancing step.


## Methods

Because we wanted to have the same result as in the paper, we selected the same metrics.

### Definitions

- TN : True Negative
- TP : True Positive
- FN : False Negative
- FP : False Positive

### Metrics calculation

- Accuracy : $\frac{TP+TN}{TP+TN+FP+FN}$
- Recall : $\frac{TP}{TP+FN}$
- Specificity : $\frac{TN}{TN+FP}$
- Precision : $\frac{TP}{TP+FP}$
- Negative Predictive Value : $\frac{TN}{TN+FN}$
- Balanced Accuracy : $\frac{Recall+Specificity}{2}$
- Geometric Mean : $\sqrt{Recall*Specificity}$
- Informedness : $Recall+Specificity-1$


## Notebooks

- data_exploration.ipynb : This notebook is used to explore the dataset and understand the data.
- classical_cnn.ipynb : This notebook is used to train and evaluate the classical CNN model.
- classical_svm.ipynb : This notebook is used to train and evaluate the classical SVM model.
- quantum_cnn.ipynb : This notebook is used to train and evaluate the quantum CNN model.
- quantum_svm.ipynb : This notebook is used to train and evaluate the quantum SVM model.
- quantum_cnn_early.ipynb : This notebook is used to train and evaluate the quantum CNN model with early stopping.