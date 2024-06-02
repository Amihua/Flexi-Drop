# Flexi-Drop
👋 Welcome to the repository for our paper titled "**FlexiDrop: Theoretical Insights and Practical Advances in Random Dropout Method on GNNs**".

## 🛠️ Installation

To get started with Flexi-drop, follow these installation instructions:

### Prerequisites

Ensure you have the following software installed:

- Python 3.8 or higher
- pip (Python package installer)

### Clone the Repository

Clone this repository to your local machine using:

```sh
git clone https://github.com/Amihua/Flexi-Drop/
cd  Flexi-Drop
```

### Install Dependencies
Install the required Python packages:
```shell
pip install -r requirements.txt
```
Since our implementation relies on PyTorch Geometric, you'll need to install it separately. Use the following command to install PyTorch Geometric along with the CUDA support suitable for your system:

```shell
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

## ▶️ Usage
### Graph Classification
To run the graph classification experiments:

1. Train the model:
   ```sh
   python graph_classification/train.py

   ```
2. Use multi-utility training script:
   ```sh
   python graph_classification/train_mutil.py
   ```

### Link Prediction
To run the link prediction experiments, train and use trained model:
****
```sh
python link_prediction/train.py
```

### Node Classification
To run the node classification experiments with different models:
```sh
python node_classification/main_GAT.py
python node_classification/main_GCN.py
python node_classification/main_SAGE.py
python node_classification/main_woreg.py
```

## License
This project is licensed under the **Apache License 2.0** - see the LICENSE file for details.

## Contributing & Contact
We welcome contributions! If you have any questions or issues, please open an issue in this repository or contact us at amihua@mail2.gdut.edu.cn.

Happy coding! 🚀
