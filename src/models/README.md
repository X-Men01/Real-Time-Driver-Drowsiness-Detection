# Models

This directory contains the neural network architectures and pretrained weights used for the drowsiness detection system. These models are responsible for classifying eye states (open/closed) and mouth states (yawning/not yawning).

## Model Architectures

### CNN Models (`cnn_model.py`)

#### `Custom_CNN` Class

A flexible CNN architecture with configurable parameters:

```python
def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
```

**Architecture details:**
- **Convolutional Block 1:**
  - Conv2D → BatchNorm → ReLU
  - Conv2D → BatchNorm → ReLU
  - MaxPool2D (stride 2)
  - Channels: input_shape → hidden_units

- **Convolutional Block 2:**
  - Conv2D → BatchNorm → ReLU
  - Conv2D → BatchNorm → ReLU
  - MaxPool2D (stride 2)
  - Channels: hidden_units → hidden_units * 2

- **Convolutional Block 3:**
  - Conv2D → BatchNorm → ReLU
  - Conv2D → BatchNorm → ReLU
  - MaxPool2D (stride 2)
  - Channels: hidden_units * 2 → hidden_units * 4

- **Convolutional Block 4:**
  - Conv2D → BatchNorm → ReLU
  - Conv2D → BatchNorm → ReLU
  - MaxPool2D (stride 2)
  - Channels: hidden_units * 4 → hidden_units * 4

- **Classifier:**
  - Flatten
  - Dropout (p=0.5)
  - Linear layer (hidden_units* 8 * 8 * 4 → output_shape)

This architecture progressively extracts more complex features while reducing spatial dimensions, making it effective for classifying facial features.




## Pretrained Models

The directory contains pretrained model weights:

- **Eye State Models:**
  - `eye_state_model.pt`: Original model
  - `eye_state_model_V2.pt`: Improved version (currently used)

- **Mouth State Models:**
  - `mouth_state_model.pt`: Original model
  - `mouth_state_model_V2.pt`: Improved version (currently used)

These models are trained to classify:
- Eyes: Open (1) vs. Closed (0)
- Mouth: Yawning (1) vs. Not Yawning (0)

## Model Training

The `train.py` script contains the training pipeline for these models. The training process includes:

- Data loading and preprocessing
- Model initialization
- Training loop with validation
- Model checkpointing
- Performance evaluation





## Model Performance

The models are designed to be lightweight yet accurate, balancing performance with computational efficiency for real-time operation on edge devices like the Jetson Nano. Key characteristics:

- **Input Size:** 128×128 pixels
- **Inference Speed:** Fast enough for real-time processing
- **Parameters:** ~62,572 for eye model, ~88,910 for mouth model
- **Classification Accuracy:** High accuracy for binary classification tasks

