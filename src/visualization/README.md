# Visualization

This directory contains components for visualizing the drowsiness detection results in real-time, providing visual feedback and monitoring tools for the system.

## Components

### Real-Time Display (`plot_results.py`)

This module provides functions to create visual representations of the detection process:

#### `display_frame`
The main visualization function that combines all elements:
```python
def display_frame(
    frame: np.ndarray,
    face_region,
    facial_features,
    states,
    decision,
    alarm_active,
    alarm_type,
    drowsy_conf,
    flag_normal_state,
    ratios,
) -> np.ndarray:
```

- Converts frame from RGB to BGR color space
- Flips the frame horizontally for mirror-like view
- Adds status overlay with detection information when face is detected
- Adds "Face not detected" message with red border when no face is detected
- Returns the complete visualization frame

#### `draw_status_overlay`
Creates a semi-transparent overlay with detection information:
```python
def draw_status_overlay(frame: np.ndarray, states, decision, alarm_active, drowsy_conf, ratios, alarm_type) -> np.ndarray:
```

- Creates a semi-transparent black box for text background
- Displays drowsiness status and confidence
- Shows eye states (closed/open) with confidence values
- Displays mouth state (yawning/normal) with confidence
- Shows head position status
- Displays EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) values
- Shows blink count and blink rate
- Displays yawning count and rate
- Adds warning indicators when drowsiness or distraction is detected

#### `draw_feature_windows`
Adds small preview windows of detected facial features:
```python
def draw_feature_windows(frame: np.ndarray, facial_features) -> np.ndarray:
```

- Resizes feature images to small preview size (100x100 pixels)
- Converts feature images from RGB to BGR
- Places the small windows in the bottom-right corner:
  - Left eye preview
  - Right eye preview
  - Mouth preview
- Returns the frame with added feature windows

#### `plot_pipeline`
Creates a detailed visualization of the entire detection pipeline:
```python
def plot_pipeline(original_frame, face_region, facial_features, states, decision):
```

- Uses Matplotlib to create a figure with 5 subplots
- Displays the original frame
- Shows the detected face region with confidence
- Displays extracted left eye
- Displays extracted right eye
- Shows extracted mouth
- Adds decision information as a title

### Real-Time Plotter (`realtime_plotter.py`)

This module provides real-time plotting capabilities for facial measurements:

#### `RealTimePlotter` Class
Creates and updates real-time plots of facial measurements:
```python
class RealTimePlotter:
    def __init__(self, ear_threshold: float, mar_threshold: float):
```

- Initializes a figure with two vertical subplots
- Top subplot: EAR (Eye Aspect Ratio) with threshold line
- Bottom subplot: MAR (Mouth Aspect Ratio) with threshold line
- Stores data points in lists for continuous plotting

The `update` method adds new measurement points and updates the plots:
```python
def update(self, frame_idx: int, ear: float, mar: float):
```

- Appends new measurements to data lists
- Updates the plot lines with new data
- Automatically rescales the axes
- Redraws the figure canvas
- Uses Matplotlib's interactive mode for dynamic updates

