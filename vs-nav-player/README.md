# Visual Navigation: Keyboard-Controlled Player Using ORB and VLAD

## Overview
This repository contains a Python implementation for a visual navigation game where a robot navigates a maze using first-person visual inputs. The `player.py` script defines a keyboard-controlled player, leveraging computer vision techniques to interpret the environment and determine optimal navigation paths. This implementation is built for the **Embodied AI Challenge** and provides a solid foundation for exploring advanced maze navigation strategies.

---

## Key Features
1. **Keyboard-Controlled Navigation**:
   - Use keyboard inputs (`Arrow Keys`, `Space`, `Escape`) to interact with the robot.
   - Dynamically updates actions and navigation decisions based on user inputs.

2. **ORB and VLAD Features**:
   - **ORB (Oriented FAST and Rotated BRIEF)** is used to extract lightweight and robust local visual features.
   - **VLAD (Vector of Locally Aggregated Descriptors)** compacts image features for efficient storage and retrieval.

3. **Efficient Search with BallTree**:
   - The nearest neighbor search is implemented using the BallTree structure, enabling rapid retrieval of visually similar pre-explored locations.

4. **Dynamic Visualization**:
   - Real-time display of target views and navigation states using Pygame and OpenCV.
   - Supports strategic direction suggestions for reaching the goal.

5. **Logging and Debugging**:
   - Comprehensive logging for debugging and performance monitoring (`keyboard_player.log`).

---

## Code Breakdown

### 1. **Initialization and Setup**
The `KeyboardPlayerPyGame` class inherits from the `Player` class of the `vis_nav_game` framework. Key initialization steps:
- **Data Directories**: Ensures the existence of exploration data and initializes pre-trained ORB features and codebooks.
- **Key Mapping**: Maps keyboard inputs to actions (`LEFT`, `RIGHT`, `FORWARD`, `BACKWARD`, `CHECKIN`, `QUIT`).
- **Logging**: Configures detailed logging for debugging and monitoring.

### 2. **First-Person View (FPV)**
The `see()` method handles real-time updates of the robot's FPV:
- Converts OpenCV images to Pygame-compatible formats.
- Displays the current FPV on a Pygame window.
- Provides a dynamic view of the environment during both exploration and navigation phases.

### 3. **Feature Extraction and Descriptor Aggregation**
- **ORB Feature Extraction**:
  - Extracts robust keypoints and descriptors from the exploration images.
  - The descriptors are used to build a compact codebook via K-Means clustering.
- **VLAD Descriptor Computation**:
  - Aggregates ORB descriptors into a high-dimensional VLAD vector.
  - Applies power and L2 normalization for compact and robust feature representation.

### 4. **Pre-Navigation Computations**
The `pre_nav_compute()` method prepares the system for navigation by:
- Computing and saving ORB descriptors for all exploration images.
- Creating a visual vocabulary (codebook) using K-Means clustering.
- Constructing a BallTree for efficient nearest-neighbor searches.

### 5. **Navigation and Decision-Making**
- **Keyboard Inputs**: The `act()` method maps user inputs to navigation actions.
- **Dynamic Target Views**:
  - Displays a 2x2 grid of the target's views (`Front`, `Back`, `Left`, `Right`) using OpenCV.
- **Optimal Pathfinding**:
  - Calculates the nearest neighbor in the BallTree to find the best direction toward the goal.
  - Suggests the next best view dynamically based on the current position and goal proximity.

### 6. **Challenges and Solutions**
- **Dead-End Handling**: 
  - Simulates reverse movements to escape dead ends, using visual and feature-based cues.
- **Descriptor Drift**:
  - Employs homography validation and normalization to reduce mismatches.
- **Efficient Processing**:
  - Implements parallel processing for feature computation to handle large datasets.

---

## Setup and Usage

### Prerequisites
- Python 3.7 or higher
- Required Libraries: `opencv-python`, `pygame`, `scikit-learn`, `numpy`, `joblib`, `natsort`, `tqdm`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dineshsairallapalli/visual-navigation-game.git
   cd visual-navigation-game
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Player
1. Ensure exploration data is present in the specified directory (`C:/Users/ralla/vis_nav_player/data/images`).
2. Start the game with the keyboard-controlled player:
   ```bash
   python player.py
   ```
3. Control the robot using the following keys:
   - `Arrow Keys`: Navigate (`LEFT`, `RIGHT`, `FORWARD`, `BACKWARD`).
   - `Space`: Check-in at a location.
   - `Q`: Display the next best view toward the goal.
   - `Escape`: Quit the game.

---

## Flow Diagram

1. **Exploration Phase**:
   - ORB features are extracted from images.
   - VLAD descriptors are computed and stored for navigation.

2. **Pre-Navigation**:
   - BallTree is constructed for nearest-neighbor search.
   - Goal is set based on target image views.

3. **Navigation Phase**:
   - Nearest neighbor search determines the best move.
   - The robot navigates to the goal using FPV updates and visual cues.

---

## Key Components and Functions

### Class: `KeyboardPlayerPyGame`
#### Important Methods:
- `act()`: Maps keyboard inputs to actions.
- `see(fpv)`: Updates and displays the robot's FPV.
- `compute_orb_features()`: Extracts ORB features from exploration data.
- `get_VLAD(img)`: Computes VLAD descriptor for an input image.
- `pre_nav_compute()`: Prepares BallTree and codebook for navigation.
- `display_next_best_view()`: Suggests the optimal direction to reach the goal.

---

## Future Enhancements
- Automate exploration and navigation phases using AI-driven strategies.
- Integrate advanced feature extraction techniques (e.g., deep learning-based descriptors).
- Optimize real-time performance for resource-constrained environments.

---

## Logs and Debugging
Logs are saved in `keyboard_player.log` to track actions, computations, and errors during gameplay. Check this file for troubleshooting and performance monitoring.
