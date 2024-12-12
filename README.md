# Visual Navigation for Maze Exploration Using ORB and VLAD Descriptors

## Project Overview
This project focuses on enabling a mobile robot to navigate through a maze efficiently using computer vision techniques. The navigation system is built around ORB feature extraction, VLAD descriptors, and efficient nearest-neighbor search methodologies. By leveraging these techniques, the system interprets the visual environment and determines the shortest and most reliable path to a target location within the maze.

The project is part of the **Embodied AI Challenge**, emphasizing visual navigation and efficient planning. It provides a baseline solution, with opportunities to explore advanced and optimized approaches for solving complex navigation problems.

---

## Objective
The primary goal of this project is to build a robust pipeline for:
1. **Feature Extraction:** Extracting key visual features from images using ORB (Oriented FAST and Rotated BRIEF).
2. **Compact Representation:** Aggregating and normalizing visual descriptors using VLAD (Vector of Locally Aggregated Descriptors).
3. **Optimal Pathfinding:** Leveraging efficient data structures like BallTree for nearest-neighbor searches and homography validation to improve navigation accuracy.

---

## Problem Description
The task involves navigating a robot within a randomized maze environment. During the exploration phase, the robot collects visual data from its onboard camera, which is processed to create a compact representation of the environment. This data is used to guide the robot to its target during the navigation phase.

### Challenges
- **Visual Drift:** Accurate localization in the maze is affected by descriptor drift.
- **Efficient Navigation:** Real-time decision-making based on pre-computed data.
- **Maze Randomization:** Adapting the algorithm to unknown environments with varied textures and layouts.

---

## Methodology

### Exploration Phase
1. **Feature Extraction:**
   - ORB (Oriented FAST and Rotated BRIEF) is used for lightweight and robust feature extraction.
   - Keypoints and descriptors are extracted from images collected during exploration.
   - These descriptors are stored as a dataset for further analysis during the navigation phase.

2. **Codebook Creation:**
   - ORB descriptors are clustered into 128 groups using **K-Means Clustering**.
   - The cluster centers form a compact visual vocabulary.
   - This vocabulary enables efficient storage and retrieval of image descriptors.

3. **VLAD Descriptor Computation:**
   - The VLAD technique aggregates residuals between descriptors and their assigned cluster centers.
   - Residuals are summed for all descriptors mapped to a cluster, creating high-dimensional feature vectors.
   - These vectors are normalized using power and L2 normalization for compact representation.
   - **Parallel Processing**: VLAD computation is optimized using multi-core processing for speed.

### Navigation Phase
1. **VLAD-Based Matching:**
   - The VLAD descriptor of the current view is computed and compared against pre-computed descriptors in the exploration database.
   - Nearest-neighbor search is performed using **BallTree**, minimizing Euclidean distance to find the most visually similar match.
   
2. **Direction Decision:**
   - The robot tests possible movements (left, right, forward, backward) and calculates the shortest path to the goal.
   - The direction with the highest alignment to the target location is chosen.
   - Homography validation ensures the reliability of navigation decisions.

---

## Results

### Exploration Phase
- Successfully extracted ORB descriptors and clustered them into a visual vocabulary.
- Generated compact and robust VLAD descriptors for matching and navigation.

### Navigation Phase
- Achieved accurate target identification using VLAD descriptors and BallTree matching.
- Demonstrated efficient navigation to target locations, with homography validation improving the robustness of decisions.

**Sample Results:**
- Maze 1: Successfully navigated to the target location in under 2 minutes.
- Maze 2: Achieved target localization and navigation with a high degree of accuracy.

---

## Challenges and Solutions

### 1. Dead-End Handling
**Challenge:** Navigating out of dead ends during manual and semi-automated phases.  
**Solution:** Implemented visual recognition of dead ends and manual reversal strategies.

### 2. Descriptor Drift
**Challenge:** Descriptor matching errors due to visual drift.  
**Solution:** Applied homography validation to correct mismatches and normalized VLAD descriptors for consistency.

### 3. Computational Load
**Challenge:** High computational cost of real-time descriptor matching.  
**Solution:** Pre-computed ORB descriptors during exploration phase to significantly reduce runtime computations.

---

## Flow Diagram

### Pipeline Overview
1. **Feature Extraction:** Extract features using ORB.
2. **Descriptor Aggregation:** Cluster descriptors using K-Means and compute VLAD vectors.
3. **Pre-Nav Compute:** Use BallTree for efficient nearest-neighbor searches.
4. **Navigation Phase:** Match current view descriptors with the exploration dataset and decide the next move.

Flowchart representation:
```
Exploration Phase
    ↓
ORB Feature Extraction
    ↓
K-Means Clustering → Codebook Creation
    ↓
VLAD Descriptor Aggregation
    ↓
Navigation Phase
    ↓
BallTree Search → Movement Decision
```

---

## How to Set Up and Run

1. Clone the repository:
   ```bash
   git clone https://github.com/ai4ce/vis_nav_player.git
   cd vis_nav_player
   ```
2. Set up the environment:
   ```bash
   conda env create -f environment.yaml
   conda activate game
   ```
3. Run the solution:
   ```bash
   python player.py
   ```

---

## Conclusion
This project showcases a robust framework for visual navigation within a maze environment. By combining ORB feature extraction, VLAD descriptors, and BallTree search, the system ensures efficient and accurate navigation. This methodology is adaptable and scalable to dynamic and complex environments.

### Future Scope
- Extend the approach to dynamic mazes with moving obstacles.
- Incorporate deep learning methods for feature extraction and target prediction.
- Optimize real-time performance for robotic platforms with limited computational power.

---

## References
1. [ORB Documentation](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
2. Hervé Jégou et al., "Aggregating local descriptors into a compact image representation," CVPR 2010.
3. Libraries used: Scikit-learn, NumPy, OpenCV, pygame, etc.
