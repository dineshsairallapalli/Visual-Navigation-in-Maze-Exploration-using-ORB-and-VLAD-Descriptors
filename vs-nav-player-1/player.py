# Import necessary libraries and modules
import logging
from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted
from joblib import Parallel, delayed

# Initialize logging configuration
logging.basicConfig(
    filename='keyboard_player.log',  # Log messages will be written to this file
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        logging.info("Initializing KeyboardPlayerPyGame class")
        # Initialize class variables
        logging.info("Setting self.fpv to None")
        self.fpv = None  # First-person view image
        logging.info("Setting self.last_act to Action.IDLE")
        self.last_act = Action.IDLE  # Last action taken by the player
        logging.info("Setting self.screen to None")
        self.screen = None  # Pygame screen
        logging.info("Setting self.keymap to None")
        self.keymap = None  # Mapping of keyboard keys to actions
        logging.info("Calling superclass _init_")
        super(KeyboardPlayerPyGame, self).__init__()

        # Variables for reading exploration data
               
        self.save_dir = 'C:/Users/ralla/vis_nav_player_1/data/images'
        logging.info(f"Setting self.save_dir to '{self.save_dir}'")
        if not os.path.exists(self.save_dir):
            logging.error(f"Directory {self.save_dir} does not exist, please download exploration data.")
            raise FileNotFoundError(f"Directory {self.save_dir} does not exist, please download exploration data.")

        # Initialize ORB detector
        logging.info("Initializing ORB detector")
        self.orb = cv2.ORB_create()

        # Load pre-trained orb features and codebook
        logging.info("Initializing orb_descriptors and codebook to None")
        self.orb_descriptors, self.codebook = None, None
        if os.path.exists("orb_descriptors.npy"):
            logging.info("Loading orb_descriptors from 'orb_descriptors.npy'")
            self.orb_descriptors = np.load("orb_descriptors.npy")
        else:
            logging.warning("'orb_descriptors.npy' not found. It will be computed later.")

        if os.path.exists("codebook.pkl"):
            logging.info("Loading codebook from 'codebook.pkl'")
            with open("codebook.pkl", "rb") as f:
                self.codebook = pickle.load(f)
        else:
            logging.warning("'codebook.pkl' not found. It will be computed later.")

        # Initialize database for storing VLAD descriptors of FPV
        logging.info("Setting self.database and self.goal to None")
        self.database = None
        self.goal = None

    def reset(self):
        logging.info("Resetting player state")
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        logging.info("Initializing pygame")
        pygame.init()

        # Define key mappings for actions
        logging.info("Defining keymap for actions")
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        logging.info("Handling player actions based on keyboard input")
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            logging.info(f"Processing event: {event}")
            # Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                logging.info("Received pygame.QUIT event")
                pygame.quit()
                self.last_act = Action.QUIT
                logging.info("Setting last_act to Action.QUIT")
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                logging.info("Keydown event detected")
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    logging.info(f"Key {event.key} found in keymap, updating last_act")
                    # If yes, bitwise OR the current action with the new one
                    self.last_act |= self.keymap[event.key]
                else:
                    logging.info(f"Key {event.key} not in keymap, showing target images")
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                logging.info("Keyup event detected")
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    logging.info(f"Key {event.key} found in keymap, updating last_act")
                    # If yes, bitwise XOR the current action with the new one
                    self.last_act ^= self.keymap[event.key]
        logging.info(f"Returning last_act: {self.last_act}")
        return self.last_act

    def show_target_images(self):
        logging.info("Retrieving target images")
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            logging.warning("No target images available to display")
            return

        logging.info("Creating 2x2 grid of target images")
        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        logging.info("Concatenated first two target images horizontally")
        hor2 = cv2.hconcat(targets[2:])
        logging.info("Concatenated last two target images horizontally")
        concat_img = cv2.vconcat([hor1, hor2])
        logging.info("Concatenated both horizontal images vertically")

        w, h = concat_img.shape[:2]
        logging.info(f"Image dimensions - Width: {w}, Height: {h}")

        color = (0, 0, 0)
        logging.info("Drawing vertical and horizontal lines to create grid")
        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        logging.info("Adding text labels to each quadrant")
        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        logging.info("Displaying concatenated target images using OpenCV")
        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)
        logging.info("Displayed target images")

    def set_target_images(self, images):
        logging.info("Setting target images")
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        logging.info("Displaying target images after setting")
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        logging.info("Displaying image from ID")
        """
        Display image from database based on its ID using OpenCV
        """
        path = os.path.join(self.save_dir, f"{id}.png")
        logging.info(f"Constructed image path: {path}")
        if os.path.exists(path):
            logging.info(f"Image exists at {path}, loading image")
            img = cv2.imread(path)
            if img is not None:
                logging.info(f"Displaying image in window: {window_name}")
                cv2.imshow(window_name, img)
                cv2.waitKey(1)
            else:
                logging.error(f"Failed to read image at {path}")
        else:
            logging.warning(f"Image with ID {id} does not exist")
            print(f"Image with ID {id} does not exist")

    def compute_orb_features(self):
        logging.info("Computing ORB features for images in the data directory")
        """
        Compute ORB features for images in the data directory
        """
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.png')])
        orb_descriptors = []
        logging.info(f"Found {len(files)} .png files in '{self.save_dir}'")
        for img in tqdm(files, desc="Processing images"):
            img_path = os.path.join(self.save_dir, img)
            logging.info(f"Reading image: {img_path}")
            image = cv2.imread(img_path)
            if image is None:
                logging.warning(f"Failed to read image: {img_path}")
                continue
            # Pass the image to ORB detector and get keypoints + descriptions
            logging.info("Detecting keypoints and computing descriptors with ORB")
            keypoints = self.orb.detect(image,None)
            keypoints, des = self.orb.compute(image, keypoints)
            if des is not None:
                orb_descriptors.extend(des)
            else:
                logging.warning(f"No descriptors found in image: {img_path}")
        orb_descriptors = np.asarray(orb_descriptors)
        logging.info(f"Computed {orb_descriptors.shape[0]} ORB descriptors")
        return orb_descriptors

    def get_VLAD(self, img):
        logging.info("Computing VLAD descriptor for the image")
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        # Pass the image to ORB detector and get keypoints + descriptions
        logging.info("Detecting keypoints and computing descriptors with ORB")
        keypoints = self.orb.detect(img,None)
        keypoints, des = self.orb.compute(img, keypoints)
        if des is None:
            logging.warning("No ORB descriptors found for the image")
            return np.zeros(self.codebook.cluster_centers_.shape[1] * self.codebook.n_clusters)

        # Predict the cluster labels using the pre-trained codebook
        logging.info("Predicting cluster labels using the codebook")
        pred_labels = self.codebook.predict(des)
        centroids = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        logging.info(f"Number of clusters: {k}")
        VLAD_feature = np.zeros([k, des.shape[1]])

        # Aggregate residuals
        logging.info("Aggregating residuals for VLAD")
        for i in range(k):
            mask = pred_labels == i
            if np.sum(mask) > 0:
                residual = des[mask] - centroids[i]
                VLAD_feature[i] = np.sum(residual, axis=0)
                logging.debug(f"Updated VLAD_feature for cluster {i}")

        VLAD_feature = VLAD_feature.flatten()
        # Power normalization
        logging.info("Applying power normalization to VLAD_feature")
        VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))
        # L2 normalization
        norm = np.linalg.norm(VLAD_feature)
        if norm > 0:
            VLAD_feature = VLAD_feature / norm
        else:
            logging.warning("VLAD_feature norm is zero, cannot normalize")
        logging.info("VLAD descriptor computed successfully")
        return VLAD_feature

    def get_neighbor(self, img):
        logging.info("Finding nearest neighbor in the database based on VLAD descriptor")
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        # Get the VLAD feature of the image
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        # Query the BallTree for the nearest neighbor
        logging.info("Querying BallTree for nearest neighbor")
        _, index = self.tree.query(q_VLAD, 1)
        neighbor_idx = index[0][0]
        logging.info(f"Nearest neighbor index: {neighbor_idx}")
        return neighbor_idx

    def pre_nav_compute(self):
        logging.info("Performing pre-navigation computations")
        """
        Build BallTree for nearest neighbor search and find the goal ID
        """
        # Compute ORB features for images in the database
        if self.orb_descriptors is None:
            logging.info("Computing ORB features as orb_descriptors is None")
            self.orb_descriptors = self.compute_orb_features()
            if self.orb_descriptors.size == 0:
                logging.error("No ORB descriptors were computed. Exiting pre_nav_compute.")
                raise ValueError("No ORB descriptors were computed. Ensure that images are valid and ORB descriptors are extracted.")
            np.save("orb_descriptors.npy", self.orb_descriptors)
            logging.info("Saved orb_descriptors to 'orb_descriptors.npy'")
        else:
            logging.info("Loaded ORB features from 'orb_descriptors.npy'")

        # KMeans clustering algorithm to create a codebook
        if self.codebook is None:
            logging.info("Computing codebook with KMeans")
            self.codebook = KMeans(n_clusters=128,init='k-means++', n_init=10, verbose=1).fit(self.orb_descriptors)
            with open("codebook.pkl", "wb") as f:
                pickle.dump(self.codebook, f)
            logging.info("Saved codebook to 'codebook.pkl'")
        else:
            logging.info("Loaded codebook from 'codebook.pkl'")

        # Get VLAD embedding for each image in the exploration phase
        if self.database is None:
            logging.info("Initializing database list and computing VLAD embeddings for exploration observations")
            exploration_observation = natsorted(
                [os.path.join(self.save_dir, x) for x in os.listdir(self.save_dir) if x.endswith(('.png', '.jpg'))]
            )
            logging.info(f"Number of exploration observations: {len(exploration_observation)}")

            if not exploration_observation:
                logging.error(f"No .jpg or .png files found in directory: {self.save_dir}")
                raise FileNotFoundError(f"No .jpg or .png files found in directory: {self.save_dir}")

            # Function to process a single image and return the VLAD descriptor
            def process_image(img_path):
                logging.info(f"Processing image for VLAD: {img_path}")
                image = cv2.imread(img_path)
                if image is None:
                    logging.warning(f"Failed to read image: {img_path}")
                    return np.zeros(self.codebook.cluster_centers_.shape[1] * self.codebook.n_clusters)
                return self.get_VLAD(image)

            # Parallel processing of images to compute VLAD descriptors
            logging.info("Starting parallel processing of images to compute VLAD descriptors")
            self.database = Parallel(n_jobs=-1, backend="threading")(
                delayed(process_image)(img_path) for img_path in tqdm(exploration_observation, desc="Processing images")
            )
            logging.info("Completed parallel processing of images")

            # Filter out any zero descriptors (from failed image reads)
            self.database = [desc for desc in self.database if np.any(desc)]
            if not self.database:
                logging.error("No valid VLAD descriptors were computed. BallTree cannot be built.")
                raise ValueError("No valid VLAD descriptors were computed. Ensure that images are valid and ORB descriptors are extracted.")

            # Convert list to numpy array for BallTree
            self.database = np.vstack(self.database)
            logging.info(f"Database shape for BallTree: {self.database.shape}")

            # Build a BallTree for fast nearest neighbor search
            logging.info("Building BallTree for nearest neighbor search")
            self.tree = BallTree(self.database, leaf_size=64)
            logging.info("BallTree has been built with leaf_size=64")

    def pre_navigation(self):
        logging.info("Entering pre_navigation method")
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()

    def get_neighbor_in_direction(self, index, direction):
        logging.info(f"Calculating neighbor in direction: {direction} from index: {index}")
        """
        Simulate moving to a neighbor in a given direction.
        This function returns the neighboring index based on the assumed layout of the maze.
        """
        # Define index changes for each direction based on the maze's layout
        offsets = {
            'right': 1,   # Moving to the right increases index by 1
            'left': -1,   # Moving to the left decreases index by 1
            'front': 1,   # Moving forward increases index by maze width
            'back': -1,   # Moving back decreases index by maze width
        }

        # Calculate and return the neighboring index in the specified direction
        neighbor_index = index + offsets.get(direction, 0)
        logging.info(f"Neighbor index in {direction}: {neighbor_index}")
        return neighbor_index

    def calculate_goal_distance(self, index1, index2):
        logging.info(f"Calculating distance between {index1} and {index2}")
        """
        Calculate a basic "distance" between two indices as a heuristic for proximity.
        """
        distance = abs(index1 - index2)
        logging.info(f"Calculated distance: {distance}")
        return distance

    def display_next_best_view(self):
        logging.info("Displaying the next best view to reach the target location")
        """
        Display the next best view to efficiently reach the target location.
        """
        if self.fpv is None:
            logging.warning("FPV is not set. Cannot determine next best view.")
            return

        current_index = self.get_neighbor(self.fpv)
        logging.info(f"Current neighbor index: {current_index}")

        # Define possible moves with their respective directions
        logging.info("Defining possible moves and their directions")
        moves = {
            'right': self.get_neighbor_in_direction(current_index, 'right'),
            'left': self.get_neighbor_in_direction(current_index, 'left'),
            'front': self.get_neighbor_in_direction(current_index, 'front'),
            'back': self.get_neighbor_in_direction(current_index, 'back')
        }

        # Find the direction that brings us closest to the goal
        logging.info("Finding the direction that brings closest to the goal")
        closest_direction = None
        closest_distance = float('inf')

        for direction, neighbor_index in moves.items():
            logging.info(f"Evaluating direction: {direction} with neighbor index: {neighbor_index}")
            distance_to_goal = self.calculate_goal_distance(neighbor_index, self.goal)

            if distance_to_goal < closest_distance:
                logging.info(f"New closest direction found: {direction} with distance: {distance_to_goal}")
                closest_distance = distance_to_goal
                closest_direction = direction

        if closest_direction is not None:
            # Display the image in the optimal direction
            logging.info(f"Displaying image from direction: {closest_direction}")
            self.display_img_from_id(moves[closest_direction], 'Next Best View')

            # Show the suggested direction along with current info
            logging.info(f"Next View ID: {moves[closest_direction]} || Goal ID: {self.goal}")
            print(f"Next View ID: {moves[closest_direction]} || Goal ID: {self.goal}")
            logging.info(f"Suggested Direction: {closest_direction}")
            print(f"Suggested Direction: {closest_direction}")
            logging.info(f"Distance to Goal: {closest_distance}")
            print(f"Distance to Goal: {closest_distance}")
        else:
            logging.warning("No valid direction found to move towards the goal.")

    def see(self, fpv):
        logging.info("Updating first-person view (FPV)")
        """
        Set the first-person view input
        """
        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            logging.warning("FPV is not available or has insufficient dimensions")
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        if self.screen is None:
            h, w, _ = fpv.shape
            logging.info(f"Initializing pygame screen with width: {w}, height: {h}")
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.
            """
            logging.info("Converting OpenCV image to Pygame format")
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            logging.info("Conversion complete")
            return pygame_image

        logging.info("Setting pygame window caption")
        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            logging.info(f"Game state: {self._state}")
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                logging.info("Currently in EXPLORATION phase")
                # TODO: Implement strategic exploration to improve performance
                pass

            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                logging.info("Currently in NAVIGATION phase")
                # TODO: Implement smarter navigation strategies

                if self.goal is None:
                    logging.info("Goal is not set, determining goal")
                    # Get the neighbor nearest to the front view of the target image and set it as goal
                    targets = self.get_target_images()
                    if targets and len(targets) > 0:
                        index = self.get_neighbor(targets[0])
                        self.goal = index
                        logging.info(f"Goal ID set to: {self.goal}")
                    else:
                        logging.warning("No target images available to set as goal.")

                # Key the state of the keys
                keys = pygame.key.get_pressed()
                logging.info("Checking if 'q' key is pressed for next best view")
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    logging.info("'q' key pressed, displaying next best view")
                    self.display_next_best_view()

        # Display the first-person view image on the pygame screen
        logging.info("Converting FPV to Pygame image")
        rgb = convert_opencv_img_to_pygame(fpv)
        logging.info("Blitting image to pygame screen")
        self.screen.blit(rgb, (0, 0))
        logging.info("Updating pygame display")
        pygame.display.update()


if __name__ == "__main__":
    logging.info("Starting the game with KeyboardPlayerPyGame player")
    try:
        import vis_nav_game
        vis_nav_game.play(the_player=KeyboardPlayerPyGame())
    except Exception as e:
        logging.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        print(f"An error occurred: {e}. Check 'keyboard_player.log' for more details.")