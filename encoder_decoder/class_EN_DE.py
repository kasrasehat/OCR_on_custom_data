import torch
import torchvision.transforms as transforms
from Networks_EN import Encoder, FeatureExtractor, AttnDecoderRNN, EncoderRNN # Import your model classes here
import tokenizer
import cv2
import numpy as np


class ModelLoader:
    def __init__(self, device, feature_extractor_path, decoder_path, height=480, width=480, batch_size=1):
        self.device = device
        self.batch_size = batch_size

        # Initialize models
        self.encoderCNN = Encoder(base_filters=16, adaptive_pool_size=(60, 60)).to(device)
        self.feature_extractor = FeatureExtractor(base_filters=16, reduced_filters=8, output_features=12).to(device)
        self.encoderRNN = EncoderRNN(input_size=128, hidden_size=128, dropout_p=0).to(device)
        self.decoderRNN = AttnDecoderRNN(hidden_size=128, output_size=128, dropout_p=0).to(device)

        # Load model weights
        self.load_model_weights(feature_extractor_path, decoder_path)

        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            self.expand
        ])

    def load_model_weights(self, feature_extractor_path, decoder_path):
        # Load feature extractor weights
        if feature_extractor_path is not None:
            checkpoint = torch.load(feature_extractor_path, map_location=self.device)
            self.feature_extractor.load_state_dict(checkpoint['state_dict feature_extractor'])

        # Load decoder weights
        if decoder_path is not None:
            checkpoint = torch.load(decoder_path, map_location=self.device)
            self.encoderCNN.load_state_dict(checkpoint['state_dict encoderCNN'])
            self.encoderRNN.load_state_dict(checkpoint['state_dict encoderRNN'])
            self.decoderRNN.load_state_dict(checkpoint['state_dict decoderRNN'])

    def expand(self, patch):
        patch = torch.unsqueeze(patch, 0)
        return patch.type(torch.float32)

    def process_image(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to read the image")

        # Convert image to RGB (OpenCV uses BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the transform
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension and send to device

        # Process the image through the models
        feature_map, feature_vector = self.encoderCNN(image)
        features = self.feature_extractor(image)

        # Specific operations on features
        features[:, :8] *= 720
        features[:, 8] = (features[:, 8] * 2 - 1) * 180
        features[:, 10:12] = (features[:, 10:12] * 2 - 1) * 360

        encoderRNN_init_hidden = torch.cat((feature_vector.unsqueeze(0), features.unsqueeze(0)), dim=2)
        encoderRNN_output, decoderRNN_init_hidden = self.encoderRNN(
            feature_map.view(self.batch_size, feature_map.size()[1],
                             feature_map.size()[2] * feature_map.size()[3]).permute(0, 2, 1),
                             encoderRNN_init_hidden)
        output_ctc, _, _ = self.decoderRNN(encoderRNN_output, decoderRNN_init_hidden, None)

        # Decode the output
        _, topi = output_ctc.topk(1)
        decoded_ids = topi.squeeze().permute(1, 0)
        decoded_words = []
        for vector in decoded_ids:
            for idx in vector:
                if idx.item() == 1:
                    break
                decoded_words.append(tokenizer.ind2char(idx.item()))

        output_passage = ''.join(char for char in decoded_words)
        output_passage = self.replace_tokens(output_passage)
        # Extract the first 8 elements of the features as coordinates
        person_image_corners = self.extract_coordinates(features[:, :8])
        self.draw_rectangle(img, vertices)
        draw_rectangle(img, self.transform_vertices([(0, 40), (720, 40), (720, 680), (0, 680)],
                                               self.get_transformation_matrix(-labels.iloc[matching_row].ROTATION * 180,
                                                                         labels.iloc[matching_row].SCALE, literal_eval(
                                                       labels.iloc[matching_row].TRANSPORT))))

        return output_passage, features, person_image_corners

    @staticmethod
    def replace_tokens(passage):
        passage = passage.replace('a', 'روی کارت ملی')
        passage = passage.replace('b', 'پشت کارت ملی')
        passage = passage.replace('c', 'شناسنامه جدید')
        passage = passage.replace('d', 'شناسنامه قدیم')
        return passage

    @staticmethod
    def extract_coordinates(feature_slice):
        # Reshape the feature slice into four pairs of coordinates
        coordinates = feature_slice.view(-1, 4, 2)

        # Applying any necessary transformations to match the desired format
        # Assuming each pair of numbers is a coordinate in the form (x, y)
        corners = [(float(coordinates[0][i][0]), float(coordinates[0][i][1])) for i in range(4)]

        return corners

    @staticmethod
    def get_transformation_matrix(rotation, scale, translation):
        """
        Calculate the transformation matrix based on the given rotation, scale, and translation.

        Parameters:
            rotation (float): The rotation angle in degrees.
            scale (float): The scale factor.
            translation (tuple): A tuple (tx, ty) representing the x and y translation.

        Returns:
            np.array: The 3x3 transformation matrix.
        """
        # Convert rotation angle to radians
        theta = np.radians(rotation)

        # Create individual transformation matrices
        # Rotation matrix
        T_rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Scale matrix
        T_scale = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])

        # Translation matrix
        tx, ty = translation
        T_trans = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

        # Combine transformations
        T = np.dot(T_trans, np.dot(T_rot, T_scale))

        return T

    @staticmethod
    def transform_vertices(vertices, transformation_matrix):
        """
        Apply a transformation matrix to a set of vertices.

        Parameters:
            vertices (list of tuples): The vertices to be transformed.
            transformation_matrix (np.array): The transformation matrix.

        Returns:
            list of tuples: Transformed vertices.
        """
        # Convert the list of tuples to a numpy array of shape (4, 2)
        vertices_array = np.array(vertices) - np.ones_like(vertices) * 360

        # Homogeneous coordinates: Add a column of ones to the vertices array
        vertices_homogeneous = np.hstack([vertices_array, np.ones((vertices_array.shape[0], 1))])

        # Apply the transformation matrix
        transformed_vertices = np.dot(transformation_matrix, vertices_homogeneous.T).T

        # Convert back to Cartesian coordinates
        transformed_vertices_cartesian = transformed_vertices[:, :2] / transformed_vertices[:, 2][:, np.newaxis]
        transformed_vertices_cartesian = transformed_vertices_cartesian + np.ones_like(
            transformed_vertices_cartesian) * 360

        # Rounding the coordinates to integer values
        transformed_vertices_cartesian = np.round(transformed_vertices_cartesian).astype(int)

        # Convert the numpy array back to a list of tuples
        transformed_vertices_list = [tuple(coord) for coord in transformed_vertices_cartesian]

        return transformed_vertices_list

    @staticmethod
    def draw_rectangle(image_path, vertices):
        """
        Draws a rectangle on the image based on given vertices.

        Parameters:
            image_path (str): Path to the image file
            vertices (list): List of four vertices in the format [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """

        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if image is None:
            print("Could not open or find the image.")
            return


