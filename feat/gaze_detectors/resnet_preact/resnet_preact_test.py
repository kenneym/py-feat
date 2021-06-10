import os
import torch
import cv2
import torchvision
import numpy as np
from math import radians
from scipy.spatial.transform import Rotation
from feat.utils import get_resource_path
from .resnet_preact_model import ResnetPreactModel
from .utils import Face, Camera, HeadPoseNormalizer, FacePartsName

THREED_FACE_MODEL = os.path.join(get_resource_path(), "reference_3d_68_points_trans.npy")
MODEL_FILE = "resnet_preact_0040.pth"


def create_mpiigaze_transform():
    scale = torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255)
    transform = torchvision.transforms.Compose([
        scale,
        torch.from_numpy,
        torchvision.transforms.Lambda(lambda x: x[None, :, :]),
    ])
    return transform


class ResnetPreactTest:
    """ Eye gaze detector model
    """

    def __init__(self):
        """ Initialize the model """

        # Load the Model
        self.device = torch.device('cuda')
        if not torch.cuda.is_available():
            self.device = torch.device('cpu')

        self.gaze_estimation_model = ResnetPreactModel()
        self.gaze_estimation_model.to(self.device)

        # Load pretrained weights
        checkpoint = torch.load(os.path.join(get_resource_path(), MODEL_FILE), map_location='cpu')
        self.gaze_estimation_model.load_state_dict(checkpoint['model'])
        self.gaze_estimation_model.eval()

        # Prepare transform
        self.transform = create_mpiigaze_transform()

        # Get 3D face model points
        self.model_points = np.load(THREED_FACE_MODEL, allow_pickle=True)

        # Indices of important points for gaze detection
        self.reye_indices = np.array([36, 39])
        self.leye_indices = np.array([42, 45])
        self.mouth_indices = np.array([48, 54])
        self.eye_keys = [FacePartsName.REYE, FacePartsName.LEYE]

    def __call__(self, frame, bbox, landmarks):
        """
        Predicts the pitch and yaw of an eye gaze using the passed image and precomputed head pose

        Args:
            frame (np.ndarray): A cv2 image
            bbox (list): A face bounding box ([x1, y1, x2, y2])
            landmarks (np.ndarray): 68 2D landmarks

        Returns:
            np.ndarray: Euler angles ([pitch, yaw])
        """
        # Obtain face pose
        landmarks = landmarks.astype('float32')
        bbox = np.array(bbox)
        face = Face(bbox, landmarks)

        # Get camera intrinsics and gaze normalizer
        h, w = frame.shape[:2]
        camera_matrix = np.array([[w, 0, w // 2], [0, w, h // 2], [0, 0, 1]])
        dist_coeffs = np.zeros((5, 1), dtype="float32")  # Assuming no lens distortion
        camera = Camera(width=w, height=h, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        frame = cv2.undistort(frame, camera.camera_matrix, camera.dist_coefficients)
        normalized_camera = self.get_eye_normalized_camera()
        head_pose_normalizer = HeadPoseNormalizer(camera=camera,
                                                  normalized_camera=normalized_camera, normalized_distance=0.6)

        # Get 3D face points and eye centers
        self.estimate_head_pose(face, camera)  # using default camera intrinsics
        self.compute_3d_pose(face)
        self.compute_face_eye_centers(face)

        for key in self.eye_keys:
            eye = getattr(face, key.name.lower())
            head_pose_normalizer.normalize(frame, eye)
        self.run_mpiigaze_model(face)

        gaze_vectors = {}
        total = np.array([0., 0.])
        for key in self.eye_keys:
            eye = getattr(face, key.name.lower())
            pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
            gaze = np.array([pitch, yaw])
            gaze_vectors[key.name.lower()] = gaze
            total += gaze
        gaze_vectors['avg'] = total / 2

        return gaze_vectors

    def get_eye_normalized_camera(self) -> Camera:

        # Get average width/ height of eyes in the image
        eye_w, eye_h = 60, 36
        normalized_camera_matrix = np.array([[960., 0., 30],
                                             [0., 960., 18.],
                                             [0., 0., 1.]])

        dist_coeffs = np.zeros((5, 1), dtype="float32")  # Assuming no lens distortion
        return Camera(width=eye_w, height=eye_h, camera_matrix=normalized_camera_matrix, dist_coeffs=dist_coeffs)

    def estimate_head_pose(self, face: Face, camera: Camera) -> None:
        """Estimate the head pose by fitting 3D template model."""
        rvec = np.zeros(3, dtype=np.float)
        tvec = np.array([0, 0, 1], dtype=np.float)
        _, rvec, tvec = cv2.solvePnP(self.model_points,
                                     face.landmarks,
                                     camera.camera_matrix,
                                     camera.dist_coefficients,
                                     rvec,
                                     tvec,
                                     useExtrinsicGuess=True,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        rot = Rotation.from_rotvec(rvec)
        face.head_pose_rot = rot
        face.head_position = tvec
        face.reye.head_pose_rot = rot
        face.leye.head_pose_rot = rot

    def compute_3d_pose(self, face: Face) -> None:
        """Compute the transformed model."""
        rot = face.head_pose_rot.as_matrix()
        face.model3d = self.model_points @ rot.T + face.head_position

    def compute_face_eye_centers(self, face: Face) -> None:
        """Compute the centers of the face and eyes.
        The face center is defined as the average coordinates of the
        six points at the corners of both eyes and the mouth.
        The eye centers are defined as the average coordinates of the
        corners of each eye.
        """
        face.center = face.model3d[np.concatenate(
            [self.reye_indices, self.leye_indices,
             self.mouth_indices])].mean(axis=0)
        face.reye.center = face.model3d[self.reye_indices].mean(axis=0)
        face.leye.center = face.model3d[self.leye_indices].mean(axis=0)

    def run_mpiigaze_model(self, face: Face) -> None:
        images = []
        head_poses = []
        for key in self.eye_keys:
            eye = getattr(face, key.name.lower())
            image = eye.normalized_image
            normalized_head_pose = eye.normalized_head_rot2d
            if key == FacePartsName.REYE:
                image = image[:, ::-1]
                normalized_head_pose *= np.array([1, -1])
            image = self.transform(image)
            images.append(image)
            head_poses.append(normalized_head_pose)
        images = torch.stack(images)
        head_poses = np.array(head_poses).astype(np.float32)
        head_poses = torch.from_numpy(head_poses)

        with torch.no_grad():
            images = images.to(self.device)
            head_poses = head_poses.to(self.device)
            predictions = self.gaze_estimation_model(images, head_poses)
            predictions = predictions.cpu().numpy()

        for i, key in enumerate(self.eye_keys):
            eye = getattr(face, key.name.lower())
            eye.normalized_gaze_angles = predictions[i]
            if key == FacePartsName.REYE:
                eye.normalized_gaze_angles *= np.array([1, -1])
            eye.angle_to_vector()
            eye.denormalize_gaze_vector()
