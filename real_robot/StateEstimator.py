"""
No cv2.aruco module found, install opencv-contrib-python package:
https://stackoverflow.com/questions/45972357/python-opencv-aruco-no-module-named-cv2-aruco
Detection logic:
https://docs.opencv.org/4.11.0/d5/dae/tutorial_aruco_detection.html
"""

import cv2
import os
import numpy as np
from collections import deque
from copy import deepcopy
from RealSenseCamera import RealSenseCamera
from tf_utils import to_array, axisangle2quat, quat2axisangle, tf_inverse, tf_combine, matrix_to_quaternion, adaptive_filter_pose, quat_mul
import time


class ObjPoseEstimator:
    def __init__(self, marker_size, num_history=1, cam_ext_path="cal_results/franka2cam.txt"):
        # Connect Camera first
        self.camera = RealSenseCamera(img_w=848, img_h=480, fps=60, enable_depth=False)

        # Setup aruco marker parameters
        self.marker_size = marker_size
        self.cam_matrix = self.camera.color_intrin_mat
        self.dist_coeffs = self.camera.distortion_coef
        self.axes_length = 1.5*self.marker_size
        
        # State tracking initialization; Marker_points can not be changed!
        self.marker_points =  np.array([[-marker_size / 2, marker_size / 2, 0],
                                        [marker_size / 2, marker_size / 2, 0],
                                        [marker_size / 2, -marker_size / 2, 0],
                                        [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        self.last_corners = None
        self.last_ids = None
        self.last_poses = []
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        
        # Visualization parameters
        self.marker_color = (0, 255, 0)
        self.axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        self.Panda2Cam = None
        if cam_ext_path is not None:
            self.Panda2Cam = self.get_pandabase2camera(cam_ext_path)

    
    def _estimate_pose(self, corners):
        """
        SolvePnP problem (Only R, t are unknowns):
        [u, v] = Projection([X, Y, Z], R, t, K)
        error = ||imagePoints - Project(objectPoints, R, t, K)||²

        Returns:
        - success: True if the pose estimation is successful
        - rvec: Rotation vector (Rodrigues) (3, 1)
        - tvec: Translation vector (3, 1)
        """
        success, rvec, tvec = cv2.solvePnP(self.marker_points, corners, self.cam_matrix, self.dist_coeffs)
        return success, rvec, tvec
    

    def get_pandabase2camera(self, camera_extrinsics_path):
        # .txt file containing the extrinsics of the camera
        assert os.path.exists(camera_extrinsics_path), f"Camera extrinsics file does not exist! Given path: {camera_extrinsics_path}"
        with open(camera_extrinsics_path, 'r') as f:
            lines = f.readlines()
            # Extract the rotation and translation values
            panda2camera = np.array([list(map(float, line.split())) for line in lines[:]]).reshape(4, 4)
        panda2camera = (matrix_to_quaternion(panda2camera[:3, :3]), panda2camera[:3, 3].flatten())
        return panda2camera

    
    def update(self):
        color_frame, _ = self.camera.get_rgbd_frame()
        self.color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(self.color_frame, cv2.COLOR_BGR2GRAY)
        self.last_corners, self.last_ids, rejected = self.detector.detectMarkers(gray)
        
        if self.last_ids is None:
            return None

        # Gather all detected qr codes poses
        self.last_poses.clear()
        for i, marker_id in enumerate(self.last_ids):
            try:
                marker_corners = self.last_corners[i]
                success, rvec, tvec = self._estimate_pose(marker_corners)
                self.last_poses.append((rvec, tvec))
            except KeyError:
                continue

    
    def render(self):
        img = self.color_frame.copy()
        if self.last_corners is not None and self.last_ids is not None:
            # Draw a square around the markers and its id number
            cv2.aruco.drawDetectedMarkers(img, self.last_corners, self.last_ids, borderColor=self.marker_color)
            # Draw the axes of the markers; tvec also uses self.marker_size as a scale, if we scale tvec and axes len together, it will not have any effect.
            for i, marker_id in enumerate(self.last_ids):
                rvec, tvec = self.last_poses[i]
                cv2.drawFrameAxes(img, self.cam_matrix, self.dist_coeffs, rvec, tvec, self.axes_length, thickness=2)
        return img


class CubePoseEstimator(ObjPoseEstimator):
    def __init__(self, 
                 marker_size=0.0415, 
                 cube_size=0.049, 
                 target_size=0.069, 
                 num_history=1, 
                 cam_ext_path="cal_results/franka2cam.txt",
                 default_target_pose=None):
        super().__init__(marker_size, num_history=num_history, cam_ext_path=cam_ext_path)
        
        self.cube_size = cube_size
        self.target_size = target_size
        self.axis_length = cube_size * 0.8
        self.info_cube = {
            "QrCodes2Obj": {},
            "Cam2Obj_cur": [],
            "Panda2Obj_cur": [],
            "Cam2Obj_buf": deque(maxlen=num_history),
            "Panda2Obj_buf": deque(maxlen=num_history),
        }
        self.info_target = deepcopy(self.info_cube)
        self.default_target_pose = default_target_pose if default_target_pose is None else [np.array(v) for v in default_target_pose]  # [quat, pos]
        
        # Cube face configuration (pos, rot xyz)
        base2posZ = (to_array([0, 0, cube_size/2]), axisangle2quat(to_array([0., 0., 0.])))
        base2negZ = (to_array([0, 0, -cube_size/2]), axisangle2quat(to_array([np.pi, 0., 0.])))
        base2posX = (to_array([cube_size/2, 0, 0]), axisangle2quat(to_array([0., np.pi/2., 0.])))
        base2negX = (to_array([-cube_size/2, 0, 0]), axisangle2quat(to_array([0., -np.pi/2., 0.])))
        base2posY = (to_array([0, cube_size/2, 0]), axisangle2quat(to_array([-np.pi/2., 0., 0.])))
        base2negY = (to_array([0, -cube_size/2, 0]), axisangle2quat(to_array([np.pi/2., 0., 0.])))

        target2posZ = (to_array([0, 0, target_size/2]), axisangle2quat(to_array([0., 0., 0.])))
        target2posX = (to_array([target_size/2, 0, 0]), axisangle2quat(to_array([0., np.pi/2., 0.])))
        target2negY = (to_array([0, -target_size/2, 0]), axisangle2quat(to_array([np.pi/2., 0., 0.])))
        
        base2QrCodes = [base2posZ, base2negY, base2posX, base2posY, base2negX, base2negZ] # qr code id 0~5
        target2QrCodes = [target2posZ, target2negY, target2posX]
        for i, (pos, quat) in enumerate(base2QrCodes):
            self.info_cube["QrCodes2Obj"][i] = tf_inverse(quat, pos)
        for j, (pos, quat) in enumerate(target2QrCodes): # target id 678
            self.info_target["QrCodes2Obj"][j+len(base2QrCodes)] = tf_inverse(quat, pos)

        # Misc parameters for smoothness
        self.trans_thresh = 0.025
        self.rot_thresh = np.deg2rad(15)
        self.trans_alpha = 0.8 # Warning: reduce alpha will make the pose smoother, but will introduce observation lag and policy confusing.
        self.rot_alpha = 0.1
        # There is a default offset that can not mitigate by camera calibration, so we need to adjust it manually.
        # Watchout the z value. If it is lower (offset; e.g. -0.005m) the arm will think the cube does not get lift and regrasp.
        self.panda2cube_residual_pos = np.array([-0.02, 0., 0.015])
        self.panda2target_residual_pos = np.array([-0.025, -0.02, 0.0])
        self.panda2cube_min = [None, None, cube_size / 2 - 0.013] # medal base is 0.013m
        self.panda2target_min = [None, None, target_size / 2 - 0.013]
        
        self.warmup_update()

    
    def update(self):
        super().update()
        
        # Assign poses to each object
        for info, residual_pos, pos_min in zip([self.info_cube, self.info_target], 
                                               [self.panda2cube_residual_pos, self.panda2target_residual_pos],
                                               [self.panda2cube_min, self.panda2target_min]):
            # Clear previous poses
            info["Cam2Obj_cur"].clear()
            info["Panda2Obj_cur"].clear()
            if self.last_ids is None:
                continue
        
            for i, id_np in enumerate(self.last_ids):
                id = id_np.item()
                rvec, tvec = self.last_poses[i]
                rvec, tvec = rvec.flatten(), tvec.flatten()
                Cam2QrCode = (axisangle2quat(rvec), tvec)

                if id in info["QrCodes2Obj"] and len(info["Cam2Obj_cur"])==0: # We only use one face for now
                    QrCode2Obj = info["QrCodes2Obj"][id]
                    Cam2Obj = tf_combine(*Cam2QrCode, *QrCode2Obj)
                    if len(info["Cam2Obj_buf"]) > 0:
                        prev_pose = info["Cam2Obj_buf"][-1]
                        Cam2Obj_filtered = adaptive_filter_pose(Cam2Obj, prev_pose,
                                                                trans_alpha=self.trans_alpha, rot_alpha=self.rot_alpha,
                                                                trans_thresh=self.trans_thresh, rot_thresh=self.rot_thresh)
                    else:
                        Cam2Obj_filtered = Cam2Obj
                    
                    # Append the filtered (or fresh) pose to the current list.
                    info["Cam2Obj_cur"].append(Cam2Obj_filtered)
                    if self.Panda2Cam is not None:
                        Panda2Obj_quat, Panda2Obj_pos = tf_combine(*self.Panda2Cam, *Cam2Obj_filtered)
                        # Adjust position manually to mitigate the offset
                        Panda2Obj_pos = Panda2Obj_pos + residual_pos
                        # Apply min limit if specified (e.g. keep above the table; necessary! TODO: clip upper limit as well)
                        for i in range(len(Panda2Obj_pos)): 
                            Panda2Obj_pos[i] = max(Panda2Obj_pos[i], pos_min[i]) if pos_min[i] is not None else Panda2Obj_pos[i] 
                        Panda2Obj = (Panda2Obj_quat, Panda2Obj_pos)
                        info["Panda2Obj_cur"].append(Panda2Obj)
        
            # Update poses
            if len(info["Cam2Obj_cur"]) > 0:
                Cam2Obj_cur = info["Cam2Obj_cur"][-1]
                info["Cam2Obj_buf"].append(Cam2Obj_cur)
                if self.Panda2Cam is not None:
                    Panda2Obj_cur = info["Panda2Obj_cur"][-1]
                    info["Panda2Obj_buf"].append(Panda2Obj_cur)


    def render(self, draw=False):
        img = self.color_frame.copy()
        
        if self.last_ids is not None:
            # Draw the axes of the markers for all detected objects
            for info in [self.info_cube, self.info_target]:
                if len(info["Cam2Obj_cur"])>0:
                    for i, (quat, tvec) in enumerate(info["Cam2Obj_cur"]):
                        rvec = quat2axisangle(quat)
                        cv2.drawFrameAxes(img, self.cam_matrix, self.dist_coeffs, rvec, tvec, self.axes_length, thickness=2)

        if draw:
            cv2.imshow('Cube Tracking', img)
            cv2.waitKey(1)

        return img


    def warmup_update(self, num_frames=20):
        """
        Warmup the estimator by running several update cycles to avoid initial errors.
        """
        for _ in range(num_frames):
            self.update()
    
    
    def _avg_poses(self, poses):
        pass

    def get_panda2cam(self):
        return self.Panda2Cam


    def get_cube_last_pose(self):
        return self.info_cube["Panda2Obj_buf"][-1] if len(self.info_cube["Panda2Obj_buf"])>0 else None
    

    def get_cube_pose(self):
        return self.info_cube["Panda2Obj_cur"][-1] if len(self.info_cube["Panda2Obj_cur"])>0 else None
    
    
    def get_target_last_pose(self):
        if self.default_target_pose is not None:
            return self.default_target_pose
        return self.info_target["Panda2Obj_buf"][-1] if len(self.info_target["Panda2Obj_buf"])>0 else None
    

    def get_target_pose(self):
        if self.default_target_pose is not None:
            return self.default_target_pose
        return self.info_target["Panda2Obj_cur"][-1] if len(self.info_target["Panda2Obj_cur"])>0 else None
    

    def stop(self):
        cv2.destroyAllWindows()
        self.camera.stop_stream()


class CupPoseEstimator(CubePoseEstimator):
    def __init__(self, 
                 marker_size=0.0415,
                 cup_size=[0.0375, 0.0375, 0.1], 
                 target_size=[0.0375, 0.0375, 0.1], 
                 num_history=1, 
                 cam_ext_path="cal_results/franka2cam.txt",
                 default_target_pose=None):
        """
        Detect the Cup base pose since the cup base in isaacgym is at the bottom of the cup.
        """

        ObjPoseEstimator.__init__(self, marker_size, num_history=num_history, cam_ext_path=cam_ext_path)
        
        self.cup_size = cup_size
        self.target_size = target_size
        self.thickness = 0.005
        self.axis_length = cup_size[0] * 0.8
        self.info_cube = {
            "QrCodes2Obj": {},
            "Cam2Obj_cur": [],
            "Panda2Obj_cur": [],
            "Cam2Obj_buf": deque(maxlen=num_history),
            "Panda2Obj_buf": deque(maxlen=num_history),
        }
        self.info_target = deepcopy(self.info_cube)

        target2qr1 = (to_array([0., -target_size[0], target_size[2]/2]), axisangle2quat(to_array([np.pi/2, 0., 0.])))
        target2QrCodes = [target2qr1]

        start_id = 9
        for j, (pos, quat) in enumerate(target2QrCodes): # target id 9
            self.info_target["QrCodes2Obj"][start_id+j] = tf_inverse(quat, pos)

        # Misc parameters for smoothness
        self.trans_thresh = 0.025
        self.rot_thresh = np.deg2rad(15)
        self.trans_alpha = 0.8 # Warning: reduce alpha will make the pose smoother, but will introduce observation lag and policy confusing.
        self.rot_alpha = 0.1
        # There is a default offset that can not mitigate by camera calibration, so we need to adjust it manually.
        # Watchout the z value. If it is lower (offset; e.g. -0.005m) the arm will think the cube does not get lift and regrasp.
        self.panda2cube_residual_pos = np.array([0., 0., 0.])
        self.panda2target_residual_pos = np.array([0., 0., 0.0])
        self.panda2cube_min = [None, None, 0.] # medal base is 0.013m
        self.panda2target_min = [None, None, -0.013]

        # Default target pose
        self.default_target_pose = default_target_pose if default_target_pose is None else [np.array(v) for v in default_target_pose]  # [quat, pos]

        self.warmup_update()



class DrawerHandlePoseEstimator(CubePoseEstimator):
    def __init__(self, 
                 marker_size=0.051,
                 num_history=1, 
                 cam_ext_path="cal_results/franka2cam.txt",
                 default_target_pose=None):
        """
        Detect the Cup base pose since the cup base in isaacgym is at the bottom of the cup.
        """

        ObjPoseEstimator.__init__(self, marker_size, num_history=num_history, cam_ext_path=cam_ext_path)
        
        self.axis_length = marker_size * 0.8
        self.info_cube = {
            "QrCodes2Obj": {},
            "Cam2Obj_cur": [],
            "Panda2Obj_cur": [],
            "Cam2Obj_buf": deque(maxlen=num_history),
            "Panda2Obj_buf": deque(maxlen=num_history),
        }
        self.info_target = deepcopy(self.info_cube)

        Qr1totarget = (to_array([0.155, 0.05, 0.06]), 
                       quat_mul(axisangle2quat(to_array([-np.pi/2, 0., 0.])),
                                axisangle2quat(to_array([0., 0., -np.pi/2]))))
        QrCodes2target = [Qr1totarget]

        start_id = 11
        for j, (pos, quat) in enumerate(QrCodes2target): # target id 11
            self.info_target["QrCodes2Obj"][start_id+j] = quat, pos

        # Misc parameters for smoothness
        self.trans_thresh = 0.025
        self.rot_thresh = np.deg2rad(15)
        self.trans_alpha = 0.1 # Warning: reduce alpha will make the pose smoother, but will introduce observation lag and policy confusing.
        self.rot_alpha = 0.1
        # There is a default offset that can not mitigate by camera calibration, so we need to adjust it manually.
        # Watchout the z value. If it is lower (offset; e.g. -0.005m) the arm will think the cube does not get lift and regrasp.
        self.panda2cube_residual_pos = np.array([0., 0., 0.])
        self.panda2target_residual_pos = np.array([0., 0., 0.])
        self.panda2cube_min = [None, None, 0.] # medal base is 0.013m
        self.panda2target_min = [None, None, 0.57-0.013]

        # Default target pose
        self.default_target_pose = default_target_pose if default_target_pose is None else [np.array(v) for v in default_target_pose]  # [quat, pos]
        
        self.warmup_update()



if __name__=="__main__":
    # TODO: save initial sequence of object estimations at the beginning to make sure the pose is stable
    
    # Initialize with cube parameters (size should match actual cube)
    # tracker = CubePoseEstimator(
    #     marker_size=0.0415,  # Physical size of AR markers
    #     cube_size=0.05, # Physical size of cube
    #     target_size=0.07,
    #     num_history=2, 
    #     cam_ext_path="cal_results/franka2cam.txt",
    # )

    # tracker = CupPoseEstimator(
    #     marker_size=0.0415,  # Physical size of AR markers
    #     cup_size=[0.0375, 0.0375, 0.1], # Physical size of cup
    #     target_size=[0.0375, 0.0375, 0.1],
    #     num_history=2, 
    #     cam_ext_path="cal_results/franka2cam.txt",
    # )

    tracker = DrawerHandlePoseEstimator(
        marker_size=0.051,  # Physical size of AR markers
        num_history=2, 
        cam_ext_path="cal_results/franka2cam.txt",
    )

    velocity_buf = deque(maxlen=100)

    # Process frame and visualize
    while True:
        # ~70 Hz loop
        start_time = time.perf_counter()
        tracker.update()
        cubeA_pose = tracker.get_cube_pose()
        target_pose = tracker.get_target_pose()
        print(f"FPS: {1/(time.perf_counter()-start_time):.3f}")

        if cubeA_pose is not None:
            print(f"Cube Pose: {cubeA_pose[1]} | Cube Quat: {cubeA_pose[0]}")
            if len(tracker.info_cube["Panda2Obj_buf"])>1:
                velocity_buf.append((tracker.info_cube["Panda2Obj_buf"][-1][1] - tracker.info_cube["Panda2Obj_buf"][-2][1]) / (time.perf_counter()-start_time)) 
                if len(velocity_buf) == 100:
                    velocity = np.mean(velocity_buf, axis=0)
                    vel_norm = np.linalg.norm(velocity)
                    print(f"Cube Velocity: {vel_norm:.4f} m/s")
        if target_pose is not None:
            print(f"Target Pos: {target_pose[1]} | Target Quat: {target_pose[0]}\n")
              
        visual_frame = tracker.render(draw=True)