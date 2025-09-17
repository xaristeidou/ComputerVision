import os
import cv2
import supervision as sv
from ultralytics import YOLO
from supervision.utils.file import read_yaml_file
from keypoints import Keypoints

class FallDetection():
    kp = Keypoints

    def __init__(self):
        self.edge_annotator = sv.EdgeAnnotator(thickness=2)
        self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.WHITE, radius=3)
        self._setup_cls_variables()

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                results = self.model(frame, conf = 0.40)[0]
                keypoints = sv.KeyPoints.from_ultralytics(results)

                annotated_frame = self.edge_annotator.annotate(frame.copy(), keypoints)
                annotated_frame = self.vertex_annotator.annotate(annotated_frame, keypoints)

                self.analyze_pose(results, annotated_frame)

                annotated_frame = cv2.resize(annotated_frame, (1280,720))
                cv2.imshow("Fall detection", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def analyze_pose(self, poses, scene):
        fall_detected = False

        for pose in poses:
            differences = []
            left_ankle_y = pose.keypoints.xyn[0][self.kp.LEFT_ANKLE][1]
            right_ankle_y = pose.keypoints.xyn[0][self.kp.RIGHT_ANKLE][1]

            if left_ankle_y > 0 or right_ankle_y > 0:
                for kp in self.kp.keypoints_to_check:
                    kp_y = pose.keypoints.xyn[0][kp][1]

                    if kp_y > 0:
                        if left_ankle_y > 0 and right_ankle_y > 0:
                            difference_left = abs(kp_y - left_ankle_y)
                            difference_right = abs(kp_y - right_ankle_y)
                            differences.extend([difference_left, difference_right])
                        elif left_ankle_y > 0:
                            difference_left = abs(kp_y - left_ankle_y)
                            differences.extend(difference_left)
                        elif right_ankle_y > 0:                             
                            difference_right = abs(kp_y - right_ankle_y)
                            differences.extend(difference_right)

                if differences and (sum(differences) / len(differences)) < self.sensitivity:
                    fall_detected = True

                    cv2.rectangle(
                        img = scene,
                        pt1 = (int(pose.boxes.xyxy[0][0]), int(pose.boxes.xyxy[0][1])),
                        pt2 = (int(pose.boxes.xyxy[0][2]), int(pose.boxes.xyxy[0][3])),
                        color = (0,0,255),
                        thickness = 2
                    )
                    cv2.putText(
                        img = scene,
                        text = "Fall detected",
                        org = (20,20),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.6,
                        color = (0,0,255),
                        thickness = 2
                    )

        return fall_detected


    def _setup_cls_variables(self):
        config_data = read_yaml_file(os.path.dirname(__file__) + "/config.yaml")
        
        video_source = config_data["video_source"]
        video_source = os.path.dirname(os.path.dirname(__file__)) + video_source
        
        self.cap = cv2.VideoCapture(video_source)
        self.model = YOLO(config_data["detection_model"])
        self.sensitivity = config_data["sensitivity"]
            

if __name__ == "__main__":
    fall_detection = FallDetection()
    fall_detection.run()