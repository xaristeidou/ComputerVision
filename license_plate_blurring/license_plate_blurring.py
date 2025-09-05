import os
import cv2
import supervision as sv
from ultralytics import YOLO
from supervision.utils.file import read_yaml_file

class LicensePlateBlur():
    def __init__(self):
        self.blur_annotator = sv.BlurAnnotator(kernel_size=50)
        self.box_corner_annotator = sv.BoxCornerAnnotator()
        self._setup_cls_variables()

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                results = self.model(frame, conf = 0.15)[0]
                detections = sv.Detections.from_ultralytics(results)
                annotated_frame = self.blur_annotator.annotate(frame.copy(), detections)
                annotated_frame = self.box_corner_annotator.annotate(annotated_frame, detections)

                annotated_frame = cv2.resize(annotated_frame, (1280,720))
                cv2.imshow("License plate blurring", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _setup_cls_variables(self):
        config_data = read_yaml_file(os.path.dirname(__file__) + "/config.yaml")
        
        video_source = config_data["video_source"]
        video_source = os.path.dirname(os.path.dirname(__file__)) + video_source
        
        self.cap = cv2.VideoCapture(video_source)
        self.model = YOLO(config_data["detection_model"])
            

if __name__ == "__main__":
    lpb = LicensePlateBlur()
    lpb.run()