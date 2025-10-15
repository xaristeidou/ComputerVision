import os
import cv2
import supervision as sv
from ultralytics import YOLO
from supervision.utils.file import read_yaml_file

icon_paths = [
    os.path.dirname(os.path.dirname(__file__)) + "/assets/images/not_ok.png",
    os.path.dirname(os.path.dirname(__file__)) + "/assets/images/ok.png"
]
ICON_NOT_OK=0
ICON_OK=1

class VaccineCapDetection():
    def __init__(self):
        color_palette = sv.ColorPalette([sv.Color(0,182,12), sv.Color(255,57,57)])
        self.corner_box_annotator = sv.BoxCornerAnnotator(
            color=color_palette,
            thickness=2,
            corner_length=8
        )
        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette([sv.Color(50,50,50)]),
            thickness=1
        )
        self.icon_annotator = sv.IconAnnotator(
            icon_resolution_wh=(25,25),
            icon_position=sv.Position.TOP_CENTER,
            offset_xy=(0,0)
        )
        self.label_annotator = sv.LabelAnnotator(
            color=color_palette,
            text_scale=0.4,
            text_thickness=1,
            border_radius=5,
            text_padding=6,
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.line_zone_no_cap = sv.LineZone(start=sv.Point(60, 10), end=sv.Point(60, 326))
        self.line_zone_cap = sv.LineZone(start=sv.Point(390, 10), end=sv.Point(390, 326))
        self.line_zone_annotator_no_cap = sv.LineZoneAnnotator(
            thickness=2,
            color=sv.Color(255,57,57),
            text_scale=0.4,
            text_thickness=1,
            text_padding=6,
            custom_in_text="Before",
            display_out_count=False
        )
        self.line_zone_annotator_cap = sv.LineZoneAnnotator(
            thickness=2,
            color=sv.Color(0,182,12),
            text_scale=0.4,
            text_thickness=1,
            text_padding=6,
            custom_in_text="After",
            display_out_count=False
        )
        self.byte_tracker = sv.ByteTrack()
        self._setup_cls_variables()

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                results = self.model.track(frame, conf = 0.2)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = self.byte_tracker.update_with_detections(detections)

                self.line_zone_no_cap.trigger(detections)
                self.line_zone_cap.trigger(detections)

                icons = [icon_paths[ICON_OK] if class_id == "cap" else icon_paths[ICON_NOT_OK] for class_id in detections.data["class_name"]]
                annotated_frame = self.box_annotator.annotate(frame.copy(), detections)
                annotated_frame = self.corner_box_annotator.annotate(annotated_frame, detections)
                annotated_frame = self.icon_annotator.annotate(annotated_frame, detections, icons)
                annotated_frame = self.label_annotator.annotate(annotated_frame, detections)
                annotated_frame = self.line_zone_annotator_no_cap.annotate(annotated_frame, self.line_zone_no_cap)
                annotated_frame = self.line_zone_annotator_cap.annotate(annotated_frame, self.line_zone_cap)

                annotated_frame = cv2.resize(annotated_frame, (1280,720))
                cv2.imshow("Vaccine cap detection", annotated_frame)

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
    vcp = VaccineCapDetection()
    vcp.run()