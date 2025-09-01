import os
import cv2
import numpy as np
from typing import List
import supervision as sv
from ultralytics import YOLO
from supervision.assets import download_assets, VideoAssets
from supervision.utils.file import read_yaml_file

class VehicleLaneCounting():
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=8)
        self.byte_tracker = sv.ByteTrack()
        self.line_zone = sv.LineZone(
            sv.Point(0, 1600),
            sv.Point(3840, 1600),
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=8,
            color=sv.Color.ROBOFLOW,
            text_thickness=6,
            text_scale=2.5,
            custom_in_text="Cars in",
            custom_out_text="Cars out",
            text_offset=1,
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=2.0,
            text_thickness=5,
            border_radius=30
        )
        self._setup_cls_variables()

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()                

            if success:
                results = self.model.track(frame)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = self.byte_tracker.update_with_detections(detections)
                self.line_zone.trigger(detections)

                annotated_frame = self.box_annotator.annotate(frame.copy(), detections)

                for (zone, annotator) in zip(self.polygon_zones, self.polygon_annotators):
                    zone.trigger(detections)
                    annotator.annotate(annotated_frame)

                annotated_frame = self.line_zone_annotator.annotate(annotated_frame, self.line_zone)
                annotated_frame = self.label_annotator.annotate(annotated_frame, detections)
                
                annotated_frame = cv2.resize(annotated_frame, (1280,720))
                cv2.imshow("Vehicle Lane Counting", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _setup_cls_variables(self):
        config_data = read_yaml_file(os.path.dirname(__file__) + "/config.yaml")
        
        video_source_data = config_data["video_source"]
        if video_source_data["is_supervision_asset"]:
            asset_name = getattr(VideoAssets, video_source_data["video_path"])
            download_assets(asset_name)
            video_source = asset_name.value
        else:
            video_source = video_source_data["video_path"]
        
        self.cap = cv2.VideoCapture(video_source)
        self.model = YOLO(config_data["detection_model"])

        self.polygon_zones: List[sv.PolygonZone] = []
        self.polygon_annotators: List[sv.PolygonZoneAnnotator] = []

        for idx, zone_data in enumerate(config_data["zones"], start=1):
            polygon = np.array(zone_data["polygon"])
            color = getattr(sv.Color, zone_data["color"])
            zone = sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=[sv.Position.BOTTOM_CENTER]
            )
            annotator = sv.PolygonZoneAnnotator(
                zone=zone,
                color=color,
                thickness=8,
                text_scale=2.5,
                text_thickness=6
            )
            self.polygon_zones.append(zone)
            self.polygon_annotators.append(annotator)
            setattr(self, f"polygone_zone_{idx}", zone)
            setattr(self, f"polygone_zone_annotator_{idx}", annotator)
            

if __name__ == "__main__":
    vlc = VehicleLaneCounting()
    vlc.run()