# CDTARS
CDTARS is a breakthrough in road safety, combining YOLOv8 detection, custom model training, and real-time video analysis for vehicle tracking and abnormality recognition. Overcoming challenges in resources and data, it enhances traffic flow, prevents accidents, and supports rapid emergency responses, paving the way for safer, smarter roads.
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Vehicle and Accident Detection with YOLOv8</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        pre {
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            padding: 10px;
            overflow-x: auto;
        }
        code {
            font-family: Consolas, "Courier New", Courier, monospace;
            color: #d63384;
        }
        h1, h2, h3, h4 {
            color: #333;
        }
        .section {
            margin-bottom: 40px;
        }
    </style>
</head>
<body>

    <h1>Vehicle Detection and Accident Detection with YOLOv8</h1>
    <p>This code implements vehicle detection and accident detection using the YOLOv8 model. The tutorial covers loading a pre-trained model for vehicle detection and training a custom model for accident detection using Roboflow datasets.</p>

    <div class="section">
        <h2>Normal Vehicle Detection</h2>
        <pre><code>!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

!pip install -q gdown
%cd {HOME}
!gdown '1IrSdzONaHQnVFFUDqGHxq9T4nu75rBcz'

SOURCE_VIDEO_PATH = "/content/Car.mp4"</code></pre>

        <h3>Install YOLOv8</h3>
        <pre><code>!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()</code></pre>
    </div>

    <div class="section">
        <h3>Install Roboflow Supervision</h3>
        <pre><code>!pip install supervision

import supervision as sv
print("supervision.__version__:", sv.__version__)</code></pre>

        <h3>Load Pre-Trained YOLOv8 Model</h3>
        <pre><code>MODEL = "yolov8x.pt"

from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()</code></pre>
    </div>

    <div class="section">
        <h3>Predict and Annotate Single Frame</h3>
        <pre><code># dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names
selected_classes = [2, 3, 5, 7]

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
frame = next(iter(generator))

results = model(frame, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[np.isin(detections.class_id, selected_classes)]

labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]

annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

%matplotlib inline
sv.plot_image(annotated_frame, (16,16))</code></pre>
    </div>

    <div class="section">
        <h3>Process the Whole Video</h3>
        <pre><code># settings
LINE_START = sv.Point(50, 1500)
LINE_END = sv.Point(3840-50, 1500)
TARGET_VIDEO_PATH = "/content/sample.mp4"

# create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

def callback(frame: np.ndarray, index:int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, selected_classes)]
    detections = byte_tracker.update_with_detections(detections)

    labels = [f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, tracker_id in detections]

    annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    line_zone.trigger(detections)

    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

sv.process_video(source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=callback)</code></pre>
    </div>

    <div class="section">
        <h2>Accident Detection</h2>
        <pre><code>!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

!pip install -q gdown
%cd {HOME}
!gdown '1wPYpRYZuSTFO2og94C2MshhJMwRRfNwu'</code></pre>

        <h3>Install YOLOv8 for Accident Detection</h3>
        <pre><code>!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()</code></pre>

        <h3>Load Custom Model for Accident Detection</h3>
        <pre><code>MODEL = "/content/best.pt"

from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()</code></pre>
    </div>

    <p>Continue experimenting with your YOLOv8 models for both vehicle and accident detection!</p>

</body>
</html>
