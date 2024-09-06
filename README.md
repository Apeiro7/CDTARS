<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Vehicle and Accident Detection with YOLOv8</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f8ff;
            margin: 20px;
            color: #333;
        }

        h1, h2, h3, h4 {
            color: #2e8b57;
        }

        pre {
            background-color: #272822;
            border: 1px solid #444;
            padding: 15px;
            border-radius: 10px;
            color: #f8f8f2;
            overflow-x: auto;
        }

        code {
            font-family: 'Courier New', Courier, monospace;
            color: #66d9ef;
        }

        a {
            color: #ff6347;
        }

        .section {
            margin-bottom: 40px;
            background-color: #fff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        h1, h2, h3 {
            border-bottom: 2px solid #dcdcdc;
            padding-bottom: 5px;
        }

        .important {
            color: #ff4500;
            font-weight: bold;
        }

        .container {
            max-width: 1100px;
            margin: auto;
            padding: 20px;
        }

        p, li {
            font-size: 1.1em;
            line-height: 1.6;
        }
    </style>
</head>

<body>

    <div class="container">

        <h1>Vehicle and Accident Detection with YOLOv8</h1>
        <p>CDTARS is a breakthrough in road safety, combining YOLOv8 detection, custom model training, and real-time video
            analysis for vehicle tracking and abnormality recognition.</p>
        <p>In this tutorial, you will learn to implement YOLOv8 for both normal vehicle detection and accident detection,
            with guidance on setting up models, running predictions, and processing video frames.</p>

        <div class="section">
            <h2>🚗 Normal Vehicle Detection</h2>
            <p>Start by detecting vehicles such as cars, motorcycles, buses, and trucks using YOLOv8's pre-trained models.
            </p>
            <pre><code># Check system and setup environment
!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

# Download video
!pip install -q gdown
%cd {HOME}
!gdown '1IrSdzONaHQnVFFUDqGHxq9T4nu75rBcz'

SOURCE_VIDEO_PATH = "/content/Car.mp4"</code></pre>

            <h3>📦 Install YOLOv8</h3>
            <pre><code>!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()</code></pre>
        </div>

        <div class="section">
            <h3>🔧 Install Roboflow Supervision</h3>
            <pre><code>!pip install supervision

import supervision as sv
print("supervision.__version__:", sv.__version__)</code></pre>

            <h3>⚙️ Load Pre-Trained YOLOv8 Model</h3>
            <pre><code># Load the YOLOv8 model for vehicle detection
MODEL = "yolov8x.pt"

from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()</code></pre>
        </div>

        <div class="section">
            <h3>🖼️ Predict and Annotate Single Frame</h3>
            <pre><code># Map class_id to class_name
CLASS_NAMES_DICT = model.model.names

# Classes of interest: car, motorcycle, bus, truck
selected_classes = [2, 3, 5, 7]

# Generate frames and annotate
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
frame = next(iter(generator))

results = model(frame, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[np.isin(detections.class_id, selected_classes)]

labels = [f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]

# Annotate frame
annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

%matplotlib inline
sv.plot_image(annotated_frame, (16,16))</code></pre>
        </div>

        <div class="section">
            <h3>🎥 Process the Whole Video</h3>
            <pre><code># Video tracking and annotation
LINE_START = sv.Point(50, 1500)
LINE_END = sv.Point(3840-50, 1500)

TARGET_VIDEO_PATH = "/content/sample.mp4"
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
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
            <h2>💥 Accident Detection</h2>
            <p>Now, we train a custom model for accident detection. This allows us to detect both moderate and severe accidents from video footage.</p>
            <pre><code>!nvidia-smi

import os
HOME = os.getcwd()
print(HOME)

!pip install -q gdown
%cd {HOME}
!gdown '1wPYpRYZuSTFO2og94C2MshhJMwRRfNwu'</code></pre>

            <h3>⚙️ Install YOLOv8 for Accident Detection</h3>
            <pre><code>!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()</code></pre>

            <h3>📄 Load Custom Model for Accident Detection</h3>
            <pre><code># Load the custom YOLOv8 model for accident detection
MODEL = "/content/best.pt"

from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()</code></pre>
        </div>

        <p>🎉 You have now implemented both vehicle and accident detection using YOLOv8! Happy experimenting!</p>

    </div>

</body>

</html>
