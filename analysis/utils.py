"""
Traffic analysis processing utilities.

Updates YOLOv8 video processing and image analysis functions to use 4 approaches (North, South, East, West).
"""
import cv2
import os
import uuid
import random
import numpy as np
import logging
from collections import defaultdict
from django.conf import settings

logger = logging.getLogger(__name__)


def _compute_iou(box_a, box_b):
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return (inter_area / union) if union > 0 else 0.0


def get_upload_path():
    """Return the absolute path to the uploads directory."""
    path = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(path, exist_ok=True)
    return path


def get_result_path():
    """Return the absolute path to the results directory."""
    path = os.path.join(settings.MEDIA_ROOT, 'results')
    os.makedirs(path, exist_ok=True)
    return path


def generate_unique_filename(filename):
    """Generate a unique filename using UUID."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'jpg'
    return f"{uuid.uuid4().hex}.{ext}"


def get_direction(x, y, width, height):
    """Determine quadrant (North, South, East, West) based on diagonals."""
    dy1 = y - (height / width) * x
    dy2 = y - height + (height / width) * x

    if dy1 < 0 and dy2 < 0:
        return 'North'
    elif dy1 >= 0 and dy2 >= 0:
        return 'South'
    elif dy1 < 0 and dy2 >= 0:
        return 'East'
    else:
        return 'West'


def process_image(file_path, output_path):
    """
    Process a traffic image for vehicle detection into 4 directions.
    """
    DIRECTIONS = ['North', 'South', 'East', 'West']

    VEHICLE_COLORS = {
        'car': (66, 135, 245),
        'truck': (32, 201, 151),
        'bus': (13, 202, 240),
        'motorcycle': (255, 193, 7),
    }

    dir_counts = {
        d: {'total': 0, 'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
        for d in DIRECTIONS
    }

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not read image file: {file_path}")

    height, width = image.shape[:2]

    # Draw diagonals
    cv2.line(image, (0, 0), (width, height), (255, 255, 255), 2)
    cv2.line(image, (0, height), (width, 0), (255, 255, 255), 2)

    # Simulated vehicle detection
    num_vehicles = random.randint(8, 20)

    for _ in range(num_vehicles):
        vehicle_type = random.choice(list(VEHICLE_COLORS.keys()))
        
        if vehicle_type == 'car':
            w, h = random.randint(80, 120), random.randint(40, 60)
        elif vehicle_type == 'motorcycle':
            w, h = random.randint(40, 60), random.randint(30, 50)
        elif vehicle_type == 'truck':
            w, h = random.randint(120, 180), random.randint(60, 90)
        else:
            w, h = random.randint(150, 200), random.randint(70, 100)

        x = random.randint(0, max(0, width - w))
        y = random.randint(0, max(0, height - h))
        
        cx, cy = x + w//2, y + h//2
        direction = get_direction(cx, cy, width, height)

        cv2.rectangle(image, (x, y), (x + w, y + h), VEHICLE_COLORS[vehicle_type], 2)

        confidence = random.uniform(0.85, 0.98)
        label = f"{vehicle_type.capitalize()} {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x, y - text_height - 10), (x + text_width + 5, y),
                      VEHICLE_COLORS[vehicle_type], -1)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        dir_counts[direction]['total'] += 1
        dir_counts[direction][vehicle_type] += 1

    cv2.imwrite(output_path, image)

    total_vehicles = sum(d['total'] for d in dir_counts.values())
    image_area = image.shape[0] * image.shape[1]
    scaled_vehicle_area = 5000 * (image_area / (640 * 480))
    density = min(100, (total_vehicles * scaled_vehicle_area / image_area) * 100 * 2)

    MAX_VEHICLES_PER_DIR = 15
    dir_densities = {}
    for d_name, stats in dir_counts.items():
        lane_total = stats['total']
        dir_densities[d_name] = min(100.0, (lane_total / MAX_VEHICLES_PER_DIR) * 100.0)

    return {
        'total_vehicles': total_vehicles,
        'density': density,
        'vehicle_counts': {
            'car': sum(d['car'] for d in dir_counts.values()),
            'truck': sum(d['truck'] for d in dir_counts.values()),
            'bus': sum(d['bus'] for d in dir_counts.values()),
            'motorcycle': sum(d['motorcycle'] for d in dir_counts.values()),
        },
        'lane_counts': {k: v['total'] for k, v in dir_counts.items()},
        'lane_densities': dir_densities,
        'time_series_data': [{
            'timestamp': 0.0,
            'lane_counts': {k: v['total'] for k, v in dir_counts.items()},
            'lane_densities': dir_densities,
            'density': density
        }],
    }


def process_video(input_path, output_path, *, conf_global=0.6, motorcycle_conf=0.75, iou_thresh=0.3):
    """Process a video with YOLOv8 vehicle detection into 4 quadrants."""
    try:
        from ultralytics import YOLO

        model_path = getattr(settings, 'YOLO_MODEL_PATH', 'yolov8n.pt')
        model = YOLO(model_path)

        class_map = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError("Cannot open video file")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30
        
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 1. Scale down output resolution to speed up VP8 encoding dramatically
        MAX_WIDTH = 800
        if orig_width > MAX_WIDTH:
            scale = MAX_WIDTH / orig_width
            width = MAX_WIDTH
            height = int(orig_height * scale)
        else:
            width = orig_width
            height = orig_height
            
        frame_area = width * height

        # 2. Reduce the output framerate to ~15 FPS to cut processing/encoding time in half or more
        TARGET_FPS = 15.0
        process_every_n_frames = max(1, int(round(orig_fps / TARGET_FPS)))
        out_fps = int(orig_fps / process_every_n_frames)

        if not output_path.endswith('.webm'):
            output_path = output_path.rsplit('.', 1)[0] + '.webm'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'VP80')

        out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
        if not out.isOpened():
            raise IOError("Cannot create output video file")

        vehicle_colors = {
            'car': (66, 135, 245),
            'truck': (245, 66, 66),
            'bus': (66, 245, 78),
            'motorcycle': (245, 193, 66),
        }

        DIRECTIONS = ['North', 'South', 'East', 'West']
        lane_stats = {d: defaultdict(int) for d in DIRECTIONS}
        lane_frame_totals = {d: 0 for d in DIRECTIONS}
        lane_max_counts = {d: 0 for d in DIRECTIONS}

        tracks = {name: [] for name in vehicle_colors.keys()}
        next_track_id = {name: 1 for name in vehicle_colors.keys()}
        unique_counts = {name: 0 for name in vehicle_colors.keys()}

        max_total_concurrent = 0
        density_series = []
        time_series_data = []
        sample_every = max(1, out_fps // 2) # Sample density twice per second
        accepted_conf_sum = 0.0
        accepted_conf_count = 0

        MAX_VEHICLES_PER_DIR = 15
        NUM_DIRS = 4
        raw_frame_count = 0
        frame_count = 0
        frame_idx = 0
        TRACK_TIMEOUT = 10 # timeout based on actual processed frames (approx 2/3 sec at 15fps)
        CONFIRM_HITS = 2 # verify easier since fewer frames are processed

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            raw_frame_count += 1
            if raw_frame_count % process_every_n_frames != 0:
                continue

            # Resize early to save operations on drawing and scaling later
            if orig_width > MAX_WIDTH:
                frame = cv2.resize(frame, (width, height))

            for direction in lane_stats:
                lane_stats[direction].clear()
                lane_stats[direction]['total'] = 0

            # Draw diagonals (use scaled width/height)
            cv2.line(frame, (0, 0), (width, height), (255, 255, 255), 2)
            cv2.line(frame, (0, height), (width, 0), (255, 255, 255), 2)

            # Frame Skipping for Inference (Every Nth frame)
            FRAME_SKIP = 3 # With out_fps ~15, inference runs ~5 times per sec
            if frame_count % FRAME_SKIP == 0:
                # Run YOLO on the frame (which is max 800px wide). 
                # Use standard imgsz=640 and low conf so YOLO detects small vehicles.
                results = model(frame, imgsz=640, conf=0.15, verbose=False)
            else:
                # Reuse last results for skipped frames
                pass # results remains what it was on the last processed frame

            total_this_frame = 0

            if results and len(results) > 0:
                boxes = results[0].boxes
                detections_by_class = {name: [] for name in vehicle_colors.keys()}

                for box in boxes:
                    # Boxes are already in 'frame' coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if cls in class_map:
                        vehicle_type = class_map[cls]
                        
                        # Override high thresholds for aerial/drone footage 
                        # where vehicles are very small
                        conf_threshold = min(conf_global, 0.3)
                        if vehicle_type == 'motorcycle':
                            conf_threshold = min(motorcycle_conf, 0.35)
                            
                        if conf < conf_threshold:
                            continue

                        box_area = max(1, (x2 - x1) * (y2 - y1))
                        if vehicle_type == 'motorcycle' and box_area > 0.05 * frame_area:
                            continue

                        detections_by_class[vehicle_type].append(((x1, y1, x2, y2), conf))

                for vtype, dets in detections_by_class.items():
                    fresh_tracks = []
                    for t in tracks[vtype]:
                        if frame_idx - t['last_seen'] <= TRACK_TIMEOUT:
                            fresh_tracks.append(t)
                    tracks[vtype] = fresh_tracks

                    used_det = set()
                    for ti, t in enumerate(tracks[vtype]):
                        best_j = -1
                        best_iou = 0.0
                        for j, (dbox, dconf) in enumerate(dets):
                            if j in used_det:
                                continue
                            iou = _compute_iou(t['bbox'], dbox)
                            if iou > best_iou:
                                best_iou = iou
                                best_j = j
                        if best_j != -1 and best_iou >= iou_thresh:
                            tracks[vtype][ti]['bbox'] = dets[best_j][0]
                            tracks[vtype][ti]['last_seen'] = frame_idx
                            tracks[vtype][ti]['hits'] += 1
                            if not tracks[vtype][ti]['counted'] and tracks[vtype][ti]['hits'] >= CONFIRM_HITS:
                                unique_counts[vtype] += 1
                                tracks[vtype][ti]['counted'] = True
                            used_det.add(best_j)

                    for j, (dbox, dconf) in enumerate(dets):
                        if j in used_det:
                            continue
                        tracks[vtype].append({
                            'id': next_track_id[vtype],
                            'bbox': dbox,
                            'last_seen': frame_idx,
                            'hits': 1,
                            'counted': False,
                        })
                        next_track_id[vtype] += 1

                    total_this_frame += len(dets)
                    for _, dconf in dets:
                        accepted_conf_sum += dconf
                        accepted_conf_count += 1

                for vtype, tlist in tracks.items():
                    color = vehicle_colors[vtype]
                    for t in tlist:
                        x1, y1, x2, y2 = t['bbox']
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        dir_name = get_direction(cx, cy, width, height)
                        lane_stats[dir_name][vtype] += 1
                        lane_stats[dir_name]['total'] += 1

                        roi = frame[y1:y2, x1:x2]
                        if roi.shape[0] > 0 and roi.shape[1] > 0:
                            overlay_roi = roi.copy()
                            cv2.rectangle(overlay_roi, (0, 0), (x2-x1, y2-y1), color, -1)
                            cv2.addWeighted(overlay_roi, 0.2, roi, 0.8, 0, roi)
                            frame[y1:y2, x1:x2] = roi
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{vtype}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                      (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if total_this_frame > max_total_concurrent:
                max_total_concurrent = total_this_frame

            for dir_name in lane_stats:
                cnt = lane_stats[dir_name]['total']
                lane_frame_totals[dir_name] += cnt
                if cnt > lane_max_counts[dir_name]:
                    lane_max_counts[dir_name] = cnt

            if frame_idx % sample_every == 0:
                capacity = MAX_VEHICLES_PER_DIR * NUM_DIRS
                density_pct = min(100.0, (total_this_frame / capacity) * 100.0)
                density_series.append(float(density_pct))
                
                timestamp = frame_count / out_fps if out_fps > 0 else 0
                current_lane_counts = {d: lane_stats[d]['total'] for d in DIRECTIONS}
                current_lane_densities = {d: min(100.0, (cnt / MAX_VEHICLES_PER_DIR) * 100.0) for d, cnt in current_lane_counts.items()}
                
                time_series_data.append({
                    'timestamp': float(timestamp),
                    'lane_counts': current_lane_counts,
                    'lane_densities': current_lane_densities,
                    'density': float(density_pct)
                })

            # Draw UI Panel
            for i, (dir_name, stats) in enumerate(lane_stats.items()):
                # Compact height to fit 4 blocks
                y_base = 20 + i * 85
                density_val = min(100, (stats['total'] / MAX_VEHICLES_PER_DIR) * 100)
                
                # Panel background
                cv2.rectangle(frame, (10, y_base - 15), (200, y_base + 65), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, y_base - 15), (200, y_base + 65), (255, 255, 255), 1)
                
                # Direction Name
                cv2.putText(frame, dir_name, (15, y_base),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Render only total counts instead of loop to save space
                cars = stats.get('car', 0)
                trucks = stats.get('truck', 0)
                motorcycles = stats.get('motorcycle', 0)
                buses = stats.get('bus', 0)
                
                cv2.putText(frame, f"C:{cars} T:{trucks} M:{motorcycles} B:{buses}", (15, y_base + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                
                bar_width = 160
                filled_width = int((bar_width * density_val) / 100)
                cv2.rectangle(frame, (15, y_base + 45), (15 + bar_width, y_base + 55), (100, 100, 100), 1)
                if density_val > 0:
                    if density_val > 75:
                        color = (0, 0, 255)
                    elif density_val > 50:
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.rectangle(frame, (15, y_base + 45), (15 + filled_width, y_base + 55), color, -1)
                cv2.putText(frame, f"{density_val:.1f}%", (15, y_base + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.putText(frame, f'Frame: {frame_count}', (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            out.write(frame)
            frame_count += 1
            frame_idx += 1

        cap.release()
        out.release()

        if frame_count == 0:
            raise ValueError("No frames were processed from the video")

        capacity = MAX_VEHICLES_PER_DIR * NUM_DIRS
        density_percentage = min(100.0, (max_total_concurrent / capacity) * 100.0)
        avg_conf = (accepted_conf_sum / accepted_conf_count) if accepted_conf_count else 0.0
        low_confidence = avg_conf < 0.55

        lane_avg_counts = {}
        lane_densities = {}
        for dir_name in lane_frame_totals:
            avg = (lane_frame_totals[dir_name] / frame_count) if frame_count > 0 else 0.0
            lane_avg_counts[dir_name] = float(avg)
            lane_densities[dir_name] = min(100.0, (avg / MAX_VEHICLES_PER_DIR) * 100.0)

        return {
            'total_vehicles': int(sum(unique_counts.values())),
            'density': float(density_percentage),
            'vehicle_counts': {
                'car': int(unique_counts['car']),
                'truck': int(unique_counts['truck']),
                'bus': int(unique_counts['bus']),
                'motorcycle': int(unique_counts['motorcycle']),
            },
            'lane_avg_counts': lane_avg_counts,
            'lane_max_counts': lane_max_counts,
            'lane_densities': lane_densities,
            'density_series': density_series,
            'time_series_data': time_series_data,
            'avg_confidence': float(avg_conf),
            'low_confidence': bool(low_confidence),
        }

    except Exception as e:
        import traceback
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        raise e
