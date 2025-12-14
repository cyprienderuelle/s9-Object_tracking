import cv2
import os
import numpy as np
from glob import glob
from AppearanceTracker import AppearanceTracker

DATA_ROOT = "./ADL-Rundle-6/ADL-Rundle-6/"
IMG_DIR = os.path.join(DATA_ROOT, "img1/")
DET_FILE = os.path.join(DATA_ROOT, "det/yolov5l/det.txt")

OUT_DIR = "./tracking_results/"
OUT_VIDEO = os.path.join(OUT_DIR, "tp4.mp4")
OUT_TXT = os.path.join(OUT_DIR, "tp4.txt")


def load_detections(det_path):
    """
    Charge les détections depuis le fichier det.txt
    """
    detections = {}

    try:
        data = np.loadtxt(det_path)
    except IOError:
        print("Erreur : fichier de détections introuvable")
        return {}

    for row in data:
        frame_id = int(row[0])
        bbox = row[2:7]  # x, y, w, h, conf

        if frame_id not in detections:
            detections[frame_id] = []

        detections[frame_id].append(bbox)

    return detections


def save_results(tracks, out_path):
    """
    Sauvegarde des résultats au format MOT
    """
    tracks.sort(key=lambda x: x[0])

    with open(out_path, "w") as f:
        for t in tracks:
            line = (
                f"{int(t[0])},{int(t[1])},"
                f"{t[2]:.2f},{t[3]:.2f},{t[4]:.2f},{t[5]:.2f},"
                f"{t[6]:.2f},{t[7]:.2f},{t[8]:.2f},{t[9]:.2f}\n"
            )
            f.write(line)


def run_mot_tracking():
    os.makedirs(OUT_DIR, exist_ok=True)

    detections = load_detections(DET_FILE)
    if len(detections) == 0:
        return

    image_files = sorted(glob(os.path.join(IMG_DIR, "*.jpg")))
    total_frames = len(image_files)

    # tracker
    reid_model = "reid_osnet_x025_market1501.onnx"
    tracker = AppearanceTracker(
        reid_model_path=reid_model,
        iou_th=0.3,
        max_missed=10,
        alpha=0.7,
        beta=0.3
    )

    # initialisation vidéo
    first_frame = cv2.imread(image_files[0])
    h, w, _ = first_frame.shape

    writer = cv2.VideoWriter(
        OUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (w, h)
    )

    for i, img_path in enumerate(image_files):
        frame_id = i + 1
        frame = cv2.imread(img_path)

        current_det = detections.get(frame_id, [])
        active_tracks = tracker.process_frame(frame_id, frame, current_det)

        # affichage des tracks
        for t in active_tracks:
            x, y, w, h = map(int, t.bbox)
            tid = t.track_id

            color = (tid * 60 % 255, tid * 100 % 255, tid * 140 % 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID {tid}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        cv2.putText(
            frame,
            f"Frame {frame_id}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)
        cv2.imshow("Tracking TP4", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cv2.destroyAllWindows()

    # sauvegarde des résultats
    all_tracks = tracker.get_all_track_history()
    save_results(all_tracks, OUT_TXT)


if __name__ == "__main__":
    run_mot_tracking()
