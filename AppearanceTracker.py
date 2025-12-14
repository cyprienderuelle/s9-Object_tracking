import numpy as np
import cv2
import onnxruntime as rt
from scipy.optimize import linear_sum_assignment
from Trackertp3 import KalmanTrack, KalmanGuidedIoUTracker


def iou(box1, box2):
    """Calcule l'IoU entre deux bounding boxes (x, y, w, h)"""

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    if xb <= xa or yb <= ya:
        return 0.0

    inter = (xb - xa) * (yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2

    return inter / (area1 + area2 - inter)


class ReIDModel:
    """Modèle ReID simple avec ONNX"""

    def __init__(self, model_path):
        self.w = 64
        self.h = 128

        self.mean = np.array([0.485, 0.456, 0.406]) * 255
        self.std = np.array([0.229, 0.224, 0.225]) * 255

        self.session = rt.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        img = cv2.resize(img, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = np.moveaxis(img, -1, 0)
        return img

    def extract(self, patches):
        if len(patches) == 0:
            return np.array([])

        batch = np.stack([self.preprocess(p) for p in patches]).astype(np.float32)
        feats = self.session.run(None, {self.input_name: batch})[0]
        return feats.astype(np.float32)

    def similarity(self, f1, f2):
        d = np.linalg.norm(f1 - f2)
        return 1.0 / (1.0 + d)


class AppearanceTrack(KalmanTrack):
    """Track avec feature ReID"""

    def __init__(self, frame_id, bbox, conf, feat):
        super().__init__(frame_id, bbox, conf)
        self.feat = feat

    def update(self, frame_id, bbox, conf, feat):
        super().update(frame_id, bbox, conf)
        self.feat = feat


class AppearanceTracker(KalmanGuidedIoUTracker):
    """Tracker IoU + ReID"""

    def __init__(self, reid_model_path,
                 iou_th=0.3, max_missed=10,
                 alpha=0.8, beta=0.2):

        super().__init__(iou_th, max_missed)

        self.alpha = alpha
        self.beta = beta
        self.reid = ReIDModel(reid_model_path)

    def process_frame(self, frame_id, frame, detections):

        boxes = [d[:4] for d in detections]
        confs = [d[4] for d in detections]

        patches = []
        valid = []

        # extraction des crops
        for i, b in enumerate(boxes):
            x, y, w, h = map(int, b)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            if x2 > x1 and y2 > y1:
                patches.append(frame[y1:y2, x1:x2])
                valid.append(detections[i])

        feats = self.reid.extract(patches)

        boxes = [d[:4] for d in valid]
        confs = [d[4] for d in valid]

        tracks = [t for t in self.tracks if t.active]
        preds = [t.predict_bbox() for t in tracks]

        N, M = len(tracks), len(boxes)
        matched_t = set()
        matched_d = set()

        if N > 0 and M > 0:
            cost = np.ones((N, M))

            for i in range(N):
                for j in range(M):
                    s_iou = iou(preds[i], boxes[j])
                    s_reid = self.reid.similarity(tracks[i].feat, feats[j])
                    s = self.alpha * s_iou + self.beta * s_reid

                    if s > self.iou_threshold:
                        cost[i, j] = 1.0 - s

            rows, cols = linear_sum_assignment(cost)

            for i, j in zip(rows, cols):
                if cost[i, j] <= 1.0 - self.iou_threshold:
                    tracks[i].update(
                        frame_id,
                        boxes[j],
                        confs[j],
                        feats[j]
                    )
                    matched_t.add(i)
                    matched_d.add(j)

        # tracks non associés
        for i, t in enumerate(tracks):
            if i not in matched_t:
                t.mark_missed(self.max_missed_frames)

        # nouvelles pistes
        for j in range(M):
            if j not in matched_d:
                self.tracks.append(
                    AppearanceTrack(
                        frame_id,
                        boxes[j],
                        confs[j],
                        feats[j]
                    )
                )

        return [t for t in self.tracks if t.active]
