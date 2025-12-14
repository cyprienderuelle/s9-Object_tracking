import numpy as np
from scipy.optimize import linear_sum_assignment
from KalmanFilter import KalmanFilter


def calculate_iou(box1, box2):
    """
    Calcule l'IoU entre deux boîtes [x, y, w, h]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    if xi2 < xi1 or yi2 < yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)
    union_area = w1 * h1 + w2 * h2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


class KalmanTrack:
    track_count = 0

    def __init__(self, frame, bbox, conf):
        KalmanTrack.track_count += 1
        self.track_id = KalmanTrack.track_count

        self.bbox = bbox
        self.conf = conf
        self.missed_frames = 0
        self.active = True
        self.predicted_bbox = bbox

        # Kalman Filter
        dt = 0.1
        u_x = u_y = 1
        std_acc = 1
        x_sdt_meas = y_sdt_meas = 0.1
        self.kf = KalmanFilter(dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas)

        # Init state center
        x_c = bbox[0] + bbox[2] / 2
        y_c = bbox[1] + bbox[3] / 2
        self.kf.x = np.array([[x_c], [y_c], [0], [0]])

        self.history = [[frame, self.track_id] + list(bbox) + [conf, -1, -1, -1]]

    def predict_bbox(self):
        x_pred = self.kf.predict()
        x_c, y_c = x_pred[0][0], x_pred[1][0]
        w, h = self.bbox[2], self.bbox[3]

        self.predicted_bbox = [x_c - w/2, y_c - h/2, w, h]
        return self.predicted_bbox

    def update(self, frame, new_bbox, new_conf):
        x_c = new_bbox[0] + new_bbox[2] / 2
        y_c = new_bbox[1] + new_bbox[3] / 2
        z = np.array([[x_c], [y_c]])

        self.kf.update(z)

        self.bbox = new_bbox
        self.conf = new_conf
        self.missed_frames = 0

        self.history.append([frame, self.track_id] + list(new_bbox) + [new_conf, -1, -1, -1])

    def mark_missed(self, max_missed):
        self.missed_frames += 1
        self.bbox = self.predicted_bbox
        if self.missed_frames > max_missed:
            self.active = False


class KalmanGuidedIoUTracker:
    def __init__(self, iou_threshold=0.5, max_missed_frames=30):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

    def process_frame(self, frame_num, detections):
        active_tracks = [t for t in self.tracks if t.active]
        tracked_pred = [t.predict_bbox() for t in active_tracks]

        new_boxes = [d[:4] for d in detections]
        new_confs = [d[4] for d in detections]

        N, M = len(tracked_pred), len(new_boxes)
        cost = np.ones((N, M))

        for i, t_box in enumerate(tracked_pred):
            for j, d_box in enumerate(new_boxes):
                iou = calculate_iou(t_box, d_box)
                if iou >= self.iou_threshold:
                    cost[i, j] = 1 - iou

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_tracks, matched_dets = set(), set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] <= 1 - self.iou_threshold:
                active_tracks[i].update(frame_num, new_boxes[j], new_confs[j])
                matched_tracks.add(i)
                matched_dets.add(j)

        for i, t in enumerate(active_tracks):
            if i not in matched_tracks:
                t.mark_missed(self.max_missed_frames)

        for j, (bbox, conf) in enumerate(zip(new_boxes, new_confs)):
            if j not in matched_dets:
                self.tracks.append(KalmanTrack(frame_num, bbox, conf))

        return [t for t in self.tracks if t.active]

    def get_all_track_history(self):
        all_data = []
        for t in self.tracks:
            for record in t.history:
                rec = list(record)
                rec[6] = 1
                all_data.append(rec)
        return all_data
