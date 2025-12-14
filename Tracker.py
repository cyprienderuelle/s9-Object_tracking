import numpy as np
from scipy.optimize import linear_sum_assignment


def calculate_iou(box1, box2):
    """
    Calcule l'IoU entre deux boîtes [x, y, w, h].
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


class Track:
    track_count = 0

    def __init__(self, frame, bbox, conf):
        Track.track_count += 1
        self.track_id = Track.track_count

        self.bbox = bbox
        self.conf = conf
        self.missed_frames = 0
        self.active = True

        self.history = [[frame, self.track_id] + list(bbox) + [conf, -1, -1, -1]]

    def update(self, frame, bbox, conf):
        self.bbox = bbox
        self.conf = conf
        self.missed_frames = 0
        self.history.append([frame, self.track_id] + list(bbox) + [conf, -1, -1, -1])

    def mark_missed(self, max_missed):
        self.missed_frames += 1
        if self.missed_frames > max_missed:
            self.active = False

    def get_latest_data(self):
        data = self.history[-1]
        data[6] = 1
        return data


class IoUTracker:
    def __init__(self, iou_threshold=0.5, max_missed_frames=30):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

    def process_frame(self, frame_num, detections):
        boxes = [d[:4] for d in detections]
        confs = [d[4] for d in detections]

        active_tracks = [t for t in self.tracks if t.active]
        tracked_boxes = [t.bbox for t in active_tracks]

        N, M = len(tracked_boxes), len(boxes)
        cost = np.ones((N, M))

        for i, t_box in enumerate(tracked_boxes):
            for j, d_box in enumerate(boxes):
                iou = calculate_iou(t_box, d_box)
                if iou >= self.iou_threshold:
                    cost[i, j] = 1 - iou

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] <= 1 - self.iou_threshold:
                active_tracks[i].update(frame_num, boxes[j], confs[j])
                matched_tracks.add(i)
                matched_dets.add(j)

        for i, track in enumerate(active_tracks):
            if i not in matched_tracks:
                track.mark_missed(self.max_missed_frames)

        for j, (bbox, conf) in enumerate(zip(boxes, confs)):
            if j not in matched_dets:
                self.tracks.append(Track(frame_num, bbox, conf))

        return [t for t in self.tracks if t.active]

    def get_all_track_history(self):
        all_data = []
        for t in self.tracks:
            for record in t.history:
                rec = list(record)
                rec[6] = 1
                all_data.append(rec)
        return all_data
