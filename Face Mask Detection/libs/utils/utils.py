import time

import cv2
import numpy as np


def compare_embedding(embedding, facebank, threshold=0.7):
    if len(facebank) == 0:
        return None
    embedding = np.array(embedding)
    facebank = np.array(facebank).squeeze(axis=1)
    diff = embedding - facebank
    dist = np.sum(np.square(diff), axis=1)
    min_idx = np.argmin(dist)
    if dist[min_idx] < threshold:
        return min_idx
    return None


def show_bboxes(img, bounding_boxes, facial_landmarks, names):
    """ Draw bounding boxes and facial landmarks. """
    bounding_boxes = np.asarray(bounding_boxes, dtype=int)
    for b, n in zip(bounding_boxes, names):
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 1)
        cv2.putText(img, n, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(img, (p[i], p[i + 5]), 2, color=(0, 255, 0), thickness=-1)
    return img

