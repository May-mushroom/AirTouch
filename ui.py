import cv2 as cv

FONT = cv.FONT_HERSHEY_SIMPLEX
ACCENT = (88, 205, 54)
JOINT_COLOR = (255, 255, 255)
BOX_PAD = 20

# Standard 21-point MediaPipe hand skeleton (wrist=0, thumb=1-4, index=5-8,
# middle=9-12, ring=13-16, pinky=17-20).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def draw_hands(frame, detection_result, gestures=None):
    """Draw skeleton + a bounding box with the gesture label for every detected hand.

    `frame` is expected to be the flipped/mirrored display frame, while
    `detection_result`'s landmarks are normalized against the unflipped frame
    that was fed to the hand tracker — so x is mirrored here to line back up.

    `gestures`, if given, must be aligned with `detection_result.hand_landmarks`
    (one entry per hand, in the same order): each entry is either `None` or a
    `(gesture_name, score)` pair, so each hand's box carries its own label —
    two hands making two different gestures stay independently visible.
    """
    if not detection_result or not detection_result.hand_landmarks:
        return frame

    h, w = frame.shape[:2]
    for i, (hand_landmarks, handedness) in enumerate(
        zip(detection_result.hand_landmarks, detection_result.handedness)
    ):
        points = [(w - int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for start, end in HAND_CONNECTIONS:
            cv.line(frame, points[start], points[end], ACCENT, 2, cv.LINE_AA)
        for point in points:
            cv.circle(frame, point, 4, JOINT_COLOR, -1, cv.LINE_AA)
            cv.circle(frame, point, 4, ACCENT, 1, cv.LINE_AA)

        x_min = max(min(p[0] for p in points) - BOX_PAD, 0)
        y_min = max(min(p[1] for p in points) - BOX_PAD, 0)
        x_max = min(max(p[0] for p in points) + BOX_PAD, w)
        y_max = min(max(p[1] for p in points) + BOX_PAD, h)
        cv.rectangle(frame, (x_min, y_min), (x_max, y_max), ACCENT, 2, cv.LINE_AA)

        gesture = gestures[i] if gestures and i < len(gestures) else None
        # gesture_name, score = gesture if gesture else (None, None)
        gesture_name = gesture if gesture else None
        score = None
        _draw_hand_label(frame, x_min, y_min, x_max, handedness[0].category_name, gesture_name, score, w)

    return frame


def _draw_hand_label(frame, x_min, y_min, x_max, hand_label, gesture_name, score, frame_w):
    text = f"{hand_label}: {gesture_name}" if gesture_name else f"{hand_label}: ..."
    if score is not None:
        text += f" ({score * 100:.0f}%)"

    (text_w, text_h), _ = cv.getTextSize(text, FONT, 0.6, 2)
    pad = 6

    # Place the label just above the box, or just inside its top edge if the
    # box is near the top of the frame, so it never gets clipped off-screen.
    label_baseline = y_min - 10 if y_min - 10 - text_h - pad > 0 else y_min + text_h + pad + 4
    bg_x2 = min(x_min + text_w + pad * 2, frame_w)
    cv.rectangle(
        frame,
        (x_min, label_baseline - text_h - pad),
        (max(bg_x2, x_min + 1), label_baseline + pad // 2),
        (30, 30, 30),
        -1,
    )
    cv.putText(frame, text, (x_min + pad, label_baseline), FONT, 0.6, ACCENT, 2, cv.LINE_AA)


def draw_fps(frame, fps):
    """Draw a small, unobtrusive FPS readout in the top-right corner."""
    if fps is None:
        return frame

    h, w = frame.shape[:2]
    text = f"{fps:.0f} FPS"
    text_w, _ = cv.getTextSize(text, FONT, 0.6, 1)[0]
    cv.putText(frame, text, (w - text_w - 20, 30), FONT, 0.6, (160, 160, 160), 1, cv.LINE_AA)
    return frame
