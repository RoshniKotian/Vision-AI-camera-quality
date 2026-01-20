from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import cv2


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float


def iter_video_frames(
    video_path: Path,
    frame_step: int = 3,
    max_frames: int | None = 300,
) -> Tuple[VideoMeta, Iterator[tuple[int, any]]]:
    """
    Open a video file and yield sampled frames.

    frame_step: sample every Nth frame
    max_frames: stop after N sampled frames (None = no limit)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = (frame_count / fps) if fps > 0 else 0.0

    meta = VideoMeta(
        path=video_path,
        fps=float(fps),
        frame_count=frame_count,
        width=width,
        height=height,
        duration_s=float(duration_s),
    )

    def frame_generator():
        sampled = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % frame_step == 0:
                yield idx, frame
                sampled += 1
                if max_frames is not None and sampled >= max_frames:
                    break

            idx += 1

        cap.release()

    return meta, frame_generator()
