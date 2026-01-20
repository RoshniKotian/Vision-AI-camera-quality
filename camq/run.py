from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from camq.io.video_reader import iter_video_frames


def find_videos(input_path: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    if input_path.is_file():
        return [input_path]

    return [p for p in input_path.rglob("*") if p.suffix.lower() in exts]


def process_video(video_path: Path, frame_step: int, max_frames: int | None) -> dict:
    meta, frames = iter_video_frames(
        video_path, frame_step=frame_step, max_frames=max_frames
    )

    sampled_frames = 0
    for _idx, _frame in frames:
        sampled_frames += 1

    return {
        "video": meta.path.name,
        "fps": meta.fps,
        "resolution": f"{meta.width}x{meta.height}",
        "duration_s": round(meta.duration_s, 2),
        "sampled_frames": sampled_frames,
        "frame_step": frame_step,
    }


def main():
    parser = argparse.ArgumentParser(description="Vision AI Camera Quality Lab - Day 1")
    parser.add_argument("--input", required=True, help="Video file or folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--frame_step", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    max_frames = None if args.max_frames == -1 else args.max_frames

    videos = find_videos(input_path)
    if not videos:
        raise SystemExit("No videos found.")

    rows = []
    for v in videos:
        print(f"[INFO] Processing {v.name}")
        rows.append(process_video(v, args.frame_step, max_frames))

    df = pd.DataFrame(rows)
    csv_path = output_path / "metrics.csv"
    df.to_csv(csv_path, index=False)

    print(f"[DONE] Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
