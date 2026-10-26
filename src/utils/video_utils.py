"""
video_utils.py — Video utilities
==============================
Provides concat_episode_videos() to merge the per-episode videos
produced by gym.wrappers.RecordVideo into one video per run.
"""

import glob
import os


def concat_episode_videos(video_folder: str, output_name: str = "full_run.mp4",
                          delete_sources: bool = True) -> str | None:
    """
    Concatenate every .mp4 in `video_folder` into a single file.

    Args:
        video_folder:    Directory containing the individual .mp4 files.
        output_name:     Output file name (in the same directory).
        delete_sources:  If True, delete the individual files after concatenation.

    Returns:
        Path of the concatenated file, or None if no video was found.
    """
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError:
        print("[WARN] moviepy unavailable — skipping video concatenation.")
        return None

    # List and sort episodes in natural order
    clips_paths = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    out_path = os.path.join(video_folder, output_name)

    # Exclude the output if it already exists (rerun)
    clips_paths = [p for p in clips_paths if os.path.basename(p) != output_name]

    if not clips_paths:
        return None

    print(f"  → Concatenating {len(clips_paths)} videos into {out_path}...")
    try:
        clips = [VideoFileClip(p) for p in clips_paths]
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(out_path, logger=None, codec="libx264", audio=False)

        # Close clips cleanly
        for c in clips:
            c.close()
        final.close()

        # Delete the individual episodes
        if delete_sources:
            for p in clips_paths:
                os.remove(p)

        print(f"  ✓ Final video: {out_path}")
        return out_path

    except Exception as e:
        print(f"  [WARN] Concatenation failed: {e}")
        return None
