"""
v4l2-ctl --list-devices

60fps 4k
ffplay -f v4l2 -framerate 60 -video_size 3840x2160 -vf "vflip" -i /dev/video0
ffmpeg -y -f v4l2 -r 60 -video_size 3840x2160 -i /dev/video0 -vf "vflip" -c:v h264 out.mp4

30fps 2k
ffplay -f v4l2 -framerate 30 -video_size 2048x1080 -vf "vflip" -i /dev/video0
ffmpeg -y -f v4l2 -r 30 -video_size 2048x1080 -i /dev/video0 -vf "vflip" -c:v h264 out.mp4
"""
import subprocess
import argparse
from datetime import datetime
import sys
import os

# Argument setup
parser = argparse.ArgumentParser(description="Record video using ffmpeg")
parser.add_argument(
    "--output_filename",
    type=str,
    default=f"./data/raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
    help="Output filename for the video",
)
parser.add_argument(
    "--framerate", type=int, default=30, help="Framerate of the recording"
)
parser.add_argument(
    "--video_size",
    type=str,
    default="2048x1080",
    help="Video size (resolution) of the recording",
)

start_time = None


def record_ffmpeg(output_filename, framerate=30, video_size="2048x1080"):
    global start_time
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "v4l2",
        "-r", str(framerate),
        "-video_size", video_size,
        "-i", "/dev/video0",
        "-vf", "vflip",
        "-c:v", "h264",
        output_filename,
    ]

    process = None

    try:
        start_time = datetime.now()
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        process.communicate()
    except KeyboardInterrupt:
        if process:
            process.stdin.write(b'q')  # Send 'q' to ffmpeg's stdin
            process.communicate()
        end_time = datetime.now()
        duration = (end_time - start_time).seconds
        print("Recording safely terminated.")
        print(f"Output filename: {output_filename}")
        print(f"Length of recording: {duration} seconds")
        sys.exit(0)
    except subprocess.CalledProcessError:
        print("Error: ffmpeg process failed.")


if __name__ == "__main__":
    args = parser.parse_args()

    # Check if the output directory exists. If not, create it.
    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    record_ffmpeg(args.output_filename, args.framerate, args.video_size)