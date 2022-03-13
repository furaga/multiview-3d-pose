"""
python record.py --out_dir ../data/Recorded
"""

import argparse
import cv2
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()
    return args


def main(args):
    args.out_dir.mkdir(exist_ok=True, parents=True)

    all_cam_ids = [0, 2, 3, 4]
    all_caps = [cv2.VideoCapture(c) for c in all_cam_ids]

    finish = False

    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out_videos = {}

    while not finish:
        all_imgs = []
        for cam_id, cap in zip(all_cam_ids, all_caps):
            ret, img = cap.read()
            if not ret:
                continue
            all_imgs.append(img)

            if cam_id not in out_videos:
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_video_path = args.out_dir / f"camera_{cam_id}.mp4"
                out_videos[cam_id] = cv2.VideoWriter(
                    str(out_video_path), fourcc, fps, (width, height)
                )
                print("fps, width, height", fps, width, height)

            out_videos[cam_id].write(img)

        show_img = cv2.vconcat(
            [
                cv2.hconcat(all_imgs[:2]),
                cv2.hconcat(all_imgs[2:]),
            ]
        )
        cv2.imshow(f"Cameras", show_img)
        if cv2.waitKey(1) == ord("q"):
            finish = True
            
    for out in out_videos:
        out.release()

    for cap in all_caps:
        cap.release()



if __name__ == "__main__":
    main(parse_args())
