import socketserver
import cv2
import numpy as np
import sys
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_select", type=int, default=0)
    parser.add_argument("--keypoint_score", type=float, default=0.4)
    args = parser.parse_args()
    return args


def run_inference(onnx_session, input_size, image):
    image_width, image_height = image.shape[1], image.shape[0]

    # 前処理
    input_image = cv.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = input_image.astype("int32")  # int32へキャスト

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    outputs = onnx_session.run([output_name], {input_name: input_image})

    keypoints_with_scores = outputs[0]
    keypoints_with_scores = np.squeeze(keypoints_with_scores)

    # キーポイント、スコア取り出し
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = int(image_width * keypoints_with_scores[index][1])
        keypoint_y = int(image_height * keypoints_with_scores[index][0])
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores


def draw_debug(
    image,
    elapsed_time,
    keypoint_score_th,
    keypoints,
    scores,
):
    debug_image = copy.deepcopy(image)

    # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
    # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
    # Line：鼻 → 左目
    index01, index02 = 0, 1
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 右目
    index01, index02 = 0, 2
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左目 → 左耳
    index01, index02 = 1, 3
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右目 → 右耳
    index01, index02 = 2, 4
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 左肩
    index01, index02 = 0, 5
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：鼻 → 右肩
    index01, index02 = 0, 6
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 右肩
    index01, index02 = 5, 6
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 左肘
    index01, index02 = 5, 7
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肘 → 左手首
    index01, index02 = 7, 9
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肩 → 右肘
    index01, index02 = 6, 8
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肘 → 右手首
    index01, index02 = 8, 10
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左股関節 → 右股関節
    index01, index02 = 11, 12
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左肩 → 左股関節
    index01, index02 = 5, 11
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左股関節 → 左ひざ
    index01, index02 = 11, 13
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：左ひざ → 左足首
    index01, index02 = 13, 15
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右肩 → 右股関節
    index01, index02 = 6, 12
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右股関節 → 右ひざ
    index01, index02 = 12, 14
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)
    # Line：右ひざ → 右足首
    index01, index02 = 14, 16
    if scores[index01] > keypoint_score_th and scores[index02] > keypoint_score_th:
        point01 = keypoints[index01]
        point02 = keypoints[index02]
        cv.line(debug_image, point01, point02, (255, 255, 255), 4)
        cv.line(debug_image, point01, point02, (0, 0, 0), 2)

    # Circle：各点
    for keypoint, score in zip(keypoints, scores):
        if score > keypoint_score_th:
            cv.circle(debug_image, keypoint, 6, (255, 255, 255), -1)
            cv.circle(debug_image, keypoint, 3, (0, 0, 0), -1)

    # 処理時間
    cv.putText(
        debug_image,
        "Elapsed Time : " + "{:.1f}".format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        debug_image,
        "Elapsed Time : " + "{:.1f}".format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv.LINE_AA,
    )

    return debug_image


def main():
    args = parse_args()

    # For webcam input:
    all_cam_ids = [0, 2, 3, 4]
    all_caps = [cv2.VideoCapture(c) for c in all_cam_ids]

    if args.model_select == 0:
        model_path = "onnx/movenet_singlepose_lightning_4.onnx"
        input_size = 192
    elif args.model_select == 1:
        model_path = "onnx/movenet_singlepose_thunder_4.onnx"
        input_size = 256
    else:
        sys.exit(0)

    onnx_session = onnxruntime.InferenceSession(model_path)

    finish = False
    prev = time.time()
    while not finish:
        all_kps = []
        all_imgs = []
        dt = time.time() - prev
        prev = time.time()
        for cam_id, cap in zip(all_cam_ids, all_caps):
            if not cap.isOpened():
                continue

            ret, img = cap.read()
            if not ret:
                continue

            keypoints, scores = run_inference(
                onnx_session,
                input_size,
                img,
            )

            img = draw_debug(img, dt, args.keypoint_score, keypoints, scores)
            all_imgs.append(img)

        show_img = cv2.vconcat(
            [
                cv2.hconcat(all_imgs[:2]),
                cv2.hconcat(all_imgs[2:]),
            ]
        )
        cv2.imshow(f"Cameras", show_img)
        if cv2.waitKey(1) == ord("q"):
            finish = True

    for cap in all_caps:
        cap.release()


if __name__ == "__main__":
    main()
