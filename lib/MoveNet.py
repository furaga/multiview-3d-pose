import sys
import cv2 as cv
import numpy as np
import onnxruntime


class MoveNet:
    def __init__(self, model_select=1):
        if model_select == 0:
            self.model_path = "onnx/movenet_singlepose_lightning_4.onnx"
            self.input_size = 192
        elif model_select == 1:
            self.model_path = "onnx/movenet_singlepose_thunder_4.onnx"
            self.input_size = 256
        else:
            sys.exit(f"invalid model_select {model_select}")

        self.onnx_session = onnxruntime.InferenceSession(self.model_path)

    def run_inference(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        # 前処理
        input_image = cv.resize(image, dsize=(self.input_size, self.input_size))  # リサイズ
        input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換
        input_image = input_image.reshape(
            -1, self.input_size, self.input_size, 3
        )  # リシェイプ
        input_image = input_image.astype("int32")  # int32へキャスト

        # 推論
        input_name = self.onnx_session.get_inputs()[0].name
        output_name = self.onnx_session.get_outputs()[0].name
        outputs = self.onnx_session.run([output_name], {input_name: input_image})

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
        self,
        image,
        elapsed_time,
        keypoint_score_th,
        keypoints,
        scores,
    ):
        debug_image = image.copy()

        # 0:鼻 1:左目 2:右目 3:左耳 4:右耳 5:左肩 6:右肩 7:左肘 8:右肘 # 9:左手首
        # 10:右手首 11:左股関節 12:右股関節 13:左ひざ 14:右ひざ 15:左足首 16:右足首
        # Line：鼻 → 左目

        skeleton = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (0, 5),
            (0, 6),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (11, 12),
            (5, 11),
            (11, 13),
            (13, 15),
            (6, 12),
            (12, 14),
            (14, 16),
        ]

        for index01, index02 in skeleton:
            if (
                scores[index01] > keypoint_score_th
                and scores[index02] > keypoint_score_th
            ):
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
