import os
import json
import logging
import numpy as np
import cv2


class CourtConfig:
    """球场配置基类，包含球场尺寸、角点、参考图片及投影方法。"""

    def __init__(
        self,
        court_width: float,
        court_height: float,
        court_top_height: float,
        court_margin: float,
        reference_image_path: str = None,
        image_court_left_pix: int = 0,
        image_court_right_pix: int = 0,
        image_court_top_pix: int = 0,
        image_court_bottom_pix: int = 0,
    ):
        self.court_width = court_width
        self.court_height = court_height
        self.court_top_height = court_top_height
        self.court_margin = court_margin

        # 球场地面角点 (带 margin)
        m = court_margin
        self.court_ground_corners = np.asarray(
            [
                [0 - m, 0 - m, 0],
                [0 - m, court_height + m, 0],
                [court_width + m, court_height + m, 0],
                [court_width + m, 0 + m, 0],
            ]
        )

        # 球场顶部角点
        h = court_top_height
        self.court_top_corners = np.asarray(
            [
                [0, 0, h],
                [0, court_height, h],
                [court_width, court_height, h],
                [court_width, 0, h],
            ]
        )

        # bbox 过滤范围
        self.court_range_x = [-m, court_width + m]
        self.court_range_y = [-m, court_height + m]

        # 参考图片
        self.reference_image_path = reference_image_path
        self._reference_image = None

        # 参考图片上球场区域的像素坐标
        self.image_court_left_pix = image_court_left_pix
        self.image_court_right_pix = image_court_right_pix
        self.image_court_top_pix = image_court_top_pix
        self.image_court_bottom_pix = image_court_bottom_pix

        # 像素/米 比率
        self.width_rate = (
            image_court_right_pix - image_court_left_pix + 1
        ) / court_width
        self.height_rate = (
            image_court_bottom_pix - image_court_top_pix + 1
        ) / court_height

    @property
    def reference_image(self) -> np.ndarray:
        """延迟加载参考图片。"""
        if self._reference_image is None and self.reference_image_path is not None:
            if os.path.exists(self.reference_image_path):
                self._reference_image = cv2.imread(self.reference_image_path)
            else:
                self._reference_image = np.zeros((580, 1024, 3), dtype=np.uint8)
        return self._reference_image

    def project_point_to_image(self, point: np.ndarray):
        """将球场坐标 (米) 投影到参考图片像素坐标。"""
        x = int(point[0] * self.width_rate + self.image_court_left_pix)
        y = int(self.image_court_bottom_pix - point[1] * self.height_rate)
        return x, y

    @classmethod
    def from_json(cls, json_path: str) -> "CourtConfig":
        """从 JSON 文件加载自定义球场配置。

        JSON 格式示例::

            {
                "court_width": 3.8,
                "court_length": 3.8,
                "court_top_height": 2.5,
                "court_margin": 0.5
            }
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # court_length 映射为 court_height
        court_height = data.get("court_length", data.get("court_height"))
        return cls(
            court_width=data["court_width"],
            court_height=court_height,
            court_top_height=data.get("court_top_height", 2.5),
            court_margin=data.get("court_margin", 0.5),
            reference_image_path=data.get("reference_image_path"),
            image_court_left_pix=data.get("image_court_left_pix", 0),
            image_court_right_pix=data.get("image_court_right_pix", 0),
            image_court_top_pix=data.get("image_court_top_pix", 0),
            image_court_bottom_pix=data.get("image_court_bottom_pix", 0),
        )


class VolleyballCourt(CourtConfig):
    """排球场配置: 28.1m × 15.1m"""

    def __init__(self):
        super().__init__(
            court_width=28.1,
            court_height=15.1,
            court_top_height=2.5,
            court_margin=0.5,
            reference_image_path="assets/tactical-board.png",
            image_court_left_pix=51,
            image_court_right_pix=972,
            image_court_top_pix=42,
            image_court_bottom_pix=537,
        )


class BadmintonCourt(CourtConfig):
    """羽毛球场配置: 13.4m × 6.1m"""

    def __init__(self):
        super().__init__(
            court_width=13.4,
            court_height=6.1,
            court_top_height=2.0,
            court_margin=0.5,
            reference_image_path="assets/badminton-board.png",
            image_court_left_pix=51,
            image_court_right_pix=972,
            image_court_top_pix=42,
            image_court_bottom_pix=537,
        )


# ---- 预设球场类型 ----
PRESET_COURTS = {
    "volleyball": VolleyballCourt,
    "badminton": BadmintonCourt,
}


def load_court(court_spec: str) -> CourtConfig:
    """根据预设名称或 JSON 文件路径加载球场配置。

    Args:
        court_spec: 预设名称 ("volleyball" / "badminton") 或 JSON 文件路径。
    """
    if court_spec in PRESET_COURTS:
        return PRESET_COURTS[court_spec]()
    if os.path.isfile(court_spec):
        return CourtConfig.from_json(court_spec)
    raise ValueError(
        f"未知的球场配置: {court_spec!r}。"
        f"可选预设: {list(PRESET_COURTS.keys())}，或传入 JSON 文件路径。"
    )


# ---- 默认球场实例 & 向后兼容别名 ----
default_court: CourtConfig = VolleyballCourt()


def set_default_court(court: CourtConfig) -> None:
    """设置全局默认球场，更新模块级别向后兼容别名。"""
    global default_court, court_width, court_height
    global court_ground_corners, court_top_corners, court_reference_image_raw
    default_court = court
    court_width = court.court_width
    court_height = court.court_height
    court_ground_corners = court.court_ground_corners
    court_top_corners = court.court_top_corners
    court_reference_image_raw = court.reference_image
    logging.getLogger(__name__).info(
        f"球场配置已更新: {court.court_width}m × {court.court_height}m"
    )


court_width = default_court.court_width
court_height = default_court.court_height
court_ground_corners = default_court.court_ground_corners
court_top_corners = default_court.court_top_corners
court_reference_image_raw = default_court.reference_image


def project_point_to_image(point: np.ndarray):
    return default_court.project_point_to_image(point)
