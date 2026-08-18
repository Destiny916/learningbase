import os
import sys

import time
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_error,
    now_sec,
    TimedFrame,
)
import threading
from dataclasses import dataclass
import numpy as np

from copy import deepcopy
import kfc
import cv2


class _kingfisher:
    def __init__(self):
        # Kingfisher相机
        self.kingfisher_initialized = False
        self.kfc_camera = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None

    def init_kingfisher(self, AutoExposure=True, Exposure=10000):
        """初始化Kingfisher相机-KFC版本"""
        try:

            log_info("Initializing Kingfisher camera...")

            import cv2, time

            self.kfc_camera = kfc.Camera()
            if not self.kfc_camera.connect("99"):
                log_info("Connect KFC success! ")
            else:
                log_error("cannot open /dev/video99 (or id 99).")
                return

            height, width = self.kfc_camera.getResolution()
            log_info(f"kfc capture Resolution is {height}*{width}")

            if AutoExposure:
                ret = self.kfc_camera.setAutoExposure(True)
                if ret != 0:
                    log_info("set autoexposure failed")
            else:
                ret = self.kfc_camera.setExposure(Exposure)
                if ret != 0:
                    log_info("set exposure failed")

            self.kingfisher_initialized = True
            log_info("Kingfisher camera initialized successfully")
            return True

        except Exception as e:
            log_error(f"Failed to initialize Kingfisher camera: {e}")
            return False

    def get_kingfisher_images(self):
        """获取Kingfisher相机图像"""
        if not self.kingfisher_initialized:
            return None, None

        try:

            left_image, right_image = self.kfc_camera.capture()

            t = now_sec()
            if left_image is None or np.max(left_image) == 0:
                time.sleep(0.005)
                log_error(f"Error capturing Kingfisher images: {e}")

            if left_image is not None and right_image is not None:
                return left_image, right_image, t
            return None, None
        except Exception as e:
            log_error(f"Error capturing Kingfisher images: {e}")
            return None, None

    def disconnect_kfc(self):
        if not self.kfc_camera:
            self.kfc_camera.disConnect()
            log_info("kfc disconnected!")
