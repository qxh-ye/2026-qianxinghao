"""管理检测目标的短期有效深度历史"""
import math

class DepthHistory:
    """按成熟度和像素位置管理短期深度记录"""

    def __init__(
            self,
            max_age_frames=15,
            max_match_distance_px=20.0,
    ):
        try:
            max_age_frames = int(max_age_frames)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "max_age_frames 必须是整数"
            ) from error

        try:
            max_match_distance_px = float(max_match_distance_px)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "max_match_distance_px 必须是浮点数"
            ) from error

        if max_age_frames < 1:
            raise ValueError(
                "max_age_frames 必须大于等于 1"
            )

        if (not math.isfinite(max_match_distance_px)
            or max_match_distance_px <= 0.0
        ):
            raise ValueError(
                "max_match_distance_px 必须是有限正数"
            )

        self.max_age_frames = max_age_frames
        self.max_match_distance_px = max_match_distance_px

        self._records = []

    @staticmethod
    def _normalize_target(
        maturity,
        center,
        frame_index,
    ):
        if not isinstance(maturity, str) or not maturity:
            raise ValueError(
                "maturity 必须是非空字符串"
            )

        if center is None or len(center) != 2:
            raise ValueError(
                "center 必须包含两个坐标"
            )

        try:
            center_x = float(center[0])
            center_y = float(center[1])
            frame_index = int(frame_index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "中心坐标和帧编号必须是数值"
            ) from error

        if not math.isfinite(center_x):
            raise ValueError(
                "center_x 必须是有限数值"
            )

        if not math.isfinite(center_y):
            raise ValueError(
                "center_y 必须是有限数值"
            )

        if frame_index < 0:
            raise ValueError(
                "frame_index 不能小于 0"
            )

        return (
            maturity,
            center_x,
            center_y,
            frame_index,
        )

    def _remove_expired_records(self, frame_index):
        """删除超过最大帧龄的历史记录"""
        self._records = [
            record
            for record in self._records
            if (
                0
                <= frame_index - record["frame_index"]
                <= self.max_age_frames
            )
        ]

    def _find_nearest_record_index(
        self,
        maturity,
        center_x,
        center_y,
    ):
        """寻找成熟度相同且像素距离最近的记录。"""
        best_index = None
        best_distance_squared = None

        maximum_distance_squared = (
            self.max_match_distance_px ** 2
        )

        for index, record in enumerate(
                self._records,
        ):
            if record["maturity"] != maturity:
                continue

            difference_x = (
                center_x - record["center_x"]
            )
            difference_y = (
                center_y - record["center_y"]
            )

            distance_squared = (
                difference_x ** 2
                + difference_y ** 2
            )

            if distance_squared > maximum_distance_squared:
                continue

            if (
                best_distance_squared is None
                or distance_squared
                < best_distance_squared
            ):
                best_distance_squared = distance_squared
                best_index = index

        return best_index

    def update(
            self, 
            maturity,
            center,
            depth_m,
            frame_index,
    ):
        """使用当前有效测量新增或更新历史记录"""
        (
            maturity,
            center_x,
            center_y,
            frame_index,
        ) = self._normalize_target(
            maturity,
            center,
            frame_index
        )

        try:
            depth_m = float(depth_m)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "depth_m 必须是数值"
            ) from error

        if not math.isfinite(depth_m) or depth_m <= 0.0:
            raise ValueError(
                "depth_m 必须是有限正数"
            )

        self._remove_expired_records(frame_index)

        record_index = (
            self._find_nearest_record_index(
                maturity,
                center_x,
                center_y,
            )
        )

        new_record = {
            "maturity": maturity,
            "center_x": center_x,
            "center_y": center_y,
            "depth_m": depth_m,
            "frame_index": frame_index,
        }

        if record_index is None:
            self._records.append(
                new_record
            )
        else:
            self._records[record_index] = (
                new_record
            )

    def get(
        self,
        maturity,
        center,
        frame_index,
    ):
        """返回匹配目标的近期有效深度"""
        (
            maturity,
            center_x,
            center_y,
            frame_index,
        ) = self._normalize_target(
            maturity,
            center,
            frame_index,
        )

        self._remove_expired_records(frame_index)

        record_index = (
            self._find_nearest_record_index(
                maturity,
                center_x,
                center_y,
            )
        )

        if record_index is None:
            return None

        return float(self._records[record_index]["depth_m"])







