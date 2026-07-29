# Week 02 Summary

## 2026-07-28

实验目标：开始推进 ROS2 + Gazebo 苹果视觉识别与机械臂采摘任务。

### 今日完成

1. 阅读并分析新一周任务要求。
2. 检查现有仪表识别项目和 ROS2 功能包结构。
3. 确认 ROS2 Humble、rclpy、sensor_msgs、cv_bridge 和 RViz2 已安装。
4. 确认当前 ROS2 功能包仍是基础骨架，还没有图像处理节点。
5. 确定先完成 ROS2 图像订阅与发布，再进行苹果颜色识别。
6. 决定暂时保留原仪表项目结构，不进行大规模重构。

### 当前问题

1. `package.xml` 中的 `sensor_msgs` 拼写错误。
2. `setup.py` 中还没有注册 ROS2 节点。
3. 当前没有安装或配置 Gazebo、MoveIt 2 和机械臂模型。
4. 苹果采摘任务内容较多，需要分阶段完成。

### 下一步计划

1. 修正 ROS2 功能包依赖。
2. 创建 `apple_detector_node`。
3. 订阅 `/camera/image_raw` 图像话题。
4. 使用 `cv_bridge` 将 ROS 图像转换为 OpenCV 图像。
5. 发布处理后的标注图像。
6. 使用 `image_tools` 验证图像处理流程。
7. 图像通信测试成功后，再加入红色和绿色苹果检测。