# -*- coding = utf-8 -*-
# @Time : 2023/5/30 14:33
# @Author : Xinye Yang
# @File : show_info.py
# @Software : PyCharm


import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
for s in device.sensors:
    print(s.get_info(rs.camera_info.name))

cfg = pipeline.start(config)
device1 = cfg.get_device()
for s in device1.sensors:
    print(s.get_info(rs.camera_info.name))
