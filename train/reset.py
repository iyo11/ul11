from ultralytics import settings

# 查看当前所有路径配置
print(settings)

# 如果发现 runs_dir 不对，可以手动重置（或者重置为默认值）
settings.reset()