import os

base_path1 = "/root/autodl-tmp/data/train/rain100h/"
# base_path2 = "/root/autodl-tmp/WeatherDiffusion-main/scratch/data/snow100k/train/input/"
output_file = "/root/autodl-tmp/data/train/rain100h_train.txt"
image_exts = ('.jpg', '.png', '.jpeg', '.gif')  # 支持的格式
rain_paths = [f"{base_path1}rain-0{i}.png" for i in range(1, 100)]

with open(output_file, 'w') as f:
    f.write("\n".join(rain_paths))
    # for root, dirs, files in os.walk(base_path2):
    #     for file in files:
    #         if file.lower().endswith(image_exts):
    #             f.write(os.path.join(root, file) + "\n")  # 写入完整路径
    #             # 若只需文件名：f.write(file + "\n")