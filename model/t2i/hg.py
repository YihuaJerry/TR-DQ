import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download PixArt-alpha/PixArt-alpha/t5-v1_1-xxl --local-dir ./model/t2i/pre_models/pixart_alpha/t5-v1_1-xxl')
