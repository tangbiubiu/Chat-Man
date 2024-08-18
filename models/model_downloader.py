from modelscope import snapshot_download, AutoModel, AutoTokenizer
import yaml
import argparse

# 读取配置文件
with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

model_list = ['MiniCPM', 'BaiChuan', 'GLM4', 'qwen2', 'LLaMA3']

# 参数设定
parser = argparse.ArgumentParser(description='模型下载参数')
parser.add_argument('model_name',  type=str, nargs='+',
                    choices=model_list,
                    help='想要下载的模型的名字。')
args = parser.parse_args()

# 根据模型名称获取模型的配置
def get_model_config(model_name, config):
    for model in config['models']:
        if model_name in model:
            return model[model_name]
    return None

# 下载模型
for model_name in args.model_name:
    model_config = get_model_config(model_name, cfg)
    if model_config:
        model_dir = snapshot_download(model_config["model_id"], 
                                      cache_dir='./Model_Lib', 
                                      revision=model_config["revision"])
        print(f'{model_name} 下载完成，存储在 {model_dir}')
    else:
        print(f'未找到模型配置: {model_name}')
