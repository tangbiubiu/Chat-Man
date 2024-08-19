# 

<div align=center>
  <img src="./imgs/chat-man.png" >
  <h1>🤖Chat Man🤖</h1>
</div>

![GitHub stars](https://img.shields.io/github/stars/tangbiubiu/Chat-Man?style=social)![GitHub last commit](https://img.shields.io/github/last-commit/tangbiubiu/Chat-Man)![License](https://img.shields.io/badge/License-MIT-red.svg "Author")

“技术平权“是人工智能时代的主题，AI让每一个普通人都有创造无限价值的潜力。随着AI大模型走向开源时代，一方面我们享受着巨额资源创造的基座模型变得容易；但另一方面，**面对层出不穷各有特色的模型，高效使用它们逐渐变成了一项挑战**，尤其是对于初学者来说。项目的动机在于：**无需考虑大模型之间的差异，尽管去调用它吧！（理论上支持Liunx/Windows双平台）。**主要特性包括：

1. 一键下载基座模型（针对国内环境，保证吃满带宽）；
2. 基于langchain的模型封装
3. 检索增强生成（RAG）框架
4. 一键微调

> **Sometimes you gotta run before you can walk.**     ——托尼 史塔克

## 支持模型列表💖

1. **MiniCPM**
   * **model_id:** OpenBMB/MiniCPM-2B-sft-fp32
   * **revision:** master
   * **所需空间：**10GB
2. **BaiChuan**
   * **model_id:** baichuan-inc/Baichuan2-7B-Chat
   * **revision:** v1.0.4
   * **所需空间：**
3. **GLM4**
   * **model_id:** ZhipuAI/glm-4-9b-chat
   * **revision:** master
   * **所需空间：**
4. **qwen2**
   * **model_id:** qwen/Qwen2-7B-Instruct
   * **revision:** master
   * **所需空间：**
5. **LLaMA3**
   * **model_id:** LLM-Research/Meta-Llama-3-8B-Instruct
   * **revision:** master
   * **所需空间：**

## 模型下载**😊**

可一键下载多个模型。

```shell
python models/model_downloader.py XXX1 XXX2 XXX3
```

下载`MiniCPM`和`GLM4`为例：

```shell
python models/model_downloader.py MiniCPM GLM4
```

## LangChain封装✨

TODO

## 联系我们📣

yu.tang333@qq.com
