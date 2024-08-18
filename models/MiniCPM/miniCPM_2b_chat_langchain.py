from langchain.llms.base import LLM
from typing import Any, List, Optional, AsyncGenerator, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch, sys, os
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from threading import Thread

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from utils import choose_device, torch_gc

class MiniCPM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None
    device:str = None

    def __init__(self, model_path :str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()

        self.device = choose_device()
        torch_gc(self.device)

        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16).to(self.device)
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = ['.', '。', '?', '？', '!', '！'], 
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:     
        last_message = messages[-1].content  # 获取最后一条消息
        inputs = self.tokenizer.encode(last_message, return_tensors="pt").to(self.device)  # 将消息内容编码为张量
        outputs = self.model.generate(inputs, max_length=self.n + len(inputs[0]), temperature=self.temperature)  # 调用模型生成文本张量
        tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  # 将张量解码为文本

        # 截取合适的长度（找到完整句子的结束位置）
        end_positions = [tokens.find(c) for c in stop if tokens.find(c) != -1]
        if end_positions:
            end_pos = max(end_positions) + 1  # 包括结束符
            if end_pos < self.n:
                tokens = tokens[:end_pos]
            else:
                tokens = tokens[:self.n]
        else:
            tokens = tokens[:self.n]

        message = AIMessage(content=tokens)  # 将生成的文本封装为消息
        generation = ChatGeneration(message=message)  # 封装为ChatGeneration
        
        return ChatResult(generations=[generation])


    def _call(self, prompt : str, stop: Optional[List[str]] = ['.', '。', '?', '？', '!', '！'],
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> str:
        responds, history = self.model.chat(self.tokenizer, prompt, temperature=0.5, top_p=0.8, repetition_penalty=1.02)
        return responds
    
    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, 
            run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        last_message = messages[-1].content
        inputs = self.tokenizer(last_message, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(tokenizer=self.tokenizer)  # transformers提供的标准接口，专为流式输出而生
        inputs.update({"streamer": streamer, "max_new_tokens": 512})  # 这是model.generate的参数，可以自由发挥
        thread = Thread(target=self.model.generate, kwargs=inputs)  # TextIteratorStreamer需要配合线程使用
        thread.start()
        for new_token in streamer:
        # 用迭代器的返回形式 。
            yield ChatGenerationChunk(message=AIMessage(content=new_token))
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None,
                     run_manager: Optional[CallbackManagerForLLMRun] = None,
                     **kwargs: Any) -> str:
        responds, history = await self.model.chat(self.tokenizer, prompt, temperature=0.5, top_p=0.8, repetition_penalty=1.02)
        return responds

    async def _astream(self, prompt: str, stop: Optional[List[str]] = None,
                     run_manager: Optional[CallbackManagerForLLMRun] = None,
                     **kwargs: Any) -> AsyncGenerator[str, None]:
        for chunk in self.model.stream_chat(self.tokenizer, prompt, temperature=0.5, top_p=0.8, repetition_penalty=1.02):
            yield chunk
        
    @property
    def _llm_type(self) -> str:
        return "MiniCPM_LLM"

def stream_generate_text(llm, text):
        """
        流式输出的方式打印到终端。
        原理就是每生成一个新token，就清空屏幕，然后把原先生成过的所有tokens
        都打印一次。
        """
        import os
        generated_text = ''
        count = 0
        message = [HumanMessage(text)]
        print(message)
        for new_text in llm._stream(message):
            generated_text += new_text.text
            count += 1
            if count % 8 == 0:  # 避免刷新太频繁，每8个tokens刷新一次
                os.system("clear")
                print(generated_text)
            os.system("clear")
            print(generated_text)

if __name__ == "__main__":

    path = 'D:/LLM/models/OpenBMB/MiniCPM-2B-sft-fp32'
    llm = MiniCPM_LLM(path)
    
    while True:
        query = input("请输入：")
        stream_generate_text(llm, query)
        print("")