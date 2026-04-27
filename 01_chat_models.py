import os
from openai import OpenAI


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-turbo"


api_key = "sk-078d05fef71240278ece028d37bfe9c3"
if not api_key:
    raise RuntimeError("没有读取到环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url=QWEN_BASE_URL,
)

completion = client.chat.completions.create(
    model=QWEN_CHAT_MODEL,
    messages=[
        {
            "role": "system",
            "content": "你是自然语言处理课程助教，回答要准确、简洁。",
        },
        {
            "role": "user",
            "content": "请用三句话解释什么是自然语言处理。",
        },
    ],
    temperature=0.3,
)

answer = completion.choices[0].message.content
print(answer)