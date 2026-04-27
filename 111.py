from openai import OpenAI
import os
# 标准初始化方式（修正了括号问题）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 直接填从控制台复制的完整密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 调用免费的qwen-turbo模型
completion = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你认识麻夏文吗"},
    ]
)

# 直接打印模型回答
print(completion.choices[0].message.content)