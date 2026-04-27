import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 千问模型配置
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-plus"

# 获取API Key
api_key = os.getenv("DASHSCOPE_API_KEY")

# 初始化OpenAI客户端
def get_client():
    if not api_key:
        raise RuntimeError("没有读取到环境变量 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=QWEN_BASE_URL,
    )

# 根路径，返回前端页面
@app.get("/")
async def root():
    return JSONResponse({"message": "请访问 /static/index.html"})

# 聊天接口
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        if not message:
            return JSONResponse({"error": "请输入消息"}, status_code=400)
        
        client = get_client()
        completion = client.chat.completions.create(
            model=QWEN_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "你是自然语言处理课程助教，回答要准确、简洁。",
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            temperature=0.3,
        )
        
        answer = completion.choices[0].message.content
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# 健康检查接口
@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})
