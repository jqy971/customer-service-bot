import os
import requests
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from openai import OpenAI

app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===================== 原千问配置 =====================
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen-plus"
api_key = os.getenv("DASHSCOPE_API_KEY")

# ===================== FastGPT 配置 =====================
FASTGPT_API_KEY = "fastgpt-wSXwXkclYFSF6KDclAcaOqTwqGZyAfCOxOj7nmAsS9zUS552ItMT0DyNbG3FGpx"
FASTGPT_BASE_URL = "https://cloud.fastgpt.cn/api"
FASTGPT_CHAT_ID = "master"


def get_client():
    if not api_key:
        raise RuntimeError("没有读取到环境变量 DASHSCOPE_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url=QWEN_BASE_URL,
    )


# 根路径
@app.get("/")
async def root():
    return JSONResponse({"message": "请访问 /static/index.html"})


# 健康检查
@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})


# ===================== 原千问聊天接口 =====================
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
                {"role": "system", "content": "你是自然语言处理课程助教，回答要准确、简洁。"},
                {"role": "user", "content": message}
            ],
            temperature=0.3,
        )
        answer = completion.choices[0].message.content
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ===================== FastGPT Agent 接口 =====================
@app.post("/fastgpt")
async def fastgpt_chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message")
        if not message:
            return JSONResponse({"error": "请输入消息"}, status_code=400)

        headers = {
            "Authorization": f"Bearer {FASTGPT_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "chatId": FASTGPT_CHAT_ID,
            "stream": False,
            "detail": False,
            "messages": [
                {"role": "user", "content": message}
            ]
        }

        response = requests.post(
            url=f"{FASTGPT_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            return JSONResponse({"answer": answer})
        else:
            return JSONResponse({
                "error": f"FastGPT请求失败 {response.status_code}",
                "detail": response.text
            }, status_code=response.status_code)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
