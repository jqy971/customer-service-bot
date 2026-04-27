from openai import OpenAI
import numpy as np

# 初始化客户端（直接写API Key，避免环境变量问题）
client = OpenAI(
    api_key="sk-078d05fef71240278ece028d37bfe9c3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 余弦相似度计算函数（参考代码）
def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

# 待处理文本
texts = [
    "我喜欢自然语言处理，尤其是大语言模型。",
    "大模型可以完成文本生成、摘要和问答任务。",
    "今天学校食堂的红烧肉很好吃。",
    "语义向量可以用来计算两个句子的相似度。",
]

# 步骤1：获取所有句子的向量
embeddings = []
for text in texts:
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=text
    )
    embeddings.append(response.data[0].embedding)

# 步骤2：计算两两相似度并打印
print("=== 句子两两相似度矩阵 ===")
for i in range(len(texts)):
    for j in range(len(texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"句子{i+1} 与 句子{j+1} 相似度: {sim:.4f}")
    print("---")

# 步骤3：查找与目标句子语义最相似的句子
target_sentence = "语义向量有哪些作用"
# 获取目标句子的向量
target_embedding = client.embeddings.create(
    model="text-embedding-v4",
    input=target_sentence
).data[0].embedding

# 计算与所有句子的相似度
similarities = []
for idx, emb in enumerate(embeddings):
    sim = cosine_similarity(target_embedding, emb)
    similarities.append((idx, sim))

# 按相似度从高到低排序
similarities.sort(key=lambda x: x[1], reverse=True)

print("\n=== 与目标句子最相似的句子 ===")
print(f"目标句子：{target_sentence}")
print("相似度排序：")
for idx, sim in similarities:
    print(f"句子{idx+1}（相似度：{sim:.4f}）：{texts[idx]}")