from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import json

embedModelName = r"E:\Code\test_py\lunwen\embeddingmodel\bge-m3"
embedModel = HuggingFaceEmbeddings(model_name=embedModelName)

with open("data/train_cmrc2018.json","r",encoding="utf8") as file:
    datas = json.load(file)
contexts = []
for data in datas:
    for index,doc in data["top_k_docs"].items():
        if doc not in contexts:
            contexts.append(doc)
print(len(contexts))
metadata = [{"source": f"context_{i}"} for i in range(len(contexts))]  # 为每个文本添加元数据

vector_store_path='dataset/train_cmrc2018_index'
if not os.path.exists(vector_store_path):
    vector_store = FAISS.from_texts(texts=contexts, embedding=embedModel, metadatas=metadata)
    vector_store.save_local(vector_store_path)
    print("向量化成功")
else:
    vector_store = FAISS.load_local(vector_store_path, embeddings=embedModel,allow_dangerous_deserialization=True)



def get_negs(query,pos):
    negs=set()
    results = vector_store.similarity_search_with_score(query, k=100)  # 返回最相似的2条结果
    for i, (result, score) in enumerate(results):
        if result.page_content not in pos:
            if score>=0.3 and score<=3:
                negs.add(result.page_content)
                print(score)
                if len(negs)>=5:
                    break
    return list(negs)

if __name__ == '__main__':
    total=[]
    for data in datas:
        query = data["question"]
        pos = list(data["top_k_docs"].values())
        negs = get_negs(query,pos)
        data["negs"] = negs
        total.append(data)
    with open("data/after_train_cmrc2018.json","w",encoding="utf8") as f:
        json.dump(total,f,ensure_ascii=False,indent=4)