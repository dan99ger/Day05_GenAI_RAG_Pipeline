import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_and_query_rag(query_text):
    print("--- 1. تحميل المستند وتقطيعه (Chunking) ---")
    loader = TextLoader("company_policy.txt", encoding="utf-8")
    documents = loader.load()

    # تقطيع النص إلى أجزاء بحجم 200 حرف مع تداخل 30 حرف لضمان السياق
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    print(f"تم تقطيع المستند إلى {len(docs)} أجزاء (Chunks).")

    print("\n--- 2. تحويل النصوص إلى Vector Embeddings وخزنها في ChromaDB ---")
    # استخدام نموذج Embeddings مجاني خفيف ويعمل محلياً
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db_dir = "artifacts/chroma_db"
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory=db_dir
    )
    print(f"تم بناء قاعدة بيانات المتجهات وحفظها في {db_dir}")

    print("\n--- 3. إجراء البحث بالتشابه (Similarity Search) ---")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(query_text)

    print(f"\n[السؤال المطلوب]: {query_text}\n")
    print("--- [المقاطع المسترجعة ذات الصلة (Retrieved Context)] ---")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"المقطع {i}: {doc.page_content}\n")

    # صياغة البرومبت النهائي الموجه للـ LLM
    context = "\n".join([d.page_content for d in retrieved_docs])
    prompt = f"""إليك السياق التالي من مستند الشركة:
{context}

بناءً على السياق أعلاه فقط، أجب عن السؤال التالي بشكل مباشر:
السؤال: {query_text}
الإجابة:"""
    
    print("--- [الـ Prompt النهائي الجاهز للإرسال للـ LLM] ---")
    print(prompt)

if __name__ == '__main__':
    user_query = "كم الميزانية المتاحة لتدريب المهندس سنوياً ومتى تُجدد؟"
    build_and_query_rag(user_query)