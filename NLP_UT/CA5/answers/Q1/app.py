import chainlit as cl
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the index (assuming it's saved)
try:
    with open("data/indexed_dataset.pkl", "rb") as f:
        store = pickle.load(f)
    docs = store["docs"]
    metadata = store["metadata"]
    vectorizer = store["vectorizer"]
    X = store["X"]
    print("Index loaded successfully for Chainlit.")
except FileNotFoundError:
    print("Error: 'data/indexed_dataset.pkl' not found. Please run the indexing steps first.")
    docs = []
    metadata = []
    vectorizer = None
    X = None

# Define helper functions
GREET_KEYWORDS = ["سلام", "درود", "صبح بخیر", "وقت بخیر" , "خوش آمدید", "خوش اومدید", "سلامتی"]
ABUSE_KEYWORDS = ["مسخره", "بی‌ادب", "احمق", "نادان", "بی‌فکر", "بی‌عقل", "بی‌شعور" , "بی‌خرد"]

def rewrite_query(query: str) -> str:
    q = query.strip()
    q = re.sub(r"\s+", " ", q)
    return q

def classify_intent(query: str) -> dict:
    q = query.lower()
    for w in GREET_KEYWORDS:
        if w in q:
            return {"intent": "greeting", "response": "سلام! چطور می‌توانم کمک کنم؟"}
    for w in ABUSE_KEYWORDS:
        if w in q:
            return {
                "intent": "abuse",
                "response": "من اینجا هستم تا کمک کنم؛ لطفاً مودبانه سوالاتتان را مطرح کنید.",
            }
    return {"intent": "law_question"}

def extract_metadata(rewritten_query: str) -> dict:
    m = {}
    # try to extract year
    years = re.findall(r"13\d{2}|14\d{2}|\d{4}", rewritten_query)
    if years:
        try:
            m["year"] = int(years[0])
        except:
            pass

    # Key extraction for filtering
    keywords = []
    if "کار" in rewritten_query:
        keywords.append("کار")
    if "مدنی" in rewritten_query:
        keywords.append("مدنی")
    if "جرایم" in rewritten_query or "رایانه" in rewritten_query:
        keywords.append("جرایم")
    if keywords:
        m["keywords"] = keywords
    return m

def context_retrieve(rewritten_query: str, metadata_filter: dict = None, K: int = 10):
    if vectorizer is None or X is None:
        return []
    q_vec = vectorizer.transform([rewritten_query])
    sims = cosine_similarity(q_vec, X).flatten()

    idxs = np.argsort(-sims)

    if metadata_filter:
        filtered = []
        for i in idxs:
            meta = metadata[i]
            ok = True
            if "year" in metadata_filter and "year" in meta:
                ok = ok and (meta.get("year") == metadata_filter["year"])

            if "keywords" in metadata_filter:
                found_k = any(
                    k in meta.get("title", "") or k in docs[i]
                    for k in metadata_filter["keywords"]
                )
                ok = ok and found_k

            if ok:
                filtered.append((i, sims[i]))
        results = filtered[:K]
    else:
        results = [(int(i), float(sims[int(i)])) for i in idxs[:K]]

    contexts = []
    for i, score in results:
        contexts.append(
            {"doc_id": i, "text": docs[i], "metadata": metadata[i], "score": score}
        )
    return contexts

def rerank(contexts, top_n=3, relevance_threshold=0.01):
    if not contexts:
        return []
    contexts = sorted(contexts, key=lambda x: -x["score"])
    top = contexts[:top_n]

    # relevance check
    if all(c["score"] < relevance_threshold for c in top):
        return []
    return top

def generate_answer(rewritten_query: str, contexts):
    if not contexts:
        return "متاسفانه نتوانستم پاسخی مرتبط در اسناد پیدا کنم. لطفاً سوال را با کلمات کلیدی دقیق‌تر مطرح کنید."

    parts = []
    for c in contexts:
        snippet = c["text"][:400] + "..."
        source = c["metadata"]["title"]
        parts.append(f"منبع: {source} (امتیاز: {c['score']:.2f})\nمتن: {snippet}")

    answer_body = "\n\n---\n\n".join(parts)
    final = f"پاسخ پیشنهادی (یافته شده در منابع):\n\n{answer_body}"
    return final

def run_pipeline(query: str, K=10, top_n=3):
    timings = {}
    t0 = time.time()
    timings["start"] = t0

    # Rewrite
    t = time.time()
    rewritten = rewrite_query(query)
    timings["rewrite"] = time.time() - t

    # Intent
    t = time.time()
    intent = classify_intent(rewritten)
    timings["classify_intent"] = time.time() - t

    if intent.get("intent") != "law_question":
        return {"answer": intent.get("response"), "timings": timings, "contexts": []}

    # Metadata
    t = time.time()
    meta_filter = extract_metadata(rewritten)
    timings["extract_metadata"] = time.time() - t

    # Retrieve
    t = time.time()
    contexts = context_retrieve(rewritten, meta_filter, K=K)
    timings["context_retrieve"] = time.time() - t

    # Rerank
    t = time.time()
    top = rerank(contexts, top_n=top_n)

    # Fallback: if empty, try wider search without filters
    if not top:
        contexts = context_retrieve(rewritten, None, K=min(len(docs), K * 2))
        top = rerank(contexts, top_n=top_n)
    timings["rerank"] = time.time() - t

    # Generate
    t = time.time()
    answer = generate_answer(rewritten, top)
    timings["generate_answer"] = time.time() - t

    timings["total"] = time.time() - t0
    return {"answer": answer, "timings": timings, "contexts": top}

# ==========================================
# بخش 2: تنظیمات Chainlit
# ==========================================

@cl.on_chat_start
async def start():
    """این تابع هنگام شروع چت جدید اجرا می‌شود"""
    welcome_message = """سلام! 👋
من دستیار هوشمند شما هستم. سوالی درباره مستندات دارید؟ بپرسید!"""

    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    """این تابع هنگام دریافت پیام از کاربر اجرا می‌شود"""

    # 1. نمایش وضعیت "در حال فکر کردن"
    msg = cl.Message(content="")
    await msg.send()

    # 2. اجرای پایپ‌لاین RAG (به صورتسинکرون یا آسنکرون)
    # چون توابع مدل معمولا زمان‌بر هستند، بهتر است با cl.make_async اجرا شوند
    # اما برای سادگی اینجا مستقیم صدا می‌زنیم (اگر تابع شما async است از await استفاده کنید)

    response = await cl.make_async(run_pipeline)(message.content)

    # 3. استخراج جواب و منابع
    answer_text = response.get("answer", "متاسفانه پاسخی یافت نشد.")
    contexts = response.get("contexts", [])

    # 4. آماده‌سازی منابع برای نمایش در UI
    source_elements = []
    if contexts:
        for i, ctx in enumerate(contexts):
            text_content = ctx.get("text", "")
            source_name = f"منبع {i+1}"

            # ایجاد المان متنی برای نمایش منبع
            source_elements.append(
                cl.Text(name=source_name, content=text_content, display="inline")
            )

            # ارجاع به منبع در متن پاسخ (اختیاری)
            # answer_text += f"\n[{source_name}]"

    # 5. ارسال پاسخ نهایی به کاربر
    msg.content = answer_text
    msg.elements = source_elements

    await msg.update()
