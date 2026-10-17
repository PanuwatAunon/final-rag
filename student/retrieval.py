"""✏️ ส่วนที่ 3: Retrieval — ค้นหา chunk ที่เกี่ยวข้องกับคำถาม

ค่าตั้งต้นตอนยังไม่ทำ: ไม่คืน chunk ใดเลย LLM จึงไม่เห็นเอกสาร
"""
from core.types import Chunk
from core.utils import cosine, cosine_many  # noqa: F401 — ตัวช่วยคำนวณความคล้ายของ vector
from student import settings  # noqa: F401 — TOP_K, QUERY_WITH_OPTIONS, HYBRID_WEIGHT, MIN_SCORE


import re


def _extract_query_keywords(text: str) -> list[str]:
    """แยกคำค้นหาทั้งภาษาไทย ตัวเลข และภาษาอังกฤษ/รหัสเฉพาะ"""
    keywords = set()
    lowered = text.lower()

    codes_and_numbers = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", lowered)
    for token in codes_and_numbers:
        if len(token) >= 2:
            keywords.add(token)

    thai_blocks = re.findall(r"[\u0e00-\u0e7f]+", text)
    for block in thai_blocks:
        if 2 <= len(block) <= 15:
            keywords.add(block)
        if len(block) > 4:
            for n in (3, 4):
                for i in range(len(block) - n + 1):
                    keywords.add(block[i:i + n])

    return [k for k in keywords if len(k) >= 2]


def _compute_keyword_score(chunk_text: str, keywords: list[str]) -> float:
    """คำนวณคะแนนความตรงกันของคำสำคัญ (Keyword overlap) ในช่วง 0.0 ถึง 1.0"""
    if not keywords or not chunk_text:
        return 0.0
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return min(1.0, matches / len(keywords))


def make_query(question: str, options: list[str]) -> str:
    """สร้างข้อความที่จะใช้ค้นหา

    รับ:  question  คำถาม (หรือ requirement ของโจทย์เขียนโค้ด ซึ่งกรณีนี้ options จะเป็น list ว่าง)
          options   ข้อความของตัวเลือก ก–จ (ไม่มีตัวอักษร key นำหน้า)
    คืน:  สตริงคำค้น (ห้ามว่าง)

    แนวทาง: รวมคำถามกับตัวเลือกเข้าด้วยกัน เพราะคำตอบมักอยู่ในย่อหน้าที่มีคำของตัวเลือก
    """
    clean_q = question.strip()
    query_with_options = getattr(settings, "QUERY_WITH_OPTIONS", True)
    if query_with_options and options:
        clean_opts = [o.strip() for o in options if o and o.strip()]
        if clean_opts:
            return f"{clean_q}\n" + " ".join(clean_opts)
    return clean_q or "query"


def search(chunks: list[Chunk], query_text: str, query_vector: list[float] | None, k: int) -> list[tuple[Chunk, float]]:
    """เลือก chunk ที่เกี่ยวข้องที่สุด k ชิ้น

    รับ:  chunks        chunk ทั้งหมดของเอกสารชุดนี้ (chunk.text, chunk.vector, chunk.doc_id)
          query_text    คำค้น (จาก make_query)
          query_vector  vector ของคำค้น (จาก embedding.embed_query) หรือ None ถ้ายังไม่ได้ทำ embedding
          k             จำนวนที่ระบบแนะนำ (5) จะใช้ settings.TOP_K ของตัวเองแทนก็ได้
    คืน:  list ของ (chunk, score) เรียงจากเกี่ยวข้องมากไปน้อย
    """
    if not chunks:
        return []

    top_k = getattr(settings, "TOP_K", k) or k
    hybrid_weight = getattr(settings, "HYBRID_WEIGHT", 0.3)
    min_score = getattr(settings, "MIN_SCORE", None)

    has_vectors = query_vector is not None and any(c.vector is not None for c in chunks)
    vec_scores: dict[int, float] = {}

    if has_vectors:
        valid_indices = [i for i, c in enumerate(chunks) if c.vector is not None]
        if valid_indices:
            vectors = [chunks[i].vector for i in valid_indices]
            raw_scores = cosine_many(query_vector, vectors)
            for idx, s in zip(valid_indices, raw_scores):
                vec_scores[idx] = max(0.0, float(s))

    keywords = _extract_query_keywords(query_text)
    kw_scores = [_compute_keyword_score(c.text, keywords) for c in chunks]

    scored_chunks: list[tuple[Chunk, float]] = []
    for i, chunk in enumerate(chunks):
        if has_vectors and i in vec_scores:
            vs = vec_scores[i]
            ks = kw_scores[i]
            score = (1.0 - hybrid_weight) * vs + hybrid_weight * ks
        else:
            score = kw_scores[i]

        if min_score is None or score >= min_score:
            scored_chunks.append((chunk, float(score)))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]
