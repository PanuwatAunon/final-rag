"""✏️ ส่วนที่ 4: Prompt — บอก LLM ว่าจะให้ทำอะไร

คืนค่าเป็น list ของ message แบบ chat:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
(หรือคืนเป็นสตริงก็ได้ ระบบจะถือว่าเป็นข้อความของ user)

hits คือผลจาก retrieval.search เป็น list ของ (chunk, score) อ่านเนื้อหาได้จาก chunk.text และ chunk.doc_id

ค่าตั้งต้นตอนยังไม่ทำ: ไม่เรียก LLM เลย ข้อสอบปรนัยตอบ "ก" ทุกข้อ และโจทย์โค้ดได้ฟังก์ชันว่างที่ raise NotImplementedError
(ทำส่วนนี้ก่อน เพราะถ้ายังไม่ทำ ส่วนอื่นจะไม่มีผลต่อคะแนนเลย)
"""
from student import settings  # noqa: F401 — MAX_CONTEXT_CHARS


def build_mc_prompt(question: str, options: list[tuple[str, str]], hits) -> list[dict]:
    """prompt สำหรับข้อสอบปรนัย

    รับ:  question  คำถาม
          options   list ของ (key, text) เช่น [("ก", "O(1)"), ("ข", "O(log n)"), ...]
          hits      chunk ที่ค้นได้ (list ของ (chunk, score)) อาจเป็น list ว่าง
    คืน:  list ของ message

    หมายเหตุ: core บังคับให้ LLM ตอบเป็น JSON {"reason": "...", "answer": "<key>"} ให้อยู่แล้ว
    แนวทาง:
      - ใส่เนื้อหาของ hits เป็น "เอกสารอ้างอิง" แล้วสั่งให้ยึดเอกสารเป็นหลัก แม้จะขัดกับความรู้ทั่วไป
      - แสดงตัวเลือกพร้อม key (ก. ... ข. ...)
      - ระวังคำถามเชิงปฏิเสธ เช่น "ข้อใด**ไม่**ถูกต้อง"
    """
    context_parts = []
    total_chars = 0
    if hits:
        for chunk, score in hits:
            text = chunk.text.strip()
            if not text:
                continue
            if total_chars + len(text) > settings.MAX_CONTEXT_CHARS:
                remaining = settings.MAX_CONTEXT_CHARS - total_chars
                if remaining > 100:
                    context_parts.append(f"[{chunk.doc_id}]\n{text[:remaining]}...")
                break
            context_parts.append(f"[{chunk.doc_id}]\n{text}")
            total_chars += len(text)

    docs_section = "\n\n---\n\n".join(context_parts) if context_parts else "ไม่มีเอกสารอ้างอิง"
    options_text = "\n".join(f"{k}. {t}" for k, t in options)

    system_content = (
        "คุณคือ AI ผู้ช่วยทำข้อสอบปรนัยที่มีความรอบคอบและแม่นยำสูง\n"
        "กฎสำคัญ:\n"
        "1. ยึดข้อมูลจาก 'เอกสารอ้างอิง' ที่ให้มาเป็นหลักสำคัญที่สุด แม้จะขัดแย้งกับความรู้ทั่วไปหรือสามัญสำนึกก็ตาม\n"
        "2. วิเคราะห์คำถามอย่างถี่ถ้วน โดยเฉพาะกรณีคำถามมีคำปฏิเสธ เช่น 'ไม่', 'ไม่ใช่', 'ยกเว้น', 'ผิด'\n"
        "3. ให้เหตุผลสั้น ๆ ในช่อง reason แล้วระบุคีย์ตัวเลือก (เช่น ก, ข, ค, ...) ที่ถูกต้องที่สุดในช่อง answer\n"
        f"\n=== เอกสารอ้างอิง ===\n{docs_section}"
    )

    user_content = (
        f"คำถาม:\n{question}\n\n"
        f"ตัวเลือก:\n{options_text}\n\n"
        "จงเลือกคำตอบที่ถูกต้องที่สุดตามเอกสารอ้างอิง"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_code_prompt(requirement: str, signature: str | None, entry_point: str | None, hits) -> list[dict]:
    """prompt สำหรับโจทย์เขียนโปรแกรม

    รับ:  requirement  โจทย์ (ภาษาไทย)
          signature    บรรทัด def ที่ต้องใช้ เช่น "def is_prime(n: int) -> bool:"
          entry_point  ชื่อฟังก์ชัน เช่น "is_prime"
          hits         chunk ที่ค้นได้ (มีเฉพาะโจทย์ที่อ้างถึงเอกสาร ส่วนใหญ่จะเป็น list ว่าง)
    คืน:  list ของ message

    หมายเหตุ: core ดึงโค้ดจาก ```python ... ``` ในคำตอบให้ และถ้า compile ไม่ผ่านจะขอให้ LLM แก้ให้อีก 1 รอบ
    แนวทาง:
      - สั่งให้ใช้ชื่อฟังก์ชันและ signature ตามที่ให้มาเป๊ะ ๆ และใช้เฉพาะ Python standard library
      - ห้าม print หรือรับ input เอง ให้ return ค่า
      - อ่านโจทย์ให้ดี บางข้อ "ห้ามใช้" ฟังก์ชันบางตัว (เช่น max, sorted, eval) ถ้าใช้จะได้ 0 ทั้งข้อ
      - ให้ตอบเป็น code block เดียว
    """
    context_parts = []
    total_chars = 0
    if hits:
        for chunk, score in hits:
            text = chunk.text.strip()
            if not text:
                continue
            if total_chars + len(text) > settings.MAX_CONTEXT_CHARS:
                break
            context_parts.append(f"[{chunk.doc_id}]\n{text}")
            total_chars += len(text)

    docs_section = "\n\n---\n\n".join(context_parts) if context_parts else ""

    target_signature = signature or (f"def {entry_point}(*args, **kwargs):" if entry_point else "")

    system_content = (
        "คุณคือโปรแกรมเมอร์ Python 3 ผู้เชี่ยวชาญ จงเขียนโค้ดเพื่อแก้ปัญหาตามโจทย์ที่ได้รับ\n"
        "ข้อกำหนดที่ต้องปฏิบัติตามอย่างเคร่งครัด:\n"
        "1. ใช้เฉพาะ Python Standard Library เท่านั้น ห้ามนำเข้า library ภายนอก\n"
        "2. ห้ามรับ input ด้วย input() และห้าม print() คำตอบ ให้ return ผลลัพธ์จากฟังก์ชันเท่านั้น\n"
        "3. ตรวจสอบข้อห้ามในโจทย์อย่างเคร่งครัด หากโจทย์ระบุว่า 'ห้ามใช้' ฟังก์ชัน ตัวแปร หรือคำสั่งใด (เช่น max, min, sorted, eval ฯลฯ) ห้ามใช้เด็ดขาด\n"
        "4. หากมีเอกสารอ้างอิง ให้ปฏิบัติตามกฎ นิยาม และเงื่อนไขในเอกสารอย่างเคร่งครัด\n"
        "5. เขียนโค้ดที่สมบูรณ์และรันได้จริงไว้ในบล็อก ```python ... ``` เพียงบล็อกเดียวเท่านั้น โดยไม่ต้องใส่ข้อความอธิบายใด ๆ เพิ่มเติมข้างนอก"
    )

    user_prompt_lines = [f"โจทย์:\n{requirement}"]
    if target_signature:
        user_prompt_lines.append(f"\nฟังก์ชันที่ต้องเขียน (ต้องใช้ชื่อและ signature นี้เป๊ะ ๆ):\n{target_signature}")
    if docs_section:
        user_prompt_lines.append(f"\n=== เอกสารอ้างอิง ===\n{docs_section}")
    user_prompt_lines.append("\nจงเขียนโค้ด Python ที่ถูกต้อง สมบูรณ์ และผ่านการทดสอบทุกกรณี:")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_prompt_lines)},
    ]
