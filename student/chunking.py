"""✏️ ส่วนที่ 1: Chunking — แบ่งเอกสารเป็นชิ้น (chunk)

เอกสารที่ได้รับเป็น markdown เช่น lecture note หรือ case study ที่มีหัวข้อ (#, ##), ตาราง และ code block
ชิ้นที่แบ่งแล้วจะถูกนำไปทำ embedding และใช้ค้นหาตอนตอบคำถาม
แบ่งดี = ค้นเจอเนื้อหาที่ตรงคำถามและครบถ้วน

ค่าตั้งต้นตอนยังไม่ทำ: ไม่แบ่งเลย ทั้งเอกสารเป็น chunk เดียว
ค่าที่ปรับได้ (CHUNK_SIZE, CHUNK_OVERLAP, ...) อยู่ใน student/settings.py
"""
from student import settings  # noqa: F401


import re


def _split_long_text(text: str, max_size: int, overlap: int) -> list[str]:
    """ตัดข้อความยาวให้เป็นชิ้นย่อย ๆ ตามขนาด max_size และ overlap โดยพยายามไม่ตัดกลางบรรทัดหรือย่อหน้า"""
    if len(text) <= max_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_size:
            lines = para.split("\n")
            for line in lines:
                if len(line) > max_size:
                    step = max(1, max_size - overlap)
                    for i in range(0, len(line), step):
                        sub = line[i:i + max_size]
                        if sub.strip():
                            chunks.append(sub)
                else:
                    if current_len + len(line) + 1 > max_size and current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                    current_chunk.append(line)
                    current_len += len(line) + 1
        else:
            if current_len + para_len + 2 > max_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(para)
            current_len += para_len + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


def split(doc_id: str, text: str) -> list[str]:
    """แบ่งเอกสาร 1 ชิ้นเป็น list ของข้อความ

    รับ:  doc_id  ชื่อเอกสาร เช่น "session04_01_prompt_fundamentals"
          text    เนื้อหาทั้งหมดของเอกสาร (markdown)
    คืน:  list[str] ข้อความของแต่ละ chunk (ห้ามว่าง)

    แนวทาง:
      - แยกตามหัวข้อ markdown (บรรทัดที่ขึ้นต้นด้วย #) แล้วใส่ชื่อหัวข้อไว้ต้น chunk ด้วย
      - หัวข้อที่ยาวเกินไปให้ตัดย่อยเป็นช่วง ๆ (เช่น 600–1,000 ตัวอักษร) และซ้อนกันเล็กน้อย (overlap)
      - พยายามอย่าตัดกลางตาราง กลาง code block หรือกลางประโยค
    """
    clean_text = text.replace("\r\n", "\n").strip()
    if not clean_text:
        return [doc_id]

    max_size = getattr(settings, "CHUNK_SIZE", 900)
    overlap = getattr(settings, "CHUNK_OVERLAP", 150)
    split_by_heading = getattr(settings, "SPLIT_BY_HEADING", True)
    include_heading = getattr(settings, "INCLUDE_HEADING", True)

    if not split_by_heading:
        return _split_long_text(clean_text, max_size, overlap) or [clean_text]

    lines = clean_text.split("\n")
    sections: list[dict] = []
    heading_stack: list[tuple[int, str]] = []

    current_lines: list[str] = []
    current_title = ""
    current_path = ""

    heading_pattern = re.compile(r"^(#{1,6})\s+(.*)$")

    for line in lines:
        match = heading_pattern.match(line.strip())
        if match:
            if current_lines or current_title:
                body = "\n".join(current_lines).strip()
                if body or current_title:
                    sections.append({
                        "path": current_path,
                        "title": current_title,
                        "body": body
                    })
                current_lines = []

            level = len(match.group(1))
            heading_text = match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))

            current_path = " > ".join(h[1] for h in heading_stack)
            current_title = heading_text
        else:
            current_lines.append(line)

    if current_lines or current_title:
        body = "\n".join(current_lines).strip()
        if body or current_title:
            sections.append({
                "path": current_path,
                "title": current_title,
                "body": body
            })

    if not sections:
        return _split_long_text(clean_text, max_size, overlap) or [clean_text]

    chunks: list[str] = []
    for sec in sections:
        prefix = f"[{sec['path']}]\n" if (include_heading and sec["path"]) else ""
        body = sec["body"]
        full_content = f"{prefix}{body}".strip() if body else prefix.strip()

        if len(full_content) <= max_size:
            if full_content:
                chunks.append(full_content)
        else:
            sub_chunks = _split_long_text(body, max(100, max_size - len(prefix)), overlap)
            for sub in sub_chunks:
                piece = f"{prefix}{sub}".strip()
                if piece:
                    chunks.append(piece)

    final_chunks = [c.strip() for c in chunks if c.strip()]
    return final_chunks if final_chunks else [clean_text]
