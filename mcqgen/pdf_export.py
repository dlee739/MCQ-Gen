from __future__ import annotations

from io import BytesIO
from typing import Dict, List

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer


def _letters_for_choices(choices: List[Dict[str, str]]) -> Dict[str, str]:
    id_to_letter: Dict[str, str] = {}
    for idx, c in enumerate(choices):
        cid = c.get("id", "")
        if cid:
            id_to_letter[cid] = chr(ord("A") + idx)
    return id_to_letter


def _clean_explanation(text: str) -> str:
    # Basic cleanup for markdown-ish output in PDFs.
    cleaned = text.replace("**", "").replace("`", "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return cleaned.strip()


def _format_explanation_for_pdf(text: str) -> str:
    if not text:
        return ""
    lines = text.split("\n")
    out_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= 2 and stripped[1] == "." and stripped[0].isalpha():
            # Add a blank line before labeled options (A., B., C., ...).
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
        out_lines.append(stripped)
    return "<br/>".join(out_lines)


def build_results_pdf(
    *,
    output: Dict,
    answers: Dict[str, List[str]],
    title: str,
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=50,
        rightMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading2"], spaceAfter=12)
    q_style = ParagraphStyle("Q", parent=styles["Heading4"], spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], leading=14, spaceAfter=3)
    label_style = ParagraphStyle("Label", parent=styles["BodyText"], leading=14, spaceAfter=2, fontName="Helvetica-Bold")

    story: List = [Paragraph(title, title_style), Spacer(1, 6)]

    questions = output.get("questions", [])
    qtype = output.get("settings", {}).get("question_type", "MCQ")

    for q in questions:
        qid = str(q.get("id", ""))
        q_num = qid.replace("q_", "")
        stem = str(q.get("stem", ""))
        choices = q.get("choices", [])
        correct_ids = list(q.get("correct_choice_ids", []))
        user_ids = list(answers.get(qid, []))
        explanation = _format_explanation_for_pdf(
            _clean_explanation(str(q.get("explanation", "") or ""))
        )

        id_to_letter = _letters_for_choices(choices)
        correct_letters = sorted([id_to_letter.get(cid, "?") for cid in correct_ids])
        user_letters = sorted([id_to_letter.get(cid, "?") for cid in user_ids])
        is_correct = set(user_ids) == set(correct_ids)
        result_text = "Correct" if is_correct else "Incorrect"

        block: List = []
        block.append(Paragraph(f"Question {q_num}. {stem}", q_style))

        opts = "<br/>".join(f"{id_to_letter.get(c.get('id', ''), '?')}. {c.get('text', '')}" for c in choices)
        block.append(Paragraph(opts, body_style))
        block.append(Paragraph(f"<b>Your Answer:</b> {', '.join(user_letters)}", body_style))
        block.append(Paragraph(f"<b>Correct Answer:</b> {', '.join(correct_letters)}", body_style))
        block.append(Paragraph(f"<b>Result:</b> {result_text}", body_style))

        if explanation:
            block.append(Paragraph("Explanation:", label_style))
            block.append(Paragraph(explanation, body_style))

        block.append(Spacer(1, 12))
        story.append(KeepTogether(block))

    doc.build(story)
    return buffer.getvalue()
