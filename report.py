# report.py — Person 4: PDF export (renders Person 3's insight schema)
import io
from datetime import date
from typing import Dict, Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

RESULT_MARK = {"survived": "[OK]", "weakened": "[WEAK]", "fatal": "[FATAL]", "not_run": "[NOT RUN]"}


def _table(rows, widths):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def build_pdf(data: Dict[str, Any], insight: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"{data.get('company', '')} Insight Report")
    ss = getSampleStyleSheet()
    h1, h2, body = ss["Title"], ss["Heading2"], ss["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)

    def P(text, style=body):
        return Paragraph(escape(str(text)), style)

    def bullets(items):
        for i in items:
            story.append(P(f"• {i}"))

    story = [
        P(f"{data.get('company', '')} ({data.get('ticker', '')}) — Insight Report", h1),
        P(f"{data.get('sector', '')} | Generated {date.today().isoformat()} | No investment recommendation.", small),
        Spacer(1, 6),
        P("Discovery", h2),
        P(insight.get("claim", "")),
        Paragraph(f"Evidence strength: <b>{escape(insight.get('strength', ''))}</b>", body),
    ]

    if insight.get("why_it_matters"):
        story.append(P("Why it matters", h2))
        bullets(insight["why_it_matters"])

    nums = insight.get("numbers", [])
    if nums:
        story.append(P("Numbers", h2))
        keys = [k for k in nums[0] if k != "evidence_ids"]
        rows = [[k.replace("_", " ") for k in keys]] + [[P(r.get(k, ""), small) for k in keys] for r in nums]
        story.append(_table(rows, [174 * mm / len(keys)] * len(keys)))

    ev = insight.get("evidence", [])
    if ev:
        story.append(P("Evidence", h2))
        rows = [["ID", "Fact", "Value", "Role / tier", "Source"]]
        for e in ev:
            name = escape(str(e.get("source_name", ""))[:70])
            link = f'<link href="{escape(e.get("source_url", ""))}" color="blue">{name}</link>' if e.get("source_url") else name
            fact = e.get("fact", "")
            if e.get("quote"):
                fact = f"{fact}: \"{e['quote'][:160]}\""
            val = "" if e.get("value") is None else f"{e['value']:,} {e.get('unit', '')}"
            rows.append([e.get("id", ""), P(fact, small), P(val, small),
                         P(f"{e.get('role', '')} / {e.get('tier', '')}", small), Paragraph(link, small)])
        story.append(_table(rows, [12 * mm, 60 * mm, 28 * mm, 26 * mm, 48 * mm]))

    if insight.get("counterarguments"):
        story.append(P("Counterarguments", h2))
        bullets([f"{RESULT_MARK.get(c['result'], '')} {c['check'].replace('_', ' ')}: {c['detail']}"
                 for c in insight["counterarguments"]])

    if insight.get("falsification_condition"):
        story.append(P("Falsification condition", h2))
        story.append(P(insight["falsification_condition"]))

    story.append(P("Evidence strength reasoning", h2))
    bullets(insight.get("strength_reasons", []))

    if insight.get("rejected_angles"):
        story.append(P("Angles the agent dropped", h2))
        bullets([f"{r['angle']}: {r['reason']}" for r in insight["rejected_angles"]])

    if insight.get("data_quality_flags") or insight.get("limitations"):
        story.append(P("Data quality and limitations", h2))
        bullets(insight.get("data_quality_flags", []) + insight.get("limitations", []))

    doc.build(story)
    return buf.getvalue()
