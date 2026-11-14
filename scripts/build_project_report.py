"""Render the editable project-report Markdown into a paginated PDF."""

import re
from html import escape
from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Preformatted, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/project-report.md"
TARGET = ROOT / "docs/PPE_Project_Comprehensive_Report.pdf"
NAVY = colors.HexColor("#17344B")
TEAL = colors.HexColor("#087F8C")
GRAY = colors.HexColor("#526370")
LIGHT = colors.HexColor("#EDF5F7")
PAGE_WIDTH, PAGE_HEIGHT = A4
WIDTH = PAGE_WIDTH - 100


def register_fonts():
    regular = Path("C:/Windows/Fonts/segoeui.ttf")
    bold = Path("C:/Windows/Fonts/segoeuib.ttf")
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
        pdfmetrics.registerFont(TTFont("ReportSansBold", str(bold)))
        pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSansBold",
                                    italic="ReportSans", boldItalic="ReportSansBold")
        return "ReportSans", "ReportSansBold"
    return "Helvetica", "Helvetica-Bold"


REGULAR, BOLD = register_fonts()
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("ReportBody", fontName=REGULAR, fontSize=9.8, leading=14,
                          textColor=NAVY, spaceAfter=7, splitLongWords=True))
styles.add(ParagraphStyle("Section", fontName=BOLD, fontSize=20, leading=25,
                          textColor=NAVY, spaceAfter=16, keepWithNext=True))
styles.add(ParagraphStyle("Subsection", fontName=BOLD, fontSize=11.7, leading=16,
                          textColor=TEAL, spaceBefore=9, spaceAfter=6, keepWithNext=True))
styles.add(ParagraphStyle("Cell", parent=styles["ReportBody"], fontSize=8.5, leading=11.5,
                          spaceAfter=0))
styles.add(ParagraphStyle("CellHead", parent=styles["Cell"], fontName=BOLD,
                          textColor=colors.white))
styles.add(ParagraphStyle("Caption", parent=styles["ReportBody"], fontSize=8.5,
                          leading=12, textColor=GRAY, spaceAfter=12))
styles.add(ParagraphStyle("CoverTitle", fontName=BOLD, fontSize=31, leading=38,
                          textColor=NAVY, spaceAfter=20))
styles.add(ParagraphStyle("CoverSubtitle", fontName=REGULAR, fontSize=15, leading=22,
                          textColor=TEAL, spaceAfter=22))
styles.add(ParagraphStyle("CodeBlock", fontName="Courier", fontSize=8, leading=11,
                          backColor=LIGHT, borderPadding=9, spaceBefore=5, spaceAfter=12))


def rich(text):
    value = escape(text)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r'<font color="#087F8C">\1</font>', value)
    if value.startswith("https://"):
        value = f'<link href="{value}" color="#087F8C">{value}</link>'
    return value


class ReportDocument(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "Section":
            title = flowable.getPlainText()
            key = "section-" + title.split(".", 1)[0]
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=0)
            self.notify("TOCEntry", (0, title, self.page, key))


def decorate(canvas, document):
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_HEIGHT - 10, PAGE_WIDTH, 10, fill=1, stroke=0)
    if document.page > 1:
        canvas.setFont(BOLD, 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(50, PAGE_HEIGHT - 33, "PPE COMPLIANCE | PROJECT REVIEW & ROADMAP")
    canvas.setStrokeColor(colors.HexColor("#D7E4E8"))
    canvas.line(50, 39, PAGE_WIDTH - 50, 39)
    canvas.setFont(REGULAR, 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(50, 25, "Status: 14 November 2026 | Verified results and proposed next phases")
    canvas.drawRightString(PAGE_WIDTH - 50, 25, str(document.page))
    canvas.restoreState()


def make_table(lines):
    rows = [[part.strip() for part in line.strip().strip("|").split("|")] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r"[:\- ]+", part) for part in row)]
    count = len(rows[0])
    if count == 2:
        fractions = [0.29, 0.71]
    elif count == 3:
        fractions = [0.24, 0.29, 0.47]
    elif count == 4:
        fractions = [0.28, 0.24, 0.24, 0.24]
    elif count == 5:
        fractions = [0.28, 0.18, 0.18, 0.18, 0.18]
    else:
        fractions = [1/count] * count
    data = [[Paragraph(rich(cell), styles["CellHead" if index == 0 else "Cell"])
             for cell in row] for index, row in enumerate(rows)]
    table = Table(data, colWidths=[WIDTH*f for f in fractions], repeatRows=1,
                  hAlign="LEFT", splitByRow=True)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, colors.white]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 8), ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEBELOW", (0,0), (-1,0), 0.5, TEAL),
        ("LINEBELOW", (0,1), (-1,-1), 0.25, colors.HexColor("#D7E4E8")),
    ]))
    return table


def build():
    story = [Spacer(1, 65), Paragraph("AI SEMESTER PROJECT", styles["Subsection"]),
             Spacer(1, 16), Paragraph("Real-Time PPE Compliance<br/>and Safety Violation Detection", styles["CoverTitle"]),
             Paragraph("Comprehensive project review<br/>Implementation roadmap and three-member role plan", styles["CoverSubtitle"]),
             Spacer(1, 15)]
    story.append(make_table([
        "| Snapshot | Position |",
        "| Completed | Environment, training data preparation, GPU baseline and local image inference |",
        "| Next | Worker/PPE association, uncertainty rules and per-worker image output |",
        "| Remaining | Video/camera integration, interface, independent evaluation and submission |",
    ]))
    story += [Spacer(1, 28), Paragraph("14 November 2026", styles["Subsection"]),
              Paragraph("Prepared from the supplied seven-page proposal presentation and verified project artifacts. Future designs and member responsibilities are explicitly proposed; independent final evaluation is still outstanding.", styles["ReportBody"]),
              Paragraph("Member A / Member B / Member C are assignable role labels. No names, institutional details or completion dates have been invented.", styles["Caption"]), PageBreak(),
              Paragraph("Contents", styles["CoverTitle"])]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("ContentsEntry", fontName=REGULAR, fontSize=9.4,
                                      leading=13, spaceBefore=3, textColor=NAVY)]
    story += [toc, Spacer(1, 18), Paragraph("Reading guide", styles["Subsection"]),
              Paragraph("Sections 1-3 establish scope and proposal alignment. Sections 4-9 explain completed work. Sections 10-14 describe the planned remaining phases. Sections 15-17 allocate responsibilities, handoffs and risks. Sections 18-19 provide reproduction details and evidence boundaries.", styles["ReportBody"])]
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = next(i for i,line in enumerate(lines) if line.startswith("## "))
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("## "):
            story += [PageBreak(), Paragraph(rich(line[3:]), styles["Section"])]
            index += 1
        elif line.startswith("### "):
            story.append(Paragraph(rich(line[4:]), styles["Subsection"]))
            index += 1
        elif line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                block.append(lines[index]); index += 1
            story += [make_table(block), Spacer(1, 12)]
        elif line.startswith("```"):
            index += 1
            code = []
            while index < len(lines) and not lines[index].startswith("```"):
                code.extend(wrap(lines[index], width=96, break_long_words=True,
                                 break_on_hyphens=False, replace_whitespace=False) or [""])
                index += 1
            story.append(Preformatted("\n".join(code), styles["CodeBlock"]))
            index += 1
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            caption, relative = match.groups()
            path = (SOURCE.parent / relative).resolve()
            figure = Image(str(path))
            maximum_height = 215 if "local-detection" in relative else 180
            scale = min(WIDTH / figure.imageWidth, maximum_height / figure.imageHeight)
            figure.drawWidth = figure.imageWidth * scale
            figure.drawHeight = figure.imageHeight * scale
            figure.hAlign = "CENTER"
            story.append(KeepTogether([figure, Spacer(1, 7), Paragraph(rich(caption), styles["Caption"])]))
            index += 1
        else:
            paragraph = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "```", "![")):
                paragraph.append(lines[index].strip()); index += 1
            story.append(Paragraph(rich(" ".join(paragraph)), styles["ReportBody"]))
    document = ReportDocument(str(TARGET), pagesize=A4, leftMargin=50, rightMargin=50,
                              topMargin=57, bottomMargin=54,
                              title="PPE Project: Comprehensive Review and Roadmap",
                              author="PPE Compliance Project Team")
    document.addPageTemplates(PageTemplate(id="report", frames=[Frame(50,54,WIDTH,PAGE_HEIGHT-111,
                                                                      leftPadding=0,rightPadding=0,
                                                                      topPadding=0,bottomPadding=0)],
                                          onPage=decorate))
    document.multiBuild(story)
    print(f"Created {TARGET} ({TARGET.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
