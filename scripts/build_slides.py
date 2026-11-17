"""Build an offline presentation PDF from submission-slides.md (.[docs])."""

from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from build_project_report import REGULAR, BOLD

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 960, 540
INK = colors.HexColor('#163631')
TEAL = colors.HexColor('#087F6F')


def paragraph(document, text, x, y, width, style):
    block = Paragraph(escape(text), style)
    _, height = block.wrap(width, HEIGHT)
    block.drawOn(document, x, y-height)
    return y-height


def build():
    source = (ROOT/'docs/submission-slides.md').read_text(encoding='utf-8')
    slides = re.split(r'^## ', source, flags=re.MULTILINE)[1:]
    destination = ROOT/'docs/PPE_Presentation.pdf'
    document = canvas.Canvas(str(destination), pagesize=(WIDTH, HEIGHT))
    document.setTitle('PPE Vision | Presentation')
    document.setAuthor('PPE Compliance Project Team')
    title = ParagraphStyle('SlideTitle', fontName=BOLD, fontSize=32, leading=39, textColor=INK)
    bullet = ParagraphStyle('SlideBullet', fontName=REGULAR, fontSize=20, leading=29, textColor=INK)
    for index, slide in enumerate(slides, 1):
        lines = slide.splitlines()
        heading = re.sub(r'^\d+\.\s*', '', lines[0])
        document.setFillColor(colors.HexColor('#F5F7F4'))
        document.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
        document.setFillColor(TEAL)
        document.rect(0, 0, 14, HEIGHT, fill=1, stroke=0)
        document.setFont(BOLD, 11)
        document.drawString(54, HEIGHT-43, 'PPE VISION / WORKER EVIDENCE')
        paragraph(document, heading, 54, HEIGHT-82, WIDTH-108, title)
        y = HEIGHT-170
        illustration = ROOT/'docs/assets/ppe-vision-result.png'
        body_width = 385 if index == 6 and illustration.is_file() else WIDTH-145
        for line in lines[1:]:
            if not line.startswith('- '):
                continue
            document.setFillColor(TEAL)
            document.circle(62, y-13, 3, fill=1, stroke=0)
            y = paragraph(document, line[2:], 82, y, body_width, bullet)-19
        if index == 6 and illustration.is_file():
            document.drawImage(str(illustration), 500, 70, width=410, height=307.5,
                               preserveAspectRatio=True, anchor='c', mask='auto')
        if y < 55:
            raise ValueError(f'Slide {index} content overflows; shorten the source bullets')
        document.setFillColor(colors.HexColor('#536B64'))
        document.setFont(REGULAR, 10)
        document.drawString(54, 27, 'Development prototype | Independent evaluation pending')
        document.drawRightString(WIDTH-45, 27, f'{index:02d} / {len(slides):02d}')
        document.showPage()
    document.save()
    print(f'Created {destination} ({len(slides)} slides)')


if __name__ == '__main__':
    build()
