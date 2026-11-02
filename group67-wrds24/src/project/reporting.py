"""Typeset the English results write-up without changing empirical outputs."""
from pathlib import Path
import html
import re
from . import paths


def build_pdf():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,LongTable,TableStyle,Image,KeepTogether
    from reportlab.lib.utils import ImageReader
    font=Path('/usr/share/fonts/truetype/dejavu')
    if not (font/'DejaVuSans.ttf').is_file():
        raise FileNotFoundError('Install the DejaVu Sans fonts before rebuilding the PDF; table/figure rebuilding does not need them.')
    for name,file in [('Report','DejaVuSans.ttf'),('Report-Bold','DejaVuSans-Bold.ttf'),
                      ('Report-Oblique','DejaVuSans-Oblique.ttf'),('Report-BoldOblique','DejaVuSans-BoldOblique.ttf')]:
        selected=font/file
        if not selected.is_file():
            selected=font/('DejaVuSans-Bold.ttf' if 'Bold' in file else 'DejaVuSans.ttf')
        pdfmetrics.registerFont(TTFont(name,str(selected)))
    pdfmetrics.registerFontFamily('Report',normal='Report',bold='Report-Bold',italic='Report-Oblique',boldItalic='Report-BoldOblique')
    navy=colors.HexColor('#17324D');teal=colors.HexColor('#127D89');gray=colors.HexColor('#425466')
    body=ParagraphStyle('Body',fontName='Report',fontSize=9.1,leading=13.3,spaceAfter=7,textColor=colors.HexColor('#202C38'))
    title=ParagraphStyle('Title',parent=body,fontName='Report-Bold',fontSize=20,leading=25,textColor=navy,spaceAfter=13,keepWithNext=True)
    heading=ParagraphStyle('Heading',parent=body,fontName='Report-Bold',fontSize=13,leading=17,textColor=navy,spaceBefore=15,spaceAfter=7,keepWithNext=True)
    subheading=ParagraphStyle('Subheading',parent=heading,fontSize=10.5,leading=14,spaceBefore=10)
    cell=ParagraphStyle('Cell',parent=body,fontSize=7.5,leading=10.4,spaceAfter=0,splitLongWords=True)
    headcell=ParagraphStyle('HeadCell',parent=cell,fontName='Report-Bold',textColor=colors.white)
    caption=ParagraphStyle('Caption',parent=body,fontSize=8,leading=11,textColor=gray,spaceAfter=10)
    bullet=ParagraphStyle('Bullet',parent=body,leftIndent=12,firstLineIndent=0)
    report_dir=paths.ROOT/'report';source=report_dir/'phase1_results.md';output=report_dir/'phase1_results.pdf'
    width=516
    document=SimpleDocTemplate(str(output),pagesize=(612,792),leftMargin=48,rightMargin=48,
        topMargin=49,bottomMargin=46,title='Slow Momentum with Fast Reversion on a WRDS Futures Panel',
        author='Enze Hu and Jinrui Hu - Group 67',subject='WRDS method replication and conditional universe-size research hypothesis')
    def plain(text):
        return text.replace('\u2011','-').replace('\u2013','-').replace('\u2014',' - ').replace('\u2212','-')
    def inline(text):
        text=html.escape(plain(text))
        text=re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)',lambda m:'<link href="'+m[2]+'" color="#127D89">'+m[1]+'</link>',text)
        text=re.sub(r'`([^`]+)`',r'<font color="#36465B">\1</font>',text)
        text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
        text=re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',text)
        return text
    def paragraph(text,style=body):return Paragraph(inline(text),style)
    def column_widths(rows):
        count=len(rows[0])
        if count==3:return [103,189,224]
        if count==4:return [104,154,104,154]
        if count==5:return [168,91,84,80,93]
        if count==6:return [111,35,163,70,70,67]
        if count==2:return [342,174]
        return [width/count]*count
    story=[];lines=source.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                entries=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c or ' ') for c in entries):rows.append(entries)
                i+=1
            cells=[[paragraph(value,headcell if r==0 else cell) for value in row] for r,row in enumerate(rows)]
            table=LongTable(cells,colWidths=column_widths(rows),repeatRows=1,hAlign='LEFT',spaceBefore=4,spaceAfter=12)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),
                 ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                 ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                 ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F5F8')]),
                 ('LINEBELOW',(0,0),(-1,0),.5,navy),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#DCE5EC'))]))
            story.append(table);continue
        match=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if match:
            image_path=(report_dir/match[2]).resolve()
            if not image_path.is_file():raise FileNotFoundError('Missing report image: '+str(image_path))
            iw,ih=ImageReader(str(image_path)).getSize();height=width*ih/iw
            story.append(KeepTogether([Image(str(image_path),width=width,height=height),paragraph(match[1],caption)]));i+=1;continue
        if line.startswith('# '):story.append(paragraph(line[2:],title));i+=1;continue
        if line.startswith('## '):story.append(paragraph(line[3:],heading));i+=1;continue
        if line.startswith('### '):story.append(paragraph(line[4:],subheading));i+=1;continue
        match=re.match(r'(- |\d+\. )(.*)',line)
        if match:
            marker='- ' if match[1]=='- ' else match[1]
            text=marker+match[2];i+=1
            while i<len(lines) and lines[i].strip() and not re.match(r'(- |\d+\. |#|\||!\[)',lines[i].strip()):
                text+=' '+lines[i].strip();i+=1
            story.append(paragraph(text,bullet));continue
        parts=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'(#|\||!\[|- |\d+\. )',lines[i].strip()):
            parts.append(lines[i].strip());i+=1
        story.append(paragraph(' '.join(parts)))
    def page(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#DCE5EC'));canvas.line(48,31,564,31)
        canvas.setFont('Report',7);canvas.setFillColor(gray)
        canvas.drawString(48,20,'GROUP 67 | WRDS replication and cross-study evidence')
        canvas.drawRightString(564,20,str(doc.page));canvas.restoreState()
    document.build(story,onFirstPage=page,onLaterPages=page)
    print('REPORT_PDF_OK:',output)
    return output


if __name__=='__main__':build_pdf()
