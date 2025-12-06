import re
from latex2html import convert_html_tables_to_markdown, latex_table_to_html

def extract_classes_bboxes(text: str):
    _re_extract_class_bbox = re.compile(r'<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>', re.DOTALL)
    classes = []
    bboxes = []
    texts = []
    for m in _re_extract_class_bbox.finditer(text):
        x1, y1, text, x2, y2, cls = m.groups()
        classes.append(cls)
        bboxes.append((float(x1), float(y1), float(x2), float(y2)))
        texts.append(text)

    # TODO: Remove when fixed
    classes = [
        "Formula" if cls == "Inline-formula" else cls for cls in classes
    ]
    assert "Page-number" not in classes

    return classes, bboxes, texts

def transform_bbox_to_original(bbox, original_width, original_height, target_w=1648, target_h=2048):
    # Replicate exact resize logic
    aspect_ratio = original_width / original_height
    new_height = original_height
    new_width = original_width
    
    if original_height > target_h:
        new_height = target_h
        new_width = int(new_height * aspect_ratio)
   
    if new_width > target_w:
        new_width = target_w
        new_height = int(new_width / aspect_ratio)
    
    resized_width = new_width
    resized_height = new_height
    
    # Calculate padding
    pad_left = (target_w - resized_width) // 2
    pad_top = (target_h - resized_height) // 2
    
#     # Transform: use the ACTUAL resized dimensions, not the scale
#     # X coords
    left = ((bbox[0] * target_w) - pad_left) * original_width / resized_width
    right = ((bbox[2] * target_w) - pad_left) * original_width / resized_width
    
#     # Y coords - using original_height / resized_height directly
    top = ((bbox[1] * target_h) - pad_top) * original_height / resized_height
    bottom = ((bbox[3] * target_h) - pad_top) * original_height / resized_height
    
    return left, top, right, bottom

def postprocess_text(text, cls = 'Text', text_format='markdown', table_format='latex', blank_text_in_figures=False):
    assert text_format in ['markdown', 'plain'], 'Unknown text format. Supported: markdown | plain'
    assert table_format in ['latex', 'HTML', 'markdown'], 'Unknown table format. Supported: latex | HTML | markdown'
    if cls != 'Table':
        if text_format == 'plain':
            text = convert_mmd_to_plain_text_ours(text)
    elif table_format == 'HTML':
        text = latex_table_to_html(text)
    elif table_format == 'markdown':
        text = convert_html_tables_to_markdown(latex_table_to_html(text))
    if blank_text_in_figures and cls == 'Picture':
        text = ''
    return text

def remove_nemotron_formatting(text):
   text = text.replace('<tbc>', '')
   text = text.replace('\\<|unk|\\>', '')
   text = text.replace('\\unknown', '')
   return text
def convert_mmd_to_plain_text_ours(mmd_text):
    mmd_text = re.sub(r'<sup>(.*?)</sup>', r'^{\\1}', mmd_text, flags=re.DOTALL)
    mmd_text = re.sub(r'<sub>(.*?)</sub>', r'_{\\1}', mmd_text, flags=re.DOTALL)
    mmd_text = mmd_text.replace('<br>', '\n')

    # Remove headers (e.g., ##)
    mmd_text = re.sub(r'#+\s', '', mmd_text)
    
    # Remove bold (e.g., **)
    mmd_text = re.sub(r'\*\*(.*?)\*\*', r'\1', mmd_text)
    #mmd_text = mmd_text.replace("**","")
    # Remove italic (e.g., *)
    mmd_text = re.sub(r'\*(.*?)\*', r'\1', mmd_text)
    # Remove emphasized text formatting (e.g., _)
    mmd_text = re.sub(r'(?<!\w)_([^_]+)_', r'\1', mmd_text)
    
    # Remove formulas inside paragraphs (e.g., \(R_{ij}(P^{a})=0\))
    #mmd_text = re.sub(r'\\\((.*?)\\\)', '', mmd_text)
    
    # Remove asterisk in lists
    #mmd_text = re.sub(r'^\*\s', '', mmd_text, flags=re.MULTILINE)    
    return mmd_text.strip()
