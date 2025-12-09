import re
from bs4 import BeautifulSoup

def skip_whitespace(text, i):
    """Advance index i past any whitespace."""
    while i < len(text) and text[i].isspace():
        i += 1
    return i

def parse_braced_argument(text, i):
    """
    Given text and an index i that should point at an opening '{',
    return a tuple (argument_content, new_index) where argument_content is the full
    string inside the balanced braces and new_index is the position just after the matching '}'.
    """
    if i >= len(text) or text[i] != '{':
        raise ValueError("Expected '{' at position {}".format(i))
    i += 1  # skip the opening brace
    start = i
    level = 1
    while i < len(text) and level > 0:
        if text[i] == '{':
            level += 1
        elif text[i] == '}':
            level -= 1
        i += 1
    if level != 0:
        raise ValueError("Unbalanced braces starting at position {}".format(start-1))
    # The argument content is from start to i-1 (excluding the closing brace)
    return text[start:i-1], i

def parse_command(text, i):
    """
    Parse a \multirow or \multicolumn command starting at index i.
    This function assumes the command has exactly three braced arguments.

    It processes each argument recursively. For the third argument, after recursive processing,
    it replaces any unescaped & with \&.

    Returns a tuple (command_text, new_index) where command_text is the reconstructed command.
    """
    # Determine which command we have.
    if text.startswith(r"\multirow", i):
        command_name = r"\multirow"
        i += len(r"\multirow")
    elif text.startswith(r"\multicolumn", i):
        command_name = r"\multicolumn"
        i += len(r"\multicolumn")
    else:
        raise ValueError("Expected \\multirow or \\multicolumn at position {}".format(i))

    # Skip whitespace between the command name and the first argument.
    i = skip_whitespace(text, i)
    args = []
    # Expect exactly three arguments
    for arg_index in range(3):
        if i >= len(text) or text[i] != '{':
            raise ValueError("Expected '{' for argument {} at position {}".format(arg_index+1, i))
        arg_content, i = parse_braced_argument(text, i)
        # Process the content recursively to catch nested commands
        processed_arg = clean_multi_cells(arg_content)
        if arg_index == 2:
            # For the cell text (third argument), replace any unescaped &
            processed_arg = re.sub(r'(?<!\\)&', r'\\&', processed_arg)
        args.append(processed_arg)
        # Only skip whitespace between arguments, not after the last one.
        if arg_index < 2:
            i = skip_whitespace(text, i)
    # Reconstruct the full command with its three arguments
    command_text = f"{command_name}{{{args[0]}}}{{{args[1]}}}{{{args[2]}}}"
    return command_text, i

def clean_multi_cells(text):
    """
    Process an arbitrary LaTeX text string and look for occurrences of \multirow or \multicolumn commands.
    When found, the command is parsed (handling nested braces and nested commands) and its third argument is fixed.

    Returns the processed text.
    """
    result = []
    i = 0
    while i < len(text):
        # Find next occurrence of either command.
        idx_multi = text.find(r"\multirow", i)
        idx_multiC = text.find(r"\multicolumn", i)

        # Determine the next index among the two (if any)
        if idx_multi == -1 and idx_multiC == -1:
            result.append(text[i:])
            break
        if idx_multi == -1:
            next_idx = idx_multiC
        elif idx_multiC == -1:
            next_idx = idx_multi
        else:
            next_idx = min(idx_multi, idx_multiC)

        # Append text before the command (preserving any whitespace)
        result.append(text[i:next_idx])
        # Process the command starting at next_idx
        command_text, new_index = parse_command(text, next_idx)
        result.append(command_text)
        i = new_index
    return ''.join(result)

def parse_brace(s, pos):
    """
    Given a string s and an index pos pointing to an opening '{',
    returns a tuple (content, new_pos) where content is the string
    between the matching braces (handling nested braces) and new_pos is
    the index just after the closing '}'.
    """
    if pos >= len(s) or s[pos] != '{':
        raise ValueError("Expected '{' at position %d" % pos)
    pos += 1  # skip the opening brace
    content = ""
    depth = 1
    while pos < len(s) and depth:
        char = s[pos]
        if char == '{':
            depth += 1
            content += char
        elif char == '}':
            depth -= 1
            if depth:
                content += char
        else:
            content += char
        pos += 1
    if depth != 0:
        raise ValueError("Unmatched '{' in string.")
    return content, pos

def parse_command_merge(s, pos):
    """
    Parse a multirow or multicolumn command starting at s[pos]. If the content
    of the command contains a nested command, then recursively parse the inner
    command and merge its parameters with the outer ones. The merging is done
    so that the outer multirow’s parameters (e.g. rowspan and width) are kept
    while the inner command’s parameters (e.g. colspan, alignment) and its innermost
    content are returned.

    Returns a tuple (merged_dict, new_pos) where merged_dict is a dictionary
    containing the combined parameters and new_pos is the updated index after
    parsing the command.
    """
    if s.startswith(r"\multirow", pos):
        newpos = pos + len(r"\multirow")
        # Parse the three required arguments for multirow: rowspan, width, and content.
        rowspan, newpos = parse_brace(s, newpos)
        width, newpos = parse_brace(s, newpos)
        content, newpos = parse_brace(s, newpos)
        # Look for a nested command (either \multirow or \multicolumn) in the content.
        index_mr = content.find(r"\multirow")
        index_mc = content.find(r"\multicolumn")
        if index_mr == -1 and index_mc == -1:
            # No nested command found; return this command’s details.
            return {"rowspan": rowspan.strip(), "width": width.strip(), "content": content.strip()}, newpos
        else:
            # At least one nested command is present. Pick the first occurrence.
            indices = [i for i in (index_mr, index_mc) if i != -1]
            first_index = min(indices)
            # Parse the inner (nested) command from within the content.
            inner, _ = parse_command_merge(content, first_index)
            # Merge: keep the outer multirow’s parameters and add the inner ones.
            merged = {"rowspan": rowspan.strip(), "width": width.strip()}
            merged.update(inner)
            return merged, newpos

    elif s.startswith(r"\multicolumn", pos):
        newpos = pos + len(r"\multicolumn")
        # Parse the three arguments for multicolumn: colspan, alignment, and content.
        colspan, newpos = parse_brace(s, newpos)
        alignment, newpos = parse_brace(s, newpos)
        content, newpos = parse_brace(s, newpos)
        # Look for a nested command in the content.
        index_mr = content.find(r"\multirow")
        index_mc = content.find(r"\multicolumn")
        if index_mr == -1 and index_mc == -1:
            return {"colspan": colspan.strip(), "alignment": alignment.strip(), "content": content.strip()}, newpos
        else:
            indices = [i for i in (index_mr, index_mc) if i != -1]
            first_index = min(indices)
            inner, _ = parse_command_merge(content, first_index)
            merged = {"colspan": colspan.strip(), "alignment": alignment.strip()}
            merged.update(inner)
            return merged, newpos

    # Not a recognized command starting at pos.
    return None, pos

def extract_merged_commands(s):
    """
    Scan through the LaTeX string s and extract merged multirow/multicolumn commands.
    For each command found, if there is nesting the parser merges the outer and inner
    parameters so that the final result includes both the rowspan (or width) and the colspan
    (or alignment) along with the innermost content.

    Returns a list of dictionaries.
    """
    pos = 0
    results = []
    while pos < len(s):
        if s[pos] == '\\':
            res, newpos = parse_command_merge(s, pos)
            if res is not None:
                results.append(res)
                pos = newpos
                continue
        pos += 1
    return results

def remove_tags(html, tags_to_remove):
    soup = BeautifulSoup(html, "html.parser")
    # Loop through the tags to remove
    for tag_name in tags_to_remove:
        for tag in soup.find_all(tag_name):
            # Move the children of the tag to the parent tag
            tag.unwrap()  # This removes the tag but keeps its contents
    # Return the modified HTML as a string
    return str(soup)

def convert_th_to_td(html):
    """Replace all th tags with td tags
    """
    soup = BeautifulSoup(html)
    for th_tag in soup.find_all('th'):
        th_tag.name = 'td'
    return str(soup)

def replace_italic(text):
    pattern = re.compile(r'(?<!\\)_(.*?)(?<!\\)_')

    def italic_replacer(match):
        # Get the text inside the underscores.
        content = match.group(1)
        # Remove the escape (backslash) from any escaped underscores inside.
        content = content.replace(r'\_', '_')
        return f"<i>{content}</i>"

    # Replace all occurrences of the pattern using the replacer function.
    return pattern.sub(italic_replacer, text)


def replace_bold(text):
    pattern = re.compile(r'(?<!\\)\*\*(.*?)(?<!\\)\*\*')

    def bold_replacer(match):
        content = match.group(1)
        # Unescape any escaped asterisks within the captured text.
        content = content.replace(r'\*', '*')
        return f"<b>{content}</b>"

    return pattern.sub(bold_replacer, text)

def latex_table_to_html(latex_str, add_head_body = False):
    # Pattern to match the entire tabular environment
    table_pattern = r'\\begin{tabular}{([^}]*)}\s*(.*?)\\end{tabular}'

    def process_cell(cell):
        # Clean up cell content
        cell = cell.strip()

        out = extract_merged_commands(cell)
        if len(out) > 0:
            cell = process_cell(out[0]["content"])["content"]
            rowspan = int(out[0].get("rowspan", "1"))
            colspan = int(out[0].get("colspan", "1"))
            return {
                "content": cell,
                "colspan": colspan,
                "rowspan": rowspan
            }

        # Replace latex and markdown formatting with HTML tags
        cell = re.sub(r'\$([^$]*)\$', r'\1', cell)  # Remove math mode
        cell = re.sub(r'\\textbf{([^}]*)}', r'<b>\1</b>', cell)  # Convert latex bold
        cell = re.sub(r'\\textit{([^}]*)}', r'<i>\1</i>', cell)  # Convert latex italic
        cell = replace_italic(cell)
        cell = replace_bold(cell)
        cell = cell.replace("\\$", "$").replace("\\%", "%").replace("\\newline", "\n").replace("\\textless", "<").replace("\\textgreater", ">").replace("\\*", "*").replace("\\_", "_").replace("\\backslash", "\\")

        # Replace \& with & in the cell text
        cell = cell.replace(r'\&', '&')
        cell = cell.replace('<tbc>', '')
        # Preserve newlines for downstream row-splitting; clean other tokens
        cell = cell.replace('\\unknown', '').replace('\\<|unk|\\>', '').replace('<u>', '<underline>').replace('</u>', '</underline>')
        return {
            'content': cell,
            'colspan': 1,
            'rowspan': 1
        }

    def split_row(input_string):
        # Use a regular expression to split on '&' that is not preceded by a backslash
        return re.split(r'(?<!\\)&', input_string)

    def convert_table(match):
        # Extract table content
        format_spec, content = match.groups()

        # Start building HTML table
        html = ['<table>']

        # Track cells for multirow
        multirow_tracker = set()

        # Process rows
        rows = re.split(r'\\\\', content)
        current_row = 0
        
        for row in rows:
            if not row.strip():
                continue

            row = row.strip()

            # Skip \hline
            if '\\hline' in row:
                row = row.replace('\\hline', '')
                if not row.strip():
                    continue

            row = clean_multi_cells(row)

            # Process cells
            cells = split_row(row)
            processed_cells = [process_cell(cell) for cell in cells]

            # Build per-cell line lists splitting on newline or <br> tokens
            def split_lines(text):
                parts = re.split(r'(?:\n|<br\s*/?>)+', text)
                return parts if parts is not None else ['']

            line_lists = [split_lines(cell['content']) for cell in processed_cells]
            max_lines = max(len(lst) for lst in line_lists) if line_lists else 1

            # Emit one or more rows based on max_lines
            for line_idx in range(max_lines):
                if add_head_body:
                    if current_row == 0:
                        html.append(' <thead>')
                    if current_row == 1:
                        html.append(' <tbody>')
                html.append('  <tr>')
                current_col = 0

                for col_idx, cell in enumerate(processed_cells):
                    content_segment = line_lists[col_idx][line_idx] if line_idx < len(line_lists[col_idx]) else ''

                    attrs = []
                    if cell['colspan'] > 1:
                        attrs.append(f'colspan="{cell["colspan"]}"')
                    # Only apply original rowspan to the first emitted line
                    if cell['rowspan'] > 1 and line_idx == 0:
                        attrs.append(f'rowspan="{cell["rowspan"]}"')
                        for r in range(current_row + 1, current_row + cell['rowspan']):
                            for c in range(current_col, current_col + cell['colspan']):
                                multirow_tracker.add((r, c))

                    # If this position is covered by a prior rowspan, skip rendering a duplicate cell
                    if cell['rowspan'] > 1 and line_idx > 0:
                        current_col += cell['colspan']
                        continue

                    if (current_row, current_col) in multirow_tracker and content_segment == '' and cell["colspan"] == 1 and cell["rowspan"] == 1:
                        current_col += cell['colspan']
                        continue

                    attr_str = ' ' + ' '.join(attrs) if attrs else ''
                    cell_tag = 'td'
                    html.append(f'    <{cell_tag}{attr_str}>{content_segment}</{cell_tag}>')
                    current_col += cell['colspan']

                if add_head_body and current_row == 0:
                    html.append(' </thead>')
                html.append('  </tr>')
                current_row += 1
        if add_head_body:
            html.append(' </tbody>')
        html.append('</table>')
        return '\n'.join(html)

    # Convert all tabular environments in the input
    return re.sub(table_pattern, convert_table, latex_str, flags=re.DOTALL)
def convert_single_table(table):
    """
    Convert a single HTML table to Markdown format.
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        str: Markdown table string
    """
    markdown_lines = []
    rows = table.find_all('tr')
    
    for i, row in enumerate(rows):
        cells = row.find_all(['td', 'th'])
        if not cells:
            continue
        
        # Convert cells to text, handling nested elements
        row_data = []
        for cell in cells:
            # Get text content, handling nested elements
            cell_text = cell.get_text(separator=' ', strip=True)
            # Escape pipe characters
            cell_text = cell_text.replace('|', '\\|')
            row_data.append(cell_text)
        
        # Add row to markdown
        markdown_lines.append('| ' + ' | '.join(row_data) + ' |')
        
        # Add separator after header row
        if i == 0:
            separator = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
            markdown_lines.append(separator)
    
    return '\n'.join(markdown_lines)
def convert_html_tables_to_markdown(html_content):
    """
    Find all HTML tables and convert them to Markdown while preserving all other content.
    
    Args:
        html_content (str): HTML content that may contain tables
        
    Returns:
        str: HTML content with tables converted to Markdown
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all tables
    tables = soup.find_all('table')
    
    if not tables:
        return html_content  # Return original content unchanged
    
    # Convert each table to markdown and replace it
    for table in tables:
        markdown_table = convert_single_table(table)
        
        # Create a new element to replace the table
        replacement = soup.new_string('\n' + markdown_table + '\n')
        table.replace_with(replacement)
    
    return str(soup)
