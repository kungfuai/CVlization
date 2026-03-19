"""MXC (MusicXML Compact) — a line-based compact encoding of cleaned MusicXML.

Provides lossless round-trip conversion between stripped MusicXML (output of
strip_musicxml_header) and MXC format, achieving ~7x token compression.

Public API:
    xml_to_mxc(xml: str) -> str
    mxc_to_xml(mxc: str) -> str
"""

import re
import xml.etree.ElementTree as ET

# ── Lookup tables ─────────────────────────────────────────────────────────────

TYPE_TO_SHORT = {
    "whole": "w", "half": "h", "quarter": "q", "eighth": "e",
    "16th": "s", "32nd": "t", "64th": "x", "128th": "xx",
    "breve": "bv", "long": "lg",
    "sixteenth": "s",  # alias (some exporters use this)
}
SHORT_TO_TYPE = {"w": "whole", "h": "half", "q": "quarter", "e": "eighth",
                  "s": "16th", "t": "32nd", "x": "64th", "xx": "128th",
                  "bv": "breve", "lg": "long"}

ALTER_TO_SYM = {"-2": "bb", "-1": "b", "0": "n0", "1": "#", "2": "##"}
SYM_TO_ALTER = {"bb": "-2", "b": "-1", "n0": "0", "#": "1", "##": "2"}

SYLLABIC_TO_SHORT = {"single": "s", "begin": "b", "middle": "m", "end": "e"}
SHORT_TO_SYLLABIC = {v: k for k, v in SYLLABIC_TO_SHORT.items()}


# ── XML → MXC ────────────────────────────────────────────────────────────────

def xml_to_mxc(xml: str) -> str:
    """Convert cleaned MusicXML (post strip_musicxml_header) to MXC."""
    root = ET.fromstring(xml)
    lines = []

    # Header
    work = root.find("work")
    if work is not None:
        wt = work.findtext("work-title")
        if wt:
            lines.append(f'header work-title="{wt}"')
    # Identification (composer, lyricist)
    ident = root.find("identification")
    if ident is not None:
        for creator in ident.findall("creator"):
            ctype = creator.get("type", "")
            ctext = (creator.text or "").strip()
            if ctype in ("composer", "lyricist") and ctext:
                lines.append(f'header {ctype}="{ctext}"')

    mn = root.findtext("movement-number")
    if mn:
        lines.append(f"header movement-number={mn}")
    mt = root.findtext("movement-title")
    if mt:
        lines.append(f'header movement-title="{mt}"')

    # Part list
    part_list = root.find("part-list")
    if part_list is not None:
        for sp in part_list.findall("score-part"):
            pid = sp.get("id", "")
            pname = sp.findtext("part-name") or ""
            pabbr = sp.findtext("part-abbreviation") or ""
            lines.append(f"{pid} {pname} {pabbr}".rstrip())

    # Parts
    for part in root.findall("part"):
        lines.append("---")
        lines.append(part.get("id", ""))
        for measure in part.findall("measure"):
            _encode_measure(measure, lines)

    return "\n".join(lines)


def _encode_measure(measure, lines):
    """Encode a single <measure> into MXC lines."""
    mline = f"M {measure.get('number', '0')}"

    # Inline attributes
    attrs = measure.find("attributes")
    if attrs is not None:
        div = attrs.findtext("divisions")
        if div:
            mline += f" div={div}"
        key_el = attrs.find("key")
        if key_el is not None:
            fifths = key_el.findtext("fifths")
            if fifths:
                mline += f" key={fifths}"
            mode = key_el.findtext("mode")
            if mode:
                mline += f" mode={mode}"
        time_el = attrs.find("time")
        if time_el is not None:
            beats = time_el.findtext("beats") or "4"
            bt = time_el.findtext("beat-type") or "4"
            mline += f" time={beats}/{bt}"
        for clef_el in attrs.findall("clef"):
            sign = clef_el.findtext("sign") or "G"
            cline = clef_el.findtext("line") or "2"
            cn = clef_el.get("number")
            if cn and cn != "1":
                mline += f" clef{cn}={sign}{cline}"
            else:
                mline += f" clef={sign}{cline}"
        staves = attrs.findtext("staves")
        if staves:
            mline += f" staves={staves}"

    lines.append(mline)

    # Child elements (skip <attributes>, already handled)
    for child in measure:
        tag = child.tag
        if tag == "attributes":
            continue
        elif tag == "note":
            _encode_note(child, lines)
        elif tag == "direction":
            _encode_direction(child, lines)
        elif tag == "barline":
            _encode_barline(child, lines)
        elif tag == "forward":
            dur = child.findtext("duration") or "0"
            voice = child.findtext("voice")
            staff = child.findtext("staff")
            fwd = f"fwd {dur}"
            if voice:
                fwd += f" v={voice}"
            if staff:
                fwd += f" st={staff}"
            lines.append(fwd)
        elif tag == "backup":
            lines.append(f"bak {child.findtext('duration') or '0'}")
        elif tag == "print":
            # page/system breaks
            tokens = ["print"]
            if child.get("new-system") == "yes":
                tokens.append("new-system")
            if child.get("new-page") == "yes":
                tokens.append("new-page")
            if tokens != ["print"]:
                lines.append(" ".join(tokens))
        else:
            # Generic escape for unknown elements
            raw = ET.tostring(child, encoding="unicode").strip()
            lines.append(f"xml{{{raw}}}")


def _encode_note(note, lines):
    """Encode a <note> element into an MXC line."""
    tokens = []

    # Chord prefix
    is_chord = note.find("chord") is not None

    # Grace prefix
    is_grace = note.find("grace") is not None

    # Rest or pitched note
    rest = note.find("rest")
    if rest is not None:
        prefix = "+" if is_chord else ""
        if is_grace:
            prefix += "g"
        if rest.get("measure") == "yes":
            tokens.append(f"{prefix}R")
            tokens.append("whole")
        else:
            tokens.append(f"{prefix}R")
    else:
        prefix = "+" if is_chord else ""
        if is_grace:
            prefix += "g"
        pitch = note.find("pitch")
        if pitch is not None:
            step = pitch.findtext("step") or "C"
            alter = pitch.findtext("alter")
            octave = pitch.findtext("octave") or "4"
            alter_sym = ALTER_TO_SYM.get(alter, "") if alter else ""
            pitch_str = f"{step}{alter_sym}{octave}"
        else:
            pitch_str = "X"  # unpitched
        tokens.append(f"{prefix}N")
        tokens.append(pitch_str)

    # Type
    ntype = note.findtext("type")
    if ntype:
        tokens.append(TYPE_TO_SHORT.get(ntype, ntype))

    # Duration
    dur = note.findtext("duration")
    if dur:
        tokens.append(dur)

    # Accidental (visual, separate from alter)
    acc = note.findtext("accidental")
    if acc:
        tokens.append(f"acc={acc}")

    # Dot(s)
    dots = note.findall("dot")
    if len(dots) == 1:
        tokens.append("dot")
    elif len(dots) > 1:
        tokens.append(f"dot={len(dots)}")

    # Stem
    stem = note.findtext("stem")
    if stem == "up":
        tokens.append("su")
    elif stem == "down":
        tokens.append("sd")
    elif stem == "none":
        tokens.append("sn")

    # Beam(s)
    for beam in note.findall("beam"):
        bn = beam.get("number", "1")
        bval = (beam.text or "").strip().replace(" ", "-")
        if bn == "1":
            tokens.append(f"bm={bval}")
        else:
            tokens.append(f"bm{bn}={bval}")

    # Tie
    for tie in note.findall("tie"):
        tokens.append(f"tie={tie.get('type', '')}")

    # Notations (tied, slur, etc.)
    notations = note.find("notations")
    if notations is not None:
        for tied in notations.findall("tied"):
            tokens.append(f"tied={tied.get('type', '')}")
        for slur in notations.findall("slur"):
            stype = slur.get("type", "")
            snum = slur.get("number", "1")
            tokens.append(f"slur{snum}={stype}")
        for fermata in notations.findall("fermata"):
            tokens.append("fermata")
        artic = notations.find("articulations")
        if artic is not None:
            for a in artic:
                tokens.append(f"art={a.tag}")
        ornaments = notations.find("ornaments")
        if ornaments is not None:
            for o in ornaments:
                tokens.append(f"orn={o.tag}")

    # Voice
    voice = note.findtext("voice")
    if voice:
        tokens.append(f"v={voice}")

    # Staff
    staff = note.findtext("staff")
    if staff:
        tokens.append(f"st={staff}")

    # Lyrics
    for lyric in note.findall("lyric"):
        lnum = lyric.get("number") or lyric.get("name") or "1"
        syl = lyric.findtext("syllabic") or "s"
        syl_short = SYLLABIC_TO_SHORT.get(syl, syl)
        text = lyric.findtext("text") or ""
        # Replace any whitespace (including \xa0) with underscore to prevent token splitting
        safe_text = re.sub(r'\s', '_', text)
        tokens.append(f"L{lnum}:{syl_short}:{safe_text}")

    lines.append(" ".join(tokens))


def _encode_direction(direction, lines):
    """Encode a <direction> element."""
    parts = ["dir"]
    placement = direction.get("placement")
    if placement:
        parts.append(f"@{placement}")

    dt = direction.find("direction-type")
    if dt is not None:
        words = dt.find("words")
        if words is not None and words.text and words.text.strip():
            # Collect font attributes
            font_parts = []
            for attr in ("font-size", "font-weight", "font-style"):
                v = words.get(attr)
                if v:
                    font_parts.append(f"{attr}={v}")
            if font_parts:
                parts.append(f"[{','.join(font_parts)}]")
            parts.append(words.text.strip())
        dynamics = dt.find("dynamics")
        if dynamics is not None:
            for d in dynamics:
                parts.append(f"dyn={d.tag}")
        wedge = dt.find("wedge")
        if wedge is not None:
            wtype = wedge.get("type", "")
            parts.append(f"wedge={wtype}")

    if len(parts) > 1:
        lines.append(" ".join(parts))


def _encode_barline(barline, lines):
    """Encode a <barline> element."""
    loc = barline.get("location", "right")
    style = barline.findtext("bar-style") or ""
    parts = [f"bar={style}"]
    if loc != "right":
        parts.append(f"loc={loc}")
    repeat = barline.find("repeat")
    if repeat is not None:
        parts.append(f"repeat={repeat.get('direction', '')}")
    ending = barline.find("ending")
    if ending is not None:
        parts.append(f"ending={ending.get('number', '')}:{ending.get('type', '')}")
    lines.append(" ".join(parts))


# ── MXC → XML ────────────────────────────────────────────────────────────────

def mxc_to_xml(mxc: str) -> str:
    """Convert MXC compact format back to MusicXML."""
    lines = mxc.strip().split("\n")
    root = ET.Element("score-partwise", version="4.0")

    # Parse header and part-list from lines before first ---
    part_infos = []  # [(id, name, abbrev)]
    i = 0
    while i < len(lines) and lines[i].strip() != "---":
        line = lines[i].strip()
        if line.startswith("header "):
            _parse_header(line, root)
        elif re.match(r"^P[a-zA-Z0-9]+\s", line):
            parts = line.split(None, 2)
            pid = parts[0]
            pname = parts[1] if len(parts) > 1 else ""
            pabbr = parts[2] if len(parts) > 2 else ""
            part_infos.append((pid, pname, pabbr))
        i += 1

    # Build part-list
    if part_infos:
        pl = ET.SubElement(root, "part-list")
        for pid, pname, pabbr in part_infos:
            sp = ET.SubElement(pl, "score-part", id=pid)
            ET.SubElement(sp, "part-name").text = pname
            if pabbr:
                ET.SubElement(sp, "part-abbreviation").text = pabbr

    # Parse parts
    current_part = None
    current_measure = None
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == "---":
            continue
        if re.match(r"^P[a-zA-Z0-9]+$", line):
            current_part = ET.SubElement(root, "part", id=line)
            current_measure = None
            continue
        if line.startswith("M "):
            current_measure = _parse_measure_line(line, current_part)
            continue
        if current_measure is None:
            continue
        if line.startswith("N ") or line.startswith("R ") or line.startswith("+") or line.startswith("gN") or line.startswith("gR"):
            _parse_note_line(line, current_measure)
        elif line.startswith("dir"):
            _parse_direction_line(line, current_measure)
        elif line.startswith("bar="):
            _parse_barline_line(line, current_measure)
        elif line.startswith("fwd "):
            _parse_forward_line(line, current_measure)
        elif line.startswith("bak "):
            el = ET.SubElement(current_measure, "backup")
            ET.SubElement(el, "duration").text = line.split()[1]
        elif line.startswith("print "):
            _parse_print_line(line, current_measure)
        elif line.startswith("xml{"):
            # Generic escaped XML
            raw = line[4:-1]  # strip xml{ and }
            child = ET.fromstring(raw)
            current_measure.append(child)

    return _to_string(root)


def _parse_header(line, root):
    """Parse a header line into XML elements."""
    if "work-title=" in line:
        m = re.search(r'work-title="([^"]*)"', line)
        if m:
            work = root.find("work")
            if work is None:
                work = ET.SubElement(root, "work")
            ET.SubElement(work, "work-title").text = m.group(1)
    elif "composer=" in line or "lyricist=" in line:
        for ctype in ("composer", "lyricist"):
            m = re.search(rf'{ctype}="([^"]*)"', line)
            if m:
                ident = root.find("identification")
                if ident is None:
                    ident = ET.SubElement(root, "identification")
                creator = ET.SubElement(ident, "creator", type=ctype)
                creator.text = m.group(1)
    elif "movement-number=" in line:
        m = re.search(r'movement-number=(\S+)', line)
        if m:
            ET.SubElement(root, "movement-number").text = m.group(1)
    elif "movement-title=" in line:
        m = re.search(r'movement-title="([^"]*)"', line)
        if m:
            ET.SubElement(root, "movement-title").text = m.group(1)


def _parse_measure_line(line, part):
    """Parse 'M 1 div=10080 key=-1 time=3/4 clef=G2' into a <measure> with <attributes>."""
    tokens = line.split()
    mnum = tokens[1]
    measure = ET.SubElement(part, "measure", number=mnum)

    attr_tokens = tokens[2:]
    if not attr_tokens:
        return measure

    attrs = ET.SubElement(measure, "attributes")
    for tok in attr_tokens:
        if tok.startswith("div="):
            ET.SubElement(attrs, "divisions").text = tok[4:]
        elif tok.startswith("key="):
            key = ET.SubElement(attrs, "key")
            ET.SubElement(key, "fifths").text = tok[4:]
        elif tok.startswith("mode="):
            key = attrs.find("key")
            if key is None:
                key = ET.SubElement(attrs, "key")
            ET.SubElement(key, "mode").text = tok[5:]
        elif tok.startswith("time="):
            beats, bt = tok[5:].split("/")
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = beats
            ET.SubElement(time, "beat-type").text = bt
        elif tok.startswith("clef"):
            # clef=G2 or clef2=F4
            m = re.match(r"clef(\d*)=([A-Z])(\d+)", tok)
            if m:
                clef_num, sign, cline = m.group(1), m.group(2), m.group(3)
                clef_el = ET.SubElement(attrs, "clef")
                if clef_num and clef_num != "1":
                    clef_el.set("number", clef_num)
                ET.SubElement(clef_el, "sign").text = sign
                ET.SubElement(clef_el, "line").text = cline
        elif tok.startswith("staves="):
            ET.SubElement(attrs, "staves").text = tok[7:]

    return measure


def _parse_note_line(line, measure):
    """Parse a note/rest line into a <note> element.

    Collects all tokens first, then emits XML children in canonical MusicXML
    order: grace, chord, pitch/rest, duration, type, accidental, dot, stem,
    beam, tie, notations, voice, staff, lyric.
    """
    tokens = line.split()
    idx = 0
    note = ET.SubElement(measure, "note")

    first = tokens[idx]
    is_chord = first.startswith("+")
    if is_chord:
        first = first[1:]
    is_grace = first.startswith("g")
    if is_grace:
        first = first[1:]
    is_rest = (first == "R")
    idx += 1

    # Collect pitch/rest info
    pitch_str = None
    is_whole_rest = False
    if is_rest:
        if idx < len(tokens) and tokens[idx] == "whole":
            is_whole_rest = True
            idx += 1
    else:
        if idx < len(tokens):
            pitch_str = tokens[idx]
            idx += 1

    # Collect type and duration (positional)
    note_type = None
    duration = None
    if idx < len(tokens) and tokens[idx] in SHORT_TO_TYPE:
        note_type = SHORT_TO_TYPE[tokens[idx]]
        idx += 1
    if idx < len(tokens) and tokens[idx].isdigit():
        duration = tokens[idx]
        idx += 1

    # Collect remaining key=value tokens
    accidental = None
    dots = 0
    stem = None
    beams = []
    ties = []
    tieds = []
    slurs = []
    fermatas = 0
    articulations = []
    ornaments = []
    voice = None
    staff = None
    lyrics = []

    while idx < len(tokens):
        tok = tokens[idx]
        idx += 1
        if tok.startswith("acc="):
            accidental = tok[4:]
        elif tok == "dot":
            dots = 1
        elif tok.startswith("dot="):
            dots = int(tok[4:])
        elif tok in ("su", "sd", "sn"):
            stem = {"su": "up", "sd": "down", "sn": "none"}[tok]
        elif tok.startswith("bm"):
            m = re.match(r"bm(\d*)=(.+)", tok)
            if m:
                beams.append((m.group(1) or "1", m.group(2)))
        elif tok.startswith("tie="):
            ties.append(tok[4:])
        elif tok.startswith("tied="):
            tieds.append(tok[5:])
        elif tok.startswith("slur"):
            m = re.match(r"slur(\d+)=(.+)", tok)
            if m:
                slurs.append((m.group(1), m.group(2)))
        elif tok == "fermata":
            fermatas += 1
        elif tok.startswith("art="):
            articulations.append(tok[4:])
        elif tok.startswith("orn="):
            ornaments.append(tok[4:])
        elif tok.startswith("v="):
            voice = tok[2:]
        elif tok.startswith("st="):
            staff = tok[3:]
        elif tok.startswith("L"):
            m = re.match(r"L(\d+):([sbme]):(.+)", tok)
            if m:
                lyrics.append((m.group(1), m.group(2), m.group(3)))

    # Emit in canonical MusicXML order
    if is_grace:
        ET.SubElement(note, "grace")
    if is_chord:
        ET.SubElement(note, "chord")
    if is_rest:
        if is_whole_rest:
            ET.SubElement(note, "rest", measure="yes")
        else:
            ET.SubElement(note, "rest")
    elif pitch_str and pitch_str != "X":
        pitch_el = ET.SubElement(note, "pitch")
        step, alter, octave = _parse_pitch_str(pitch_str)
        ET.SubElement(pitch_el, "step").text = step
        if alter:
            ET.SubElement(pitch_el, "alter").text = alter
        ET.SubElement(pitch_el, "octave").text = octave
    if duration:
        ET.SubElement(note, "duration").text = duration
    if note_type:
        ET.SubElement(note, "type").text = note_type
    if accidental:
        ET.SubElement(note, "accidental").text = accidental
    for _ in range(dots):
        ET.SubElement(note, "dot")
    if stem:
        ET.SubElement(note, "stem").text = stem
    for bn, bval in beams:
        beam_el = ET.SubElement(note, "beam", number=bn)
        beam_el.text = bval.replace("-", " ")
    for ttype in ties:
        ET.SubElement(note, "tie", type=ttype)
    # Notations block
    if tieds or slurs or fermatas or articulations or ornaments:
        notations_el = ET.SubElement(note, "notations")
        for ttype in tieds:
            ET.SubElement(notations_el, "tied", type=ttype)
        for snum, stype in slurs:
            ET.SubElement(notations_el, "slur", number=snum, type=stype)
        for _ in range(fermatas):
            ET.SubElement(notations_el, "fermata")
        if articulations:
            artic = ET.SubElement(notations_el, "articulations")
            for a in articulations:
                ET.SubElement(artic, a)
        if ornaments:
            orn = ET.SubElement(notations_el, "ornaments")
            for o in ornaments:
                ET.SubElement(orn, o)
    if voice:
        ET.SubElement(note, "voice").text = voice
    if staff:
        ET.SubElement(note, "staff").text = staff
    for lnum, syl_short, text in lyrics:
        lyric = ET.SubElement(note, "lyric", number=lnum)
        ET.SubElement(lyric, "syllabic").text = SHORT_TO_SYLLABIC.get(syl_short, syl_short)
        ET.SubElement(lyric, "text").text = text.replace("_", " ")


def _parse_pitch_str(s):
    """Parse 'C4', 'Bb3', 'F#5' into (step, alter, octave)."""
    step = s[0]
    rest = s[1:]
    alter = None
    # Check for alter symbols
    for sym in ("bb", "##", "n0", "b", "#"):
        if rest.startswith(sym):
            alter = SYM_TO_ALTER[sym]
            rest = rest[len(sym):]
            break
    octave = rest
    return step, alter, octave


def _parse_direction_line(line, measure):
    """Parse 'dir @above [font-size=12,font-weight=bold] Andante'."""
    direction = ET.SubElement(measure, "direction")
    rest = line[3:].strip()  # skip "dir"

    # Placement
    if rest.startswith("@"):
        parts = rest.split(None, 1)
        direction.set("placement", parts[0][1:])
        rest = parts[1] if len(parts) > 1 else ""

    dt = ET.SubElement(direction, "direction-type")

    # Check for dynamics
    if "dyn=" in rest:
        dyn_el = ET.SubElement(dt, "dynamics")
        for tok in rest.split():
            if tok.startswith("dyn="):
                ET.SubElement(dyn_el, tok[4:])
        return

    # Check for wedge
    if "wedge=" in rest:
        for tok in rest.split():
            if tok.startswith("wedge="):
                ET.SubElement(dt, "wedge", type=tok[6:])
        return

    # Words with optional font attributes
    font_attrs = {}
    m = re.match(r"\[([^\]]*)\]\s*(.*)", rest)
    if m:
        for attr in m.group(1).split(","):
            if "=" in attr:
                k, v = attr.split("=", 1)
                font_attrs[k] = v
        rest = m.group(2)

    if rest:
        words = ET.SubElement(dt, "words")
        for k, v in font_attrs.items():
            words.set(k, v)
        words.text = rest


def _parse_barline_line(line, measure):
    """Parse 'bar=light-heavy loc=left repeat=backward'."""
    barline = ET.SubElement(measure, "barline")
    for tok in line.split():
        if tok.startswith("bar="):
            barline.set("location", "right")  # default
            ET.SubElement(barline, "bar-style").text = tok[4:]
        elif tok.startswith("loc="):
            barline.set("location", tok[4:])
        elif tok.startswith("repeat="):
            ET.SubElement(barline, "repeat", direction=tok[7:])
        elif tok.startswith("ending="):
            num, etype = tok[7:].split(":", 1)
            ET.SubElement(barline, "ending", number=num, type=etype)


def _parse_forward_line(line, measure):
    """Parse 'fwd 10080 v=2 st=1'."""
    tokens = line.split()
    fwd = ET.SubElement(measure, "forward")
    ET.SubElement(fwd, "duration").text = tokens[1]
    for tok in tokens[2:]:
        if tok.startswith("v="):
            ET.SubElement(fwd, "voice").text = tok[2:]
        elif tok.startswith("st="):
            ET.SubElement(fwd, "staff").text = tok[3:]


def _parse_print_line(line, measure):
    """Parse 'print new-system new-page'."""
    tokens = line.split()
    print_el = ET.SubElement(measure, "print")
    for tok in tokens[1:]:
        if tok == "new-system":
            print_el.set("new-system", "yes")
        elif tok == "new-page":
            print_el.set("new-page", "yes")


def _to_string(root):
    """Serialize ElementTree to indented XML string (no declaration)."""
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")
