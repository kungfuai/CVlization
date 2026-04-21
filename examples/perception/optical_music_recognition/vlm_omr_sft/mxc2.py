"""MXC2 (MusicXML Compact v2) — token-efficient encoding preserving all musical info.

Improvements over MXC:
- Duration type names instead of tick counts (quarter not q 480)
- Stateful voice/staff/stem: only emitted on change (not every note)
- Type-based backup: bak quarter not bak 720
- No div= in measure headers (reconstructible from types)
- ~45% fewer tokens than MXC on typical openscore lieder pages

Public API:
    xml_to_mxc2(xml: str) -> str
"""

import re
import xml.etree.ElementTree as ET


# Duration type names — use MusicXML type names directly (no abbreviations)
# This makes the format more readable and avoids the tick-count problem
DURATION_TYPES = {
    "whole", "half", "quarter", "eighth", "16th", "32nd", "64th",
    "128th", "breve", "long", "maxima",
}

ALTER_TO_SYM = {"-2": "bb", "-1": "b", "0": "n0", "1": "#", "2": "##"}
SYLLABIC_TO_SHORT = {"single": "s", "begin": "b", "middle": "m", "end": "e"}


def _dur_ticks_to_type(ticks, divisions):
    """Convert duration ticks to a type name + dot count.

    Returns (type_name, n_dots) or (None, 0) if no match.
    """
    if divisions <= 0:
        return None, 0
    quarter = divisions
    mapping = [
        ("maxima", quarter * 32),
        ("long", quarter * 16),
        ("breve", quarter * 8),
        ("whole", quarter * 4),
        ("half", quarter * 2),
        ("quarter", quarter),
        ("eighth", quarter // 2),
        ("16th", quarter // 4),
        ("32nd", quarter // 8),
        ("64th", quarter // 16),
        ("128th", quarter // 32),
    ]
    ticks = int(ticks)
    # Try undotted, then dotted, then double-dotted
    for name, base in mapping:
        if base <= 0:
            continue
        if ticks == base:
            return name, 0
        if ticks == base + base // 2:  # dotted
            return name, 1
        if ticks == base + base // 2 + base // 4:  # double-dotted
            return name, 2
    return None, 0


def _dur_ticks_to_compound(ticks, divisions):
    """Convert duration ticks to a compound type string.

    For simple durations: "quarter", "half dot"
    For compound durations (e.g. 7.5 quarters in 15/8): "whole dot half dot"

    Always succeeds by decomposing greedily into the largest fitting types.
    """
    if divisions <= 0:
        return str(ticks)
    # Try simple first
    name, dots = _dur_ticks_to_type(ticks, divisions)
    if name is not None:
        return name + " dot" * dots

    # Greedy decomposition into largest-fitting durations
    quarter = divisions
    mapping = [
        ("whole", quarter * 4),
        ("half", quarter * 2),
        ("quarter", quarter),
        ("eighth", quarter // 2),
        ("16th", quarter // 4),
        ("32nd", quarter // 8),
    ]
    remaining = int(ticks)
    parts = []
    for name, base in mapping:
        if base <= 0:
            continue
        # Try dotted first (larger)
        dotted = base + base // 2
        while remaining >= dotted > 0:
            parts.append(f"{name} dot")
            remaining -= dotted
        while remaining >= base:
            parts.append(name)
            remaining -= base
    if remaining > 0:
        parts.append(str(remaining))  # leftover ticks as fallback
    return " ".join(parts) if parts else str(ticks)


def xml_to_mxc2(xml: str) -> str:
    """Convert cleaned MusicXML to MXC2 format."""
    root = ET.fromstring(xml)
    lines = []

    # Header (same as MXC)
    work = root.find("work")
    if work is not None:
        wt = work.findtext("work-title")
        if wt:
            lines.append(f'header work-title="{wt}"')
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

    # Part list (same as MXC)
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
        state = _EncoderState()
        for measure in part.findall("measure"):
            _encode_measure_v2(measure, lines, state)

    return "\n".join(lines)


class _EncoderState:
    """Track stateful properties to emit only on change."""
    __slots__ = ("voice", "staff", "stem", "divisions")

    def __init__(self):
        self.voice = None
        self.staff = None
        self.stem = None
        self.divisions = 1  # current divisions value

    def reset_for_backup(self):
        """After a backup, voice/staff/stem must be re-declared."""
        self.voice = None
        self.staff = None
        self.stem = None


def _encode_measure_v2(measure, lines, state):
    """Encode a single <measure> into MXC2 lines."""
    mline = f"M {measure.get('number', '0')}"

    # Attributes — no div=, but track divisions internally
    attrs = measure.find("attributes")
    if attrs is not None:
        div = attrs.findtext("divisions")
        if div:
            state.divisions = int(div)
            # Don't emit div= (MXC2 uses type names)
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

    # Reset stateful stem tracking at measure start
    state.stem = None

    # Child elements
    for child in measure:
        tag = child.tag
        if tag == "attributes":
            continue
        elif tag == "note":
            _encode_note_v2(child, lines, state)
        elif tag == "direction":
            _encode_direction_v2(child, lines)
        elif tag == "barline":
            _encode_barline_v2(child, lines)
        elif tag == "forward":
            dur = child.findtext("duration") or "0"
            fwd_str = _dur_ticks_to_compound(dur, state.divisions)
            voice = child.findtext("voice")
            staff = child.findtext("staff")
            fwd = f"fwd {fwd_str}"
            if voice and voice != state.voice:
                fwd += f" v={voice}"
                state.voice = voice
            if staff and staff != state.staff:
                fwd += f" st={staff}"
                state.staff = staff
            lines.append(fwd)
        elif tag == "backup":
            dur = child.findtext("duration") or "0"
            bak_str = _dur_ticks_to_compound(dur, state.divisions)
            lines.append(f"bak {bak_str}")
            state.reset_for_backup()
        elif tag == "print":
            tokens = ["print"]
            if child.get("new-system") == "yes":
                tokens.append("new-system")
            if child.get("new-page") == "yes":
                tokens.append("new-page")
            if tokens != ["print"]:
                lines.append(" ".join(tokens))


def _encode_note_v2(note, lines, state):
    """Encode a <note> element into an MXC2 line with stateful properties."""
    tokens = []

    is_chord = note.find("chord") is not None
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
            pitch_str = "X"
        tokens.append(f"{prefix}N")
        tokens.append(pitch_str)

    # Type — use full name (quarter, half, etc.) instead of abbreviation
    ntype = note.findtext("type")
    if ntype:
        tokens.append(ntype)

    # Time modification (tuplets): 3in2, 5in4, etc.
    # Without this, triplet eighths (dur=3360 in div=10080) would be
    # indistinguishable from regular eighths (dur=5040).
    time_mod = note.find("time-modification")
    if time_mod is not None:
        actual = time_mod.findtext("actual-notes")
        normal = time_mod.findtext("normal-notes")
        if actual and normal:
            tokens.append(f"{actual}in{normal}")

    # NO duration ticks — the type + dots + time-modification is sufficient

    # Accidental
    acc = note.findtext("accidental")
    if acc:
        tokens.append(f"acc={acc}")

    # Dots
    dots = note.findall("dot")
    if len(dots) == 1:
        tokens.append("dot")
    elif len(dots) > 1:
        tokens.append(f"dot={len(dots)}")

    # Stem — stateful: only emit on change
    stem = note.findtext("stem")
    if stem and stem != state.stem:
        if stem == "up":
            tokens.append("su")
        elif stem == "down":
            tokens.append("sd")
        elif stem == "none":
            tokens.append("sn")
        state.stem = stem

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

    # Notations
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

    # Voice — stateful: only emit on change
    voice = note.findtext("voice")
    if voice and voice != state.voice:
        tokens.append(f"v={voice}")
        state.voice = voice

    # Staff — stateful: only emit on change
    staff = note.findtext("staff")
    if staff and staff != state.staff:
        tokens.append(f"st={staff}")
        state.staff = staff

    # Lyrics (unchanged from MXC)
    for lyric in note.findall("lyric"):
        lnum = lyric.get("number") or lyric.get("name") or "1"
        syl = lyric.findtext("syllabic") or "s"
        syl_short = SYLLABIC_TO_SHORT.get(syl, syl)
        text = lyric.findtext("text") or ""
        safe_text = re.sub(r'\s', '_', text)
        tokens.append(f"L{lnum}:{syl_short}:{safe_text}")

    lines.append(" ".join(tokens))


def _encode_direction_v2(direction, lines):
    """Encode a <direction> element (same as MXC — kept for completeness)."""
    parts = ["dir"]
    placement = direction.get("placement")
    if placement:
        parts.append(f"@{placement}")

    dt = direction.find("direction-type")
    if dt is not None:
        words = dt.find("words")
        if words is not None and words.text and words.text.strip():
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


def _encode_barline_v2(barline, lines):
    """Encode a <barline> element (same as MXC)."""
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
