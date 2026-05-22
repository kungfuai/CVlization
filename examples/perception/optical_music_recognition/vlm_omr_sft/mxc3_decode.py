"""DEPRECATED — see mxc3.py for context. Kept for reference only.

MXC3 → MusicXML decoder. Round-trip oracle for mxc3.py."""

import re
from collections import defaultdict, OrderedDict
from xml.etree.ElementTree import Element, SubElement, tostring
from mxc3 import (
    key_implied_alter, SYM_TO_ALTER, SYM_TO_STEM, SHORT_TO_SYLLABIC,
    ALTER_TO_SYM,
)


DIVISIONS = 96
TYPE_TO_TICKS = {
    "whole": DIVISIONS * 4, "half": DIVISIONS * 2, "quarter": DIVISIONS,
    "eighth": DIVISIONS // 2, "16th": DIVISIONS // 4, "32nd": DIVISIONS // 8,
    "64th": DIVISIONS // 16, "128th": DIVISIONS // 32,
    "breve": DIVISIONS * 8, "long": DIVISIONS * 16, "maxima": DIVISIONS * 32,
}
ACC_LONG = {-2: "double-flat", -1: "flat", 0: "natural",
            1: "sharp", 2: "double-sharp"}


def _unescape(s: str) -> str:
    return s.replace("_", " ")


def _parse_attrs(s: str) -> dict:
    out = {}
    # Match either: key="...quoted-with-escapes..." or key=unquoted-value
    for m in re.finditer(r'(\w[\w.-]*)=(?:"((?:[^"\\]|\\.)*)"|(\S+))', s):
        k, qv, uv = m.group(1), m.group(2), m.group(3)
        if qv is not None:
            qv = qv.replace('\\"', '"').replace('\\\\', '\\')
            out[k] = qv
        else:
            out[k] = uv
    return out


def _parse_dur(tok: str):
    if tok.endswith(".."):
        return tok[:-2], 2
    if tok.endswith("."):
        return tok[:-1], 1
    return tok, 0


def _ticks_of(dur_tok: str) -> int:
    t, dots = _parse_dur(dur_tok)
    if t.startswith("q="):
        q = float(t.split("=", 1)[1])
        return int(q * DIVISIONS)
    if t.startswith("ticks="):
        return int(t.split("=", 1)[1])
    base = TYPE_TO_TICKS.get(t, DIVISIONS)
    total = base
    extra = base // 2
    for _ in range(dots):
        total += extra
        extra //= 2
    return total


def _measure_ticks(beats: int, beat_type: int) -> int:
    """Ticks per measure for given time signature."""
    return int(beats * (4 / beat_type) * DIVISIONS)


def _is_grace(tok: str) -> bool:
    return tok.startswith("g") and not tok.startswith("ga")  # 'g' prefix, not e.g. 'G4'


def _strip_grace(tok: str) -> str:
    return tok[1:] if tok.startswith("g") and (tok[1] == "[" or (len(tok) > 1 and not tok[1].isdigit())) else tok


def _split_pos(tok: str):
    """Returns (kind, list_of_(step,octave))."""
    if tok == "R":
        return "rest", []
    if tok == "Rm":
        return "rest_m", []
    if tok == "F":
        return "forward", []
    grace = False
    if tok.startswith("g") and len(tok) > 1 and (tok[1] == "[" or tok[1].isalpha()) and tok[1] != "R":
        # 'g' followed by chord or step letter
        grace = True
        tok = tok[1:]
    if tok == "R":
        # grace rest
        return ("grace_rest", [])
    if tok.startswith("[") and tok.endswith("]"):
        items = tok[1:-1].split(",")
    else:
        items = [tok]
    out = []
    for it in items:
        m = re.match(r"^([A-G])(-?\d+)$", it)
        if not m:
            raise ValueError(f"bad pos element: {it!r}")
        out.append((m.group(1), m.group(2)))
    return ("grace" if grace else "note", out)


def _parse_overrides_kv(line: str):
    out = []
    for tok in line.split():
        m = re.match(r"^([\d.]+)=(\S+)$", tok)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _parse_acc(line: str):
    out = {}
    for k, v in _parse_overrides_kv(line):
        # k is '3' or '3.2'
        if "." in k:
            i, j = k.split(".")
            out[(int(i), int(j))] = SYM_TO_ALTER.get(v, 0)
        else:
            out[(int(k), None)] = SYM_TO_ALTER.get(v, 0)
    return out


def _parse_indexed_seq(line: str):
    """Parse '<i>=<v>' or '<i>.<j>=<v>' tokens.
    Returns: leader_map {int(i): [v]}, chord_map {(int(i), int(j)): [v]}."""
    leader = defaultdict(list)
    chord = defaultdict(list)
    for k, v in _parse_overrides_kv(line):
        if "." in k:
            i, j = k.split(".")
            chord[(int(i), int(j))].append(v)
        else:
            leader[int(k)].append(v)
    return dict(leader), dict(chord)


def _parse_simple_seq(line: str):
    leader, _ = _parse_indexed_seq(line)
    return {k: v[0] for k, v in leader.items()}


def _parse_multi_seq(line: str):
    leader, _ = _parse_indexed_seq(line)
    return leader


def _parse_chord_aware(line: str):
    """Returns (leader_dict, chord_dict)."""
    return _parse_indexed_seq(line)


def _parse_slur(line: str):
    """Parse '<num>:<i>=<type>' or '<num>:<i>.<j>=<type>' →
    list of (num, beat, chord_idx_or_None, type)."""
    out = []
    for tok in line.split():
        m = re.match(r"^(\d+):(\d+)(?:\.(\d+))?=(\S+)$", tok)
        if m:
            out.append((m.group(1), int(m.group(2)),
                        int(m.group(3)) if m.group(3) is not None else None,
                        m.group(4)))
    return out


def _parse_lyric_line(line: str, n_beats: int):
    """Parse parallel array of '<syl>:<text>' or '--' tokens."""
    toks = line.split()
    result = [None] * n_beats
    for i, t in enumerate(toks):
        if i >= n_beats:
            break
        if t == "--":
            continue
        if ":" in t:
            syl, txt = t.split(":", 1)
            result[i] = (SHORT_TO_SYLLABIC.get(syl, syl), _unescape(txt))
        else:
            result[i] = ("single", _unescape(t))
    return result


def _parse_stem(line: str, n_beats: int):
    toks = line.split()
    return [SYM_TO_STEM.get(t) if t != "." else None
            for t in toks[:n_beats]]


def _parse_dir(line: str):
    """Parse positional direction tokens '<i>:@<place>:<type>=<value>'."""
    out = []
    for tok in line.split():
        m = re.match(r"^(\d+):@([^:]*):(\w+)=(.*)$", tok)
        if m:
            out.append((int(m.group(1)), m.group(2), m.group(3), m.group(4)))
    return out


def _parse_bar(line: str):
    """Parse bar: loc=X style=Y [repeat=Z] [ending=N:T]."""
    attrs = _parse_attrs(line)
    return attrs


def mxc3_to_xml(mxc3: str) -> str:
    """Decode MXC3 → MusicXML string."""
    lines = mxc3.splitlines()
    i = 0
    parts_meta = []
    header_extras = {}
    n_parts = 0

    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("HEADER"):
            rest = line[len("HEADER"):].strip()
            attrs = _parse_attrs(rest)
            n_parts = int(attrs.get("parts", "1"))
            for k in ("work-title", "movement-number", "movement-title",
                       "composer", "lyricist"):
                if k in attrs:
                    header_extras[k] = attrs[k]
            i += 1
        elif line.startswith("P="):
            attrs = _parse_attrs(line)
            pidx = int(attrs["P"])
            parts_meta.append({
                "P": pidx,
                "id": attrs.get("id") or f"P{pidx}",
                "name": attrs.get("name", ""),
                "abbr": attrs.get("abbr", ""),
            })
            i += 1
        elif not line.strip():
            i += 1
        else:
            break

    # Pad parts_meta if short
    while len(parts_meta) < n_parts:
        k = len(parts_meta) + 1
        parts_meta.append({"P": k, "id": f"P{k}", "name": "", "abbr": ""})

    # Parse sub-blocks. The `===` marker separates measure groups, which
    # is critical for measures with duplicate numbers.
    blocks = []
    cur = None
    measure_group_idx = 0   # monotonic counter for measure groups
    while i < len(lines):
        line = lines[i].rstrip()
        if line.strip() == "===":
            measure_group_idx += 1
            i += 1
            continue
        if line.startswith("M="):
            if cur:
                blocks.append(cur)
            attrs = _parse_attrs(line)
            start_q = float(attrs.get("start", "0"))
            # M may be a non-integer string (e.g. "9X1" for alternate endings)
            cur = {
                "M": attrs["M"],
                "P": int(attrs["P"]),
                "S": int(attrs.get("S", "1")),
                "V": int(attrs.get("V", "1")),
                "_group_idx": measure_group_idx,
                "_start_q": start_q,
                "key": int(attrs.get("key", "0")),
                "time": attrs.get("time", "4/4"),
                "clef": attrs.get("clef", "G2"),
                "pos": [], "dur": [], "stem": [],
                "acc": {}, "tie": {}, "tied": {},
                "tie_chord": {}, "tied_chord": {},
                "art_chord": {}, "orn_chord": {},
                "slur": [], "fer": set(),
                "art": {}, "orn": {}, "tup": {},
                "lyrics": {},
                "bars": [], "dirs": [],
            }
        elif cur and line.strip():
            ls = line.lstrip()
            if ls.startswith("pos:"):
                cur["pos"] = ls.split("pos:", 1)[1].split()
            elif ls.startswith("dur:"):
                cur["dur"] = ls.split("dur:", 1)[1].split()
            elif ls.startswith("stem:"):
                cur["stem"] = _parse_stem(ls.split("stem:", 1)[1], len(cur["pos"]))
            elif ls.startswith("acc:"):
                cur["acc"] = _parse_acc(ls.split("acc:", 1)[1])
            elif ls.startswith("tie:"):
                leader, chord = _parse_indexed_seq(ls.split("tie:", 1)[1])
                cur["tie"], cur["tie_chord"] = leader, chord
            elif ls.startswith("tied:"):
                leader, chord = _parse_indexed_seq(ls.split("tied:", 1)[1])
                cur["tied"], cur["tied_chord"] = leader, chord
            elif ls.startswith("slur:"):
                cur["slur"] = _parse_slur(ls.split("slur:", 1)[1])
            elif ls.startswith("fer:"):
                cur["fer"] = set(int(x) for x in ls.split("fer:", 1)[1].split())
            elif ls.startswith("art:"):
                leader, chord = _parse_indexed_seq(ls.split("art:", 1)[1])
                cur["art"] = {k: v[0] for k, v in leader.items()}
                cur["art_chord"] = {k: v[0] for k, v in chord.items()}
            elif ls.startswith("orn:"):
                leader, chord = _parse_indexed_seq(ls.split("orn:", 1)[1])
                cur["orn"] = {k: v[0] for k, v in leader.items()}
                cur["orn_chord"] = {k: v[0] for k, v in chord.items()}
            elif ls.startswith("tup:"):
                cur["tup"] = _parse_simple_seq(ls.split("tup:", 1)[1])
            elif ls.startswith("lyr"):
                m = re.match(r"^lyr(\d+):\s*(.*)$", ls)
                if m:
                    vnum = m.group(1)
                    cur["lyrics"][vnum] = _parse_lyric_line(m.group(2), len(cur["pos"]))
            elif ls.startswith("bar:"):
                cur["bars"].append(_parse_bar(ls.split("bar:", 1)[1]))
            elif ls.startswith("dir:"):
                cur["dirs"] = _parse_dir(ls.split("dir:", 1)[1])
        i += 1
    if cur:
        blocks.append(cur)

    # Group by (group_idx, M, P) — group_idx separates duplicate measure
    # numbers into distinct measures, while sub-blocks of the same
    # measure share the same group_idx.
    by_key = defaultdict(list)
    for b in blocks:
        by_key[(b["_group_idx"], b["M"], b["P"])].append(b)
    for k in by_key:
        by_key[k].sort(key=lambda b: (b["S"], b["V"]))

    measure_order = list(OrderedDict.fromkeys(
        (b["_group_idx"], b["M"], b["P"]) for b in blocks
    ))

    # ── Build XML ───────────────────────────────────────────────────────────
    root = Element("score-partwise", attrib={"version": "4.0"})
    if "work-title" in header_extras:
        w = SubElement(root, "work")
        SubElement(w, "work-title").text = header_extras["work-title"]
    if "movement-number" in header_extras:
        SubElement(root, "movement-number").text = header_extras["movement-number"]
    if "movement-title" in header_extras:
        SubElement(root, "movement-title").text = header_extras["movement-title"]
    if "composer" in header_extras or "lyricist" in header_extras:
        ident = SubElement(root, "identification")
        for k in ("composer", "lyricist"):
            if k in header_extras:
                cr = SubElement(ident, "creator", attrib={"type": k})
                cr.text = header_extras[k]

    pl = SubElement(root, "part-list")
    for pm in parts_meta[:n_parts]:
        sp = SubElement(pl, "score-part", attrib={"id": pm["id"]})
        SubElement(sp, "part-name").text = pm["name"]
        if pm["abbr"]:
            SubElement(sp, "part-abbreviation").text = pm["abbr"]

    # Determine per-part multi-voice / multi-staff flags from blocks
    part_flags = {}
    for pm in parts_meta[:n_parts]:
        p_num = pm["P"]
        p_blocks = [b for b in blocks if b["P"] == p_num]
        multi_voice = any(b["V"] != 1 for b in p_blocks) or \
            len({b["V"] for b in p_blocks}) > 1
        multi_staff = any(b["S"] != 1 for b in p_blocks) or \
            len({b["S"] for b in p_blocks}) > 1
        part_flags[p_num] = {"multi_voice": multi_voice, "multi_staff": multi_staff}

    for pm in parts_meta[:n_parts]:
        p_num = pm["P"]
        part_el = SubElement(root, "part", attrib={"id": pm["id"]})
        m_keys = [(gi, m, p) for (gi, m, p) in measure_order if p == p_num]
        prev_attrs = {"key": None, "time": None, "clefs": {}}
        flags = part_flags[p_num]
        for (gi, m, p) in m_keys:
            subs = by_key[(gi, m, p)]
            _emit_measure_with_subs(part_el, m, subs, prev_attrs, flags)

    return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(root, encoding="unicode")


def _emit_measure_with_subs(part_el, m_num, subs, prev_attrs, flags):
    """Emit a <measure> containing one or more (S, V) sub-blocks merged with backup."""
    m_el = SubElement(part_el, "measure", attrib={"number": str(m_num)})  # m_num may be str

    # Determine attributes (need to emit when changed; use first sub-block's)
    first = subs[0]
    # All sub-blocks of the same (M, P) share key/time. Clefs may differ per staff.
    clefs_by_staff = {}
    for s in subs:
        clefs_by_staff[str(s["S"])] = s["clef"]

    need_attrs = (
        prev_attrs["key"] is None
        or first["key"] != prev_attrs["key"]
        or first["time"] != prev_attrs["time"]
        or clefs_by_staff != prev_attrs["clefs"]
    )
    if need_attrs:
        attrs_el = SubElement(m_el, "attributes")
        if prev_attrs["key"] is None:
            SubElement(attrs_el, "divisions").text = str(DIVISIONS)
        if first["key"] != prev_attrs["key"]:
            k_el = SubElement(attrs_el, "key")
            SubElement(k_el, "fifths").text = str(first["key"])
        if first["time"] != prev_attrs["time"]:
            b, bt = first["time"].split("/")
            t_el = SubElement(attrs_el, "time")
            SubElement(t_el, "beats").text = b
            SubElement(t_el, "beat-type").text = bt
        if len(clefs_by_staff) > 1 and prev_attrs.get("staves") != len(clefs_by_staff):
            SubElement(attrs_el, "staves").text = str(len(clefs_by_staff))
            prev_attrs["staves"] = len(clefs_by_staff)
        if clefs_by_staff != prev_attrs["clefs"]:
            for staff_num in sorted(clefs_by_staff, key=int):
                clef_str = clefs_by_staff[staff_num]
                if prev_attrs["clefs"].get(staff_num) == clef_str:
                    continue  # this clef hasn't changed
                ms = re.match(r"^([A-Z])(-?\d+)$", clef_str)
                sign, line = (ms.group(1), ms.group(2)) if ms else ("G", "2")
                attrs_cl = {"number": staff_num} if len(clefs_by_staff) > 1 else {}
                c_el = SubElement(attrs_el, "clef", attrib=attrs_cl)
                SubElement(c_el, "sign").text = sign
                SubElement(c_el, "line").text = line
        prev_attrs["key"] = first["key"]
        prev_attrs["time"] = first["time"]
        prev_attrs["clefs"] = dict(clefs_by_staff)

    # Emit each (S, V) sub-block; insert <backup> between them
    beats_n, beat_type_n = first["time"].split("/")
    measure_ticks = _measure_ticks(int(beats_n), int(beat_type_n))

    # Split barlines by location: left ones come BEFORE notes, right after.
    left_bars = [bl for bl in first["bars"] if bl.get("loc") == "left"]
    right_bars = [bl for bl in first["bars"] if bl.get("loc") != "left"]

    def _emit_barline(bl):
        bl_el = SubElement(m_el, "barline")
        if "loc" in bl:
            bl_el.set("location", bl["loc"])
        if "style" in bl:
            SubElement(bl_el, "bar-style").text = bl["style"]
        if "repeat" in bl:
            SubElement(bl_el, "repeat", attrib={"direction": bl["repeat"]})
        if "ending" in bl:
            num, typ = bl["ending"].split(":", 1)
            SubElement(bl_el, "ending", attrib={"number": num, "type": typ})

    for bl in left_bars:
        _emit_barline(bl)

    # Time position tracking + inline attribute changes.
    current_time_ticks = 0
    # Track attributes "active" for emission inline within measure (after backup):
    # Only need to emit a fresh <attributes> when these change between sub-blocks.
    active_key = first["key"]
    active_time = first["time"]
    active_clefs_by_staff = dict(clefs_by_staff)
    for idx, sub in enumerate(subs):
        sub["_multi_voice"] = flags["multi_voice"]
        sub["_multi_staff"] = flags["multi_staff"]
        target_time = int(sub.get("_start_q", 0) * DIVISIONS)
        if target_time < current_time_ticks:
            bak = SubElement(m_el, "backup")
            SubElement(bak, "duration").text = str(current_time_ticks - target_time)
            current_time_ticks = target_time
        elif target_time > current_time_ticks:
            fwd = SubElement(m_el, "forward")
            SubElement(fwd, "duration").text = str(target_time - current_time_ticks)
            current_time_ticks = target_time
        # Inline attribute change if this sub-block differs from currently active
        if idx > 0:
            new_clef = sub["clef"]
            staff_str = str(sub["S"])
            attrs_changes = {}
            if sub["key"] != active_key:
                attrs_changes["key"] = sub["key"]
            if sub["time"] != active_time:
                attrs_changes["time"] = sub["time"]
            if active_clefs_by_staff.get(staff_str) != new_clef:
                attrs_changes["clef"] = (staff_str, new_clef)
            if attrs_changes:
                ae = SubElement(m_el, "attributes")
                if "key" in attrs_changes:
                    k_el = SubElement(ae, "key")
                    SubElement(k_el, "fifths").text = str(attrs_changes["key"])
                    active_key = attrs_changes["key"]
                if "time" in attrs_changes:
                    b, bt = attrs_changes["time"].split("/")
                    t_el = SubElement(ae, "time")
                    SubElement(t_el, "beats").text = b
                    SubElement(t_el, "beat-type").text = bt
                    active_time = attrs_changes["time"]
                if "clef" in attrs_changes:
                    cs_num, clef_str_new = attrs_changes["clef"]
                    ms = re.match(r"^([A-Z])(-?\d+)$", clef_str_new)
                    sign, line = (ms.group(1), ms.group(2)) if ms else ("G", "2")
                    cattr = {"number": cs_num} if len(active_clefs_by_staff) > 1 else {}
                    c_el = SubElement(ae, "clef", attrib=cattr)
                    SubElement(c_el, "sign").text = sign
                    SubElement(c_el, "line").text = line
                    active_clefs_by_staff[cs_num] = clef_str_new
        sub_ticks = _emit_sub_block_notes(m_el, sub)
        current_time_ticks += sub_ticks

    for bl in right_bars:
        _emit_barline(bl)


def _emit_sub_block_notes(m_el, sub):
    """Emit all <note> / <forward> / <direction> elements for one sub-block.

    Returns the total ticks consumed (for backup purposes).
    """
    pos = sub["pos"]
    dur = sub["dur"]
    stem_seq = sub["stem"]
    acc = sub["acc"]
    tie_d = sub["tie"]
    tied_d = sub["tied"]
    slur_d = defaultdict(list)
    for num, beat, j_chord, typ in sub["slur"]:
        if j_chord is None:
            slur_d[beat].append((num, typ))
    fer = sub["fer"]
    art = sub["art"]
    orn = sub["orn"]
    tup = sub["tup"]
    lyrics = sub["lyrics"]
    dirs_by_beat = defaultdict(list)
    for beat, place, dtype, dval in sub["dirs"]:
        dirs_by_beat[beat].append((place, dtype, dval))

    multi_staff = sub.get("_multi_staff", False)
    multi_voice = sub.get("_multi_voice", False)
    staff_num = str(sub["S"])
    voice_num = str(sub["V"])
    key = sub["key"]

    total_ticks = 0
    n_beats = len(pos)
    for i, (pos_tok, dur_tok) in enumerate(zip(pos, dur)):
        # Emit any directions queued for this beat (BEFORE the note)
        if i in dirs_by_beat:
            for place, dtype, dval in dirs_by_beat[i]:
                _emit_direction(m_el, place, dtype, dval)

        kind, notes = _split_pos(pos_tok)
        if kind == "forward":
            ticks = _ticks_of(dur_tok)
            f_el = SubElement(m_el, "forward")
            SubElement(f_el, "duration").text = str(ticks)
            total_ticks += ticks
            continue

        if dur_tok == "m":
            beats_n, beat_type_n = sub["time"].split("/")
            ticks = _measure_ticks(int(beats_n), int(beat_type_n))
            d_type, dots = None, 0
        else:
            d_type, dots = _parse_dur(dur_tok)
            ticks = _ticks_of(dur_tok)

        # Apply tuplet scaling: a triplet eighth (3in2) has type=eighth but
        # actual duration = eighth × 2/3. The <duration> ticks must reflect
        # the scaled time; the <type> stays unscaled.
        if i in tup:
            tm_match = re.match(r"^(\d+)in(\d+)$", tup[i])
            if tm_match:
                actual = int(tm_match.group(1))
                normal = int(tm_match.group(2))
                if actual > 0:
                    ticks = ticks * normal // actual

        is_grace = kind == "grace"
        if not is_grace:
            total_ticks += ticks

        if kind == "rest" or kind == "rest_m":
            n_el = SubElement(m_el, "note")
            r_el = SubElement(n_el, "rest")
            if kind == "rest_m":
                r_el.set("measure", "yes")
            SubElement(n_el, "duration").text = str(ticks)
            if d_type:
                SubElement(n_el, "type").text = d_type
                for _ in range(dots):
                    SubElement(n_el, "dot")
            # time-modification on rests (tuplets can include rests)
            if i in tup:
                tm_str = tup[i]
                m = re.match(r"^(\d+)in(\d+)$", tm_str)
                if m:
                    tm = SubElement(n_el, "time-modification")
                    SubElement(tm, "actual-notes").text = m.group(1)
                    SubElement(tm, "normal-notes").text = m.group(2)
            if multi_voice:
                SubElement(n_el, "voice").text = voice_num
            if multi_staff:
                SubElement(n_el, "staff").text = staff_num
            # Notations on rests (e.g., fermata)
            if i in fer:
                not_el = SubElement(n_el, "notations")
                SubElement(not_el, "fermata")
        else:
            # Note(s) or grace(s) — possibly a chord
            for j, (step, octave) in enumerate(notes):
                n_el = SubElement(m_el, "note")
                if is_grace:
                    SubElement(n_el, "grace")
                if j > 0:
                    SubElement(n_el, "chord")
                p_el = SubElement(n_el, "pitch")
                SubElement(p_el, "step").text = step
                chord_idx = j if len(notes) > 1 else None
                acc_key = (i, chord_idx)
                if acc_key in acc:
                    alter = acc[acc_key]
                    has_explicit_acc = True
                else:
                    alter = key_implied_alter(step, key)
                    has_explicit_acc = False
                if alter != 0 or has_explicit_acc:
                    SubElement(p_el, "alter").text = str(alter)
                SubElement(p_el, "octave").text = octave

                if not is_grace:
                    SubElement(n_el, "duration").text = str(ticks)
                # Note tie (in <note>): leader gets the leader entry, chord
                # member j gets its own entry from tie_chord if present.
                is_leader_j = (j == 0)
                if is_leader_j and i in tie_d:
                    for t in tie_d[i]:
                        SubElement(n_el, "tie", attrib={"type": t})
                # Chord-member ties: <beat>.<j>=...
                if not is_leader_j:
                    for t in sub.get("tie_chord", {}).get((i, j), []):
                        SubElement(n_el, "tie", attrib={"type": t})
                if d_type:
                    SubElement(n_el, "type").text = d_type
                    for _ in range(dots):
                        SubElement(n_el, "dot")
                if has_explicit_acc:
                    SubElement(n_el, "accidental").text = ACC_LONG.get(alter, "natural")
                # time-modification
                if i in tup:
                    tm_str = tup[i]
                    m = re.match(r"^(\d+)in(\d+)$", tm_str)
                    if m:
                        tm = SubElement(n_el, "time-modification")
                        SubElement(tm, "actual-notes").text = m.group(1)
                        SubElement(tm, "normal-notes").text = m.group(2)
                # stem
                stem_word = stem_seq[i] if i < len(stem_seq) else None
                if stem_word:
                    SubElement(n_el, "stem").text = stem_word
                # notations
                # Notations: leader gets all leader entries; chord member j
                # only gets its own per-chord-member entries (if any).
                is_leader = (j == 0)
                # Per-chord-member lookups
                cm_tieds = sub.get("tied_chord", {}).get((i, j), [])
                cm_arts = sub.get("art_chord", {}).get((i, j))
                cm_orns = sub.get("orn_chord", {}).get((i, j))
                cm_slurs = [(num, typ) for (num, ii, jj, typ) in sub["slur"]
                            if ii == i and jj == j]

                if is_leader:
                    leader_slurs = [(num, typ) for (num, ii, jj, typ) in sub["slur"]
                                    if ii == i and jj is None]
                    notations_needed = (
                        i in tied_d or leader_slurs or i in fer
                        or i in art or i in orn
                    )
                else:
                    leader_slurs = []
                    notations_needed = bool(cm_tieds or cm_slurs or cm_arts or cm_orns)

                if notations_needed:
                    not_el = SubElement(n_el, "notations")
                    if is_leader and i in tied_d:
                        for t in tied_d[i]:
                            SubElement(not_el, "tied", attrib={"type": t})
                    if not is_leader:
                        for t in cm_tieds:
                            SubElement(not_el, "tied", attrib={"type": t})
                    slurs_to_emit = leader_slurs if is_leader else cm_slurs
                    for (snum, styp) in slurs_to_emit:
                        SubElement(not_el, "slur",
                                   attrib={"number": snum, "type": styp})
                    if is_leader and i in fer:
                        SubElement(not_el, "fermata")
                    art_value = art.get(i) if is_leader else cm_arts
                    if art_value:
                        a_el = SubElement(not_el, "articulations")
                        for a in art_value.split(","):
                            SubElement(a_el, a)
                    orn_value = orn.get(i) if is_leader else cm_orns
                    if orn_value:
                        o_el = SubElement(not_el, "ornaments")
                        for o in orn_value.split(","):
                            SubElement(o_el, o)
                # voice / staff (only when part actually has multiple)
                if multi_voice:
                    SubElement(n_el, "voice").text = voice_num
                if multi_staff:
                    SubElement(n_el, "staff").text = staff_num
                # lyrics — only on the LEADER (j == 0)
                if j == 0:
                    for v_num, arr in lyrics.items():
                        if i < len(arr) and arr[i]:
                            syl, txt = arr[i]
                            lyr_el = SubElement(n_el, "lyric",
                                                attrib={"number": v_num})
                            SubElement(lyr_el, "syllabic").text = syl
                            SubElement(lyr_el, "text").text = txt

    # Emit any directions whose beat_i >= n_beats (post-loop trailing)
    for beat_i in sorted(dirs_by_beat):
        if beat_i >= n_beats:
            for place, dtype, dval in dirs_by_beat[beat_i]:
                _emit_direction(m_el, place, dtype, dval)

    return total_ticks


def _emit_direction(m_el, place, dtype, dval):
    d_el = SubElement(m_el, "direction")
    if place:
        d_el.set("placement", place)
    dt = SubElement(d_el, "direction-type")
    if dtype == "words":
        w = SubElement(dt, "words")
        w.text = _unescape(dval)
    elif dtype == "dyn":
        dyn = SubElement(dt, "dynamics")
        SubElement(dyn, dval)
    elif dtype == "wedge":
        SubElement(dt, "wedge", attrib={"type": dval})
    elif dtype == "metro":
        metro = SubElement(dt, "metronome")
        if "=" in dval:
            bu, pm = dval.split("=", 1)
            SubElement(metro, "beat-unit").text = bu
            SubElement(metro, "per-minute").text = pm


if __name__ == "__main__":
    import sys
    with open(sys.argv[1]) as f:
        print(mxc3_to_xml(f.read()))
