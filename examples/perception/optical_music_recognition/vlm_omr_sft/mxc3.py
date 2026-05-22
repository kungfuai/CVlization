"""DEPRECATED — superseded 2026-05-21 by an MXC2-based per-measure pipeline.

This module was an attempt to decouple MusicXML into per-aspect parallel
channels (pos / dur / stem / acc / tie / slur / lyr / ...) so the VLM
could train on each channel independently. It reached 100% round-trip
on synthetic L7a/L9 and 84% on openscore lieder via a musical-equivalence
normalizer.

The decoupling motivation was speculative — we never validated that the
VLM actually benefits from per-aspect channels. The Level 7a investigation
showed the residual error is one wrong-key belief that cascades through
note spellings; that's better addressed by a two-stage pipeline:

    1. image → key=N (focused classifier)
    2. image → MXC2 transcription (single model)
    3. deterministic re-spell of step 2's notes using step 1's key

See `mxc2_slice.py` (per-measure stateless slicer) and the re-spell
implementation for the active path. MXC3 is kept here as reference but
should not be used for new training.

Original MXC3 — decoupled MusicXML Compact format v3.

Channel-separated, self-contained per-(measure, part, staff, voice) encoding.
See MXC3.md for the spec.

Coverage (round-trip on L7a, L9, openscore lieder):
- Header: work-title, movement-title, composer, lyricist.
- Parts: id, name, abbr.
- Measure header: M=N P=K [S=s] [V=v] key=int time=B/B clef=<sign><line>.
- Per-(M,P,S,V) sub-block — strict parallel arrays:
    pos:  D4, [D4,F4,A4], R (rest), Rm (measure-rest), F (forward), gE4 (grace), g[E4,G4] (grace chord)
    dur:  quarter, quarter., quarter.., m (measure-derived)
    stem: u / d / n / . (parallel; emitted only if any non-.)
    acc:  <i>=<sym> or <i>.<j>=<sym> for chord member j (#, ##, b, bb, n)
    tie:  <i>=<type>   (start / stop / continue) — MusicXML <tie>
    tied: <i>=<type>   — MusicXML <tied> in notations
    slur: <num>:<i>=<type>
    fer:  <i> <j>      beats with fermata
    art:  <i>=<a>,<b>  articulations comma-joined per beat
    orn:  <i>=<a>,<b>  ornaments
    tup:  <i>=NinM     tuplet/time-modification
    lyr1: parallel array, `--` for no lyric, `<syl>:<text>` where syl ∈ {s,b,m,e}
    lyr2: ...
    bar:  loc=<l> style=<s> [repeat=<dir>] [ending=<num>:<type>] (one line per barline)
    dir:  <i>:@<place>:<type>=<value> tokens (positional within measure)

Public API:
    xml_to_mxc3(xml: str) -> str
"""

import re
import xml.etree.ElementTree as ET


STEM_TO_SYM = {"up": "u", "down": "d", "none": "n"}
SYM_TO_STEM = {v: k for k, v in STEM_TO_SYM.items()}

ALTER_TO_SYM = {-2: "bb", -1: "b", 0: "n", 1: "#", 2: "##"}
SYM_TO_ALTER = {v: k for k, v in ALTER_TO_SYM.items()}

SYLLABIC_TO_SHORT = {"single": "s", "begin": "b", "middle": "m", "end": "e"}
SHORT_TO_SYLLABIC = {v: k for k, v in SYLLABIC_TO_SHORT.items()}

SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]


def key_implied_alter(step: str, fifths: int) -> int:
    if fifths > 0:
        return 1 if step in SHARP_ORDER[:fifths] else 0
    if fifths < 0:
        return -1 if step in FLAT_ORDER[:-fifths] else 0
    return 0


def _escape_text(s: str) -> str:
    """Encode whitespace as underscore (matches MXC2 lyric escape)."""
    return re.sub(r"\s", "_", s)


def _escape_q(s: str) -> str:
    """Escape embedded double quotes for use inside `"..."` values."""
    return s.replace('\\', '\\\\').replace('"', '\\"')


def _unescape_q(s: str) -> str:
    return s.replace('\\"', '"').replace('\\\\', '\\')


def _fmt_dur(note) -> str:
    t = note.findtext("type") or "quarter"
    dots = len(note.findall("dot"))
    return t + "." * dots


def _read_pitch(note):
    p = note.find("pitch")
    if p is None:
        return None
    step = p.findtext("step") or "C"
    alter_text = p.findtext("alter")
    alter = int(alter_text) if alter_text is not None else 0
    octave = p.findtext("octave") or "4"
    return step, alter, octave


def xml_to_mxc3(xml: str) -> str:
    root = ET.fromstring(xml)
    lines = []

    # ── Header ───────────────────────────────────────────────────────────────
    parts_meta = []
    pl = root.find("part-list")
    if pl is not None:
        for sp in pl.findall("score-part"):
            parts_meta.append({
                "id": sp.get("id", ""),
                "name": (sp.findtext("part-name") or "").strip(),
                "abbr": (sp.findtext("part-abbreviation") or "").strip(),
            })

    header_extras = {}
    work = root.find("work")
    if work is not None:
        wt = work.findtext("work-title")
        if wt:
            header_extras["work-title"] = wt
    mn = root.findtext("movement-number")
    if mn:
        header_extras["movement-number"] = mn
    mt = root.findtext("movement-title")
    if mt:
        header_extras["movement-title"] = mt
    ident = root.find("identification")
    if ident is not None:
        for cr in ident.findall("creator"):
            ctype = cr.get("type", "")
            ctext = (cr.text or "").strip()
            if ctype in ("composer", "lyricist") and ctext:
                header_extras[ctype] = ctext

    header = [f"HEADER parts={len(parts_meta)}"]
    for k in ("work-title", "movement-number", "movement-title", "composer", "lyricist"):
        if k in header_extras:
            v = header_extras[k]
            if k == "movement-number" and v.isdigit():
                header.append(f"{k}={v}")
            else:
                header.append(f'{k}="{_escape_q(v)}"')
    lines.append(" ".join(header))

    for i, pm in enumerate(parts_meta, 1):
        toks = [f"P={i}"]
        if pm["name"]:
            toks.append(f'name="{pm["name"]}"')
        if pm["abbr"]:
            toks.append(f'abbr="{pm["abbr"]}"')
        if pm["id"]:
            toks.append(f'id="{pm["id"]}"')
        lines.append(" ".join(toks))

    # ── Parts → measures ─────────────────────────────────────────────────────
    parts = root.findall("part")
    ctxs = [{"key": 0, "time": "4/4", "clefs": {"1": "G2"}, "divisions": 1,
             "staves": 1, "beats": 4, "beat_type": 4} for _ in parts]

    # Measure-major emission: M=1 of all parts, then M=2, etc.
    # `===` separator between measure groups so the decoder can detect
    # measure boundaries even when the same measure number repeats.
    if not parts:
        return "\n".join(lines) + "\n"
    n_meas = max(len(p.findall("measure")) for p in parts)
    for m_i in range(n_meas):
        if m_i > 0:
            lines.append("")
            lines.append("===")
        for p_i, part in enumerate(parts):
            measures = part.findall("measure")
            if m_i >= len(measures):
                continue
            measure = measures[m_i]
            ctx = ctxs[p_i]
            _update_context(measure, ctx)
            _encode_measure(measure, ctx, p_i + 1, lines)

    return "\n".join(lines) + "\n"


def _update_context(measure, ctx):
    attrs = measure.find("attributes")
    if attrs is None:
        return
    div = attrs.findtext("divisions")
    if div:
        ctx["divisions"] = int(div)
    key_el = attrs.find("key")
    if key_el is not None:
        f = key_el.findtext("fifths")
        if f is not None:
            ctx["key"] = int(f)
    time_el = attrs.find("time")
    if time_el is not None:
        b = time_el.findtext("beats") or "4"
        bt = time_el.findtext("beat-type") or "4"
        ctx["time"] = f"{b}/{bt}"
        ctx["beats"] = int(b)
        ctx["beat_type"] = int(bt)
    staves = attrs.findtext("staves")
    if staves:
        ctx["staves"] = int(staves)
    for c in attrs.findall("clef"):
        sign = c.findtext("sign") or "G"
        line = c.findtext("line") or "2"
        num = c.get("number", "1")
        ctx["clefs"][num] = f"{sign}{line}"


def _encode_measure(measure, ctx, p_num, out_lines):
    """Group notes by (staff, voice) into sub-blocks and emit each."""
    m_num = measure.get("number", "0")

    # Pre-process: drop `<forward>` followed (possibly across intervening
    # `<direction>` / `<print>` / `<barline>`) by a `<backup>` with matching
    # duration. These are net-zero positioning tricks used by MuseScore-
    # engraved files, not musical content.
    raw_children = list(measure)
    children_iter = []
    skip = set()
    j = 0
    while j < len(raw_children):
        c = raw_children[j]
        if c.tag == "forward":
            k = j + 1
            while k < len(raw_children) and raw_children[k].tag in (
                "direction", "print", "barline"
            ):
                k += 1
            if (k < len(raw_children)
                    and raw_children[k].tag == "backup"
                    and c.findtext("duration") == raw_children[k].findtext("duration")):
                skip.add(j); skip.add(k)
        j += 1
    children_iter = [c for i, c in enumerate(raw_children) if i not in skip]

    # Walk children in document order, building a list of *segments*. A
    # segment is one contiguous run of (staff, voice) content at a given
    # time. Backups break segments (the same voice may have multiple
    # segments at different times).
    barlines = []
    segments = []
    current_seg = None
    pending_dirs = []
    current_voice = "1"
    current_staff = "1"
    current_time_q = 0.0
    divisions = ctx["divisions"]

    def _dur_q(el):
        d = el.findtext("duration")
        return (int(d) / divisions) if d and divisions else 0.0

    def _open_segment(s, v, start_q):
        nonlocal current_seg
        # Only open new segment if not already in matching one
        if (current_seg is None
                or current_seg["s"] != s
                or current_seg["v"] != v
                or current_seg.get("_closed", False)):
            current_seg = {
                "s": s, "v": v, "start_q": start_q,
                "events": [], "directions": [], "beat_count": 0,
                "key": ctx["key"], "time": ctx["time"],
                "clef": ctx["clefs"].get(s) or next(iter(ctx["clefs"].values())),
            }
            segments.append(current_seg)

    for child in children_iter:
        if child.tag == "note":
            v = child.findtext("voice") or "1"
            s = child.findtext("staff") or "1"
            current_voice, current_staff = v, s
            _open_segment(s, v, current_time_q)
            for d in pending_dirs:
                current_seg["directions"].append((current_seg["beat_count"], d))
            pending_dirs = []
            current_seg["events"].append(("note", child))
            is_chord = child.find("chord") is not None
            if not is_chord:
                current_seg["beat_count"] += 1
                if child.find("grace") is None:
                    current_time_q += _dur_q(child)
        elif child.tag == "forward":
            _open_segment(current_staff, current_voice, current_time_q)
            for d in pending_dirs:
                current_seg["directions"].append((current_seg["beat_count"], d))
            pending_dirs = []
            current_seg["events"].append(("forward", child))
            current_seg["beat_count"] += 1
            current_time_q += _dur_q(child)
        elif child.tag == "direction":
            pending_dirs.append(child)
        elif child.tag == "backup":
            current_time_q -= _dur_q(child)
            if current_seg is not None:
                current_seg["_closed"] = True
        elif child.tag == "attributes":
            div = child.findtext("divisions")
            if div:
                ctx["divisions"] = int(div)
                divisions = ctx["divisions"]
            key_el = child.find("key")
            if key_el is not None:
                f = key_el.findtext("fifths")
                if f is not None:
                    ctx["key"] = int(f)
            time_el = child.find("time")
            if time_el is not None:
                b = time_el.findtext("beats") or "4"
                bt = time_el.findtext("beat-type") or "4"
                ctx["time"] = f"{b}/{bt}"
                ctx["beats"] = int(b)
                ctx["beat_type"] = int(bt)
            for c in child.findall("clef"):
                sign = c.findtext("sign") or "G"
                line = c.findtext("line") or "2"
                num = c.get("number", "1")
                ctx["clefs"][num] = f"{sign}{line}"
            if current_seg is not None:
                current_seg["_closed"] = True
        elif child.tag == "barline":
            if current_seg is not None:
                for d in pending_dirs:
                    current_seg["directions"].append((current_seg["beat_count"], d))
            pending_dirs = []
            barlines.append(child)

    if pending_dirs and current_seg is not None:
        for d in pending_dirs:
            current_seg["directions"].append((current_seg["beat_count"], d))

    if not segments:
        clef_str = next(iter(ctx["clefs"].values()))
        out_lines.append("")
        out_lines.append(
            f"M={m_num} P={p_num} key={ctx['key']} "
            f"time={ctx['time']} clef={clef_str}"
        )
        for bl in barlines:
            out_lines.append("  " + _fmt_barline(bl))
        return

    staves = {seg["s"] for seg in segments}
    voices = {seg["v"] for seg in segments}
    has_multi_staff = len(staves) > 1 or any(s != "1" for s in staves) and ctx["staves"] > 1
    has_multi_voice = len(voices) > 1

    first_sub = True
    for seg in segments:
        staff, voice = seg["s"], seg["v"]
        events = seg["events"]
        dirs_here = seg["directions"]
        beats = _build_beats(events, divisions)
        if not beats and not (first_sub and barlines):
            continue
        clef_str = seg["clef"]

        header = [f"M={m_num}", f"P={p_num}"]
        if has_multi_staff or staff != "1":
            header.append(f"S={staff}")
        if has_multi_voice or voice != "1":
            header.append(f"V={voice}")
        header += [f"key={seg['key']}", f"time={seg['time']}", f"clef={clef_str}"]
        st = seg["start_q"]
        if st != 0:
            if st == int(st):
                header.append(f"start={int(st)}")
            else:
                header.append(f"start={st:g}")
        out_lines.append("")
        out_lines.append(" ".join(header))

        _emit_channels(beats, ctx, out_lines)
        if dirs_here:
            _emit_directions(dirs_here, out_lines)
        if first_sub:
            for bl in barlines:
                out_lines.append("  " + _fmt_barline(bl))
            first_sub = False


def _build_beats(events, divisions=1):
    """Build a list of beat dicts from (kind, element) tuples.

    A 'beat' here = one event slot (single note, chord, rest, grace, or forward).
    Chord continuation collapses into the previous beat.
    """
    beats = []
    for kind, el in events:
        if kind == "forward":
            beats.append({
                "kind": "forward",
                "dur": _fmt_dur_or_compound(el, divisions),
                "stem": None,
                "pitches": [],
                "ties": [], "tieds": [], "slurs": [], "fermata": False,
                "articulations": [], "ornaments": [], "tuplet": None,
                "lyrics": {},
            })
            continue
        # note
        is_chord = el.find("chord") is not None
        is_grace = el.find("grace") is not None
        is_rest = el.find("rest") is not None
        is_measure_rest = is_rest and el.find("rest").get("measure") == "yes"

        if is_chord and beats:
            # Add to previous beat's chord. Capture per-chord-member ties /
            # tieds / slurs / articulations / ornaments — they can differ
            # between chord members and are musically meaningful.
            pitch = _read_pitch(el)
            if pitch:
                step, alter, octave = pitch
                has_acc_glyph = el.find("accidental") is not None
                beats[-1]["pitches"].append((step, alter, octave, has_acc_glyph))
                j = len(beats[-1]["pitches"]) - 1  # chord-member index
                for tie in el.findall("tie"):
                    beats[-1].setdefault("chord_ties", []).append(
                        (j, tie.get("type", ""))
                    )
                notations = el.find("notations")
                if notations is not None:
                    for tied in notations.findall("tied"):
                        beats[-1].setdefault("chord_tieds", []).append(
                            (j, tied.get("type", ""))
                        )
                    for sl in notations.findall("slur"):
                        beats[-1].setdefault("chord_slurs", []).append(
                            (sl.get("number", "1"), j, sl.get("type", ""))
                        )
                    artic = notations.find("articulations")
                    if artic is not None:
                        for a in artic:
                            beats[-1].setdefault("chord_arts", []).append((j, a.tag))
                    orn = notations.find("ornaments")
                    if orn is not None:
                        for o in orn:
                            beats[-1].setdefault("chord_orns", []).append((j, o.tag))
            continue

        b = {
            "kind": "rest" if is_rest else ("grace" if is_grace else "note"),
            "measure_rest": is_measure_rest,
            "dur": _fmt_dur(el),
            "stem": STEM_TO_SYM.get(el.findtext("stem") or ""),
            "pitches": [],
            "ties": [], "tieds": [], "slurs": [], "fermata": False,
            "articulations": [], "ornaments": [], "tuplet": None,
            "lyrics": {},
        }

        if not is_rest:
            pitch = _read_pitch(el)
            if pitch:
                step, alter, octave = pitch
                has_acc_glyph = el.find("accidental") is not None
                b["pitches"].append((step, alter, octave, has_acc_glyph))

        # Time modification (tuplet)
        tm = el.find("time-modification")
        if tm is not None:
            a = tm.findtext("actual-notes")
            n = tm.findtext("normal-notes")
            if a and n:
                b["tuplet"] = f"{a}in{n}"

        # Ties (MusicXML <tie>)
        for tie in el.findall("tie"):
            b["ties"].append(tie.get("type", ""))

        # Notations
        notations = el.find("notations")
        if notations is not None:
            for tied in notations.findall("tied"):
                b["tieds"].append(tied.get("type", ""))
            for sl in notations.findall("slur"):
                b["slurs"].append(
                    (sl.get("number", "1"), sl.get("type", ""))
                )
            if notations.find("fermata") is not None:
                b["fermata"] = True
            artic = notations.find("articulations")
            if artic is not None:
                for a in artic:
                    b["articulations"].append(a.tag)
            orn = notations.find("ornaments")
            if orn is not None:
                for o in orn:
                    b["ornaments"].append(o.tag)

        # Lyrics
        for lyr in el.findall("lyric"):
            lnum = lyr.get("number") or lyr.get("name") or "1"
            syl = lyr.findtext("syllabic") or "single"
            txt = lyr.findtext("text") or ""
            b["lyrics"][lnum] = (SYLLABIC_TO_SHORT.get(syl, syl), txt)

        beats.append(b)

    return beats


def _fmt_dur_or_compound(el, divisions):
    """Forward elements don't have <type>; convert source-tick duration to a
    unit-free quarter fraction (q=N) so the decoder can reconstruct in its
    own divisions unit. Without this the source's div=10080 ticks would be
    misread by the decoder's div=96."""
    dur_text = el.findtext("duration")
    if dur_text and divisions > 0:
        q = int(dur_text) / divisions
        # Use int representation if exact, else keep float
        if q == int(q):
            return f"q={int(q)}"
        return f"q={q:g}"
    return "quarter"


def _emit_channels(beats, ctx, out_lines):
    pos = []
    dur = []
    stem = []
    acc_overrides = []
    tie_overrides = []
    tied_overrides = []
    slur_overrides = []
    fermata_idxs = []
    art_overrides = []
    orn_overrides = []
    tup_overrides = []
    lyric_verses = {}   # verse_num -> list of '--' or 'syl:text'

    key = ctx["key"]

    for i, b in enumerate(beats):
        if b["kind"] == "forward":
            pos.append("F")
            stem.append(".")
        elif b["kind"] == "rest":
            pos.append("Rm" if b["measure_rest"] else "R")
            stem.append(".")
        elif b["kind"] == "grace":
            if len(b["pitches"]) == 0:
                pos.append("gR")
            elif len(b["pitches"]) == 1:
                step, alter, octave, has_acc = b["pitches"][0]
                pos.append(f"g{step}{octave}")
                _add_acc(acc_overrides, i, None, step, alter, has_acc, key)
            else:
                chord = []
                for j, (step, alter, octave, has_acc) in enumerate(b["pitches"]):
                    chord.append(f"{step}{octave}")
                    _add_acc(acc_overrides, i, j, step, alter, has_acc, key)
                pos.append("g[" + ",".join(chord) + "]")
            stem.append(b["stem"] or ".")
        else:
            if len(b["pitches"]) == 1:
                step, alter, octave, has_acc = b["pitches"][0]
                pos.append(f"{step}{octave}")
                _add_acc(acc_overrides, i, None, step, alter, has_acc, key)
            else:
                chord = []
                for j, (step, alter, octave, has_acc) in enumerate(b["pitches"]):
                    chord.append(f"{step}{octave}")
                    _add_acc(acc_overrides, i, j, step, alter, has_acc, key)
                pos.append("[" + ",".join(chord) + "]")
            stem.append(b["stem"] or ".")

        # Duration
        if b["kind"] == "rest" and b["measure_rest"]:
            dur.append("m")
        else:
            dur.append(b["dur"])

        # Per-beat aux channels (chord leader: index `i`; chord member j: `i.j`)
        for t in b["ties"]:
            tie_overrides.append(f"{i}={t}")
        for t in b["tieds"]:
            tied_overrides.append(f"{i}={t}")
        for num, typ in b["slurs"]:
            slur_overrides.append(f"{num}:{i}={typ}")
        if b["fermata"]:
            fermata_idxs.append(str(i))
        if b["articulations"]:
            art_overrides.append(f"{i}=" + ",".join(b["articulations"]))
        if b["ornaments"]:
            orn_overrides.append(f"{i}=" + ",".join(b["ornaments"]))
        # Per-chord-member overrides for ties/tieds/slurs/articulations/ornaments
        for j, typ in b.get("chord_ties", []):
            tie_overrides.append(f"{i}.{j}={typ}")
        for j, typ in b.get("chord_tieds", []):
            tied_overrides.append(f"{i}.{j}={typ}")
        for num, j, typ in b.get("chord_slurs", []):
            slur_overrides.append(f"{num}:{i}.{j}={typ}")
        for j, a_tag in b.get("chord_arts", []):
            art_overrides.append(f"{i}.{j}={a_tag}")
        for j, o_tag in b.get("chord_orns", []):
            orn_overrides.append(f"{i}.{j}={o_tag}")
        if b["tuplet"]:
            tup_overrides.append(f"{i}={b['tuplet']}")

        # Lyrics (per verse)
        for v_num, (syl_short, txt) in b["lyrics"].items():
            lyric_verses.setdefault(v_num, [None] * len(beats))[i] = (
                f"{syl_short}:{_escape_text(txt)}" if txt else f"{syl_short}:"
            )

    # Pad lyric arrays
    for v_num in lyric_verses:
        arr = lyric_verses[v_num]
        if len(arr) < len(beats):
            arr.extend([None] * (len(beats) - len(arr)))

    if pos:
        out_lines.append("  pos: " + " ".join(pos))
        out_lines.append("  dur: " + " ".join(dur))
        if any(s != "." for s in stem):
            out_lines.append("  stem: " + " ".join(stem))
        if acc_overrides:
            out_lines.append("  acc: " + " ".join(acc_overrides))
        if tie_overrides:
            out_lines.append("  tie: " + " ".join(tie_overrides))
        if tied_overrides:
            out_lines.append("  tied: " + " ".join(tied_overrides))
        if slur_overrides:
            out_lines.append("  slur: " + " ".join(slur_overrides))
        if fermata_idxs:
            out_lines.append("  fer: " + " ".join(fermata_idxs))
        if art_overrides:
            out_lines.append("  art: " + " ".join(art_overrides))
        if orn_overrides:
            out_lines.append("  orn: " + " ".join(orn_overrides))
        if tup_overrides:
            out_lines.append("  tup: " + " ".join(tup_overrides))
        for v_num in sorted(lyric_verses.keys(), key=lambda x: (int(x) if x.isdigit() else x)):
            arr = lyric_verses[v_num]
            toks = [x if x else "--" for x in arr]
            out_lines.append(f"  lyr{v_num}: " + " ".join(toks))


def _add_acc(out, beat_i, chord_j, step, alter, has_acc_glyph, key):
    implied = key_implied_alter(step, key)
    if alter != implied or has_acc_glyph:
        sym = ALTER_TO_SYM.get(alter, "n")
        idx = f"{beat_i}" if chord_j is None else f"{beat_i}.{chord_j}"
        out.append(f"{idx}={sym}")


def _emit_directions(directions, out_lines):
    toks = []
    for beat_i, d in directions:
        place = d.get("placement", "")
        place_tok = f"@{place}:" if place else "@:"
        for dt in d.findall("direction-type"):
            for child in dt:
                tag = child.tag
                if tag == "words":
                    text = (child.text or "").strip()
                    font_attrs = []
                    for fa in ("font-size", "font-weight", "font-style"):
                        v = child.get(fa)
                        if v:
                            font_attrs.append(f"{fa}={v}")
                    font_str = "[" + ",".join(font_attrs) + "]" if font_attrs else ""
                    if text or font_str:
                        toks.append(
                            f"{beat_i}:{place_tok}words={font_str}{_escape_text(text)}"
                        )
                elif tag == "dynamics":
                    for d_el in child:
                        toks.append(f"{beat_i}:{place_tok}dyn={d_el.tag}")
                elif tag == "wedge":
                    wtype = child.get("type", "")
                    toks.append(f"{beat_i}:{place_tok}wedge={wtype}")
                elif tag == "metronome":
                    bu = child.findtext("beat-unit") or ""
                    pm = child.findtext("per-minute") or ""
                    toks.append(f"{beat_i}:{place_tok}metro={bu}={pm}")
                else:
                    toks.append(f"{beat_i}:{place_tok}{tag}=")
    if toks:
        out_lines.append("  dir: " + " ".join(toks))


def _fmt_barline(bl):
    loc = bl.get("location", "right")
    style = bl.findtext("bar-style") or ""
    parts = [f"bar: loc={loc}"]
    if style:
        parts.append(f"style={style}")
    repeat = bl.find("repeat")
    if repeat is not None:
        parts.append(f"repeat={repeat.get('direction', '')}")
    ending = bl.find("ending")
    if ending is not None:
        parts.append(f"ending={ending.get('number', '')}:{ending.get('type', '')}")
    return " ".join(parts)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            xml = f.read()
        print(xml_to_mxc3(xml))
    else:
        print(__doc__)
