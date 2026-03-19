"""Round-trip MXC tests on real data from zzsi/openscore dataset.

Verifies that xml_to_mxc → mxc_to_xml produces semantically equivalent
MusicXML for real training samples.
"""
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

sys.path.insert(0, ".")
from mxc import xml_to_mxc, mxc_to_xml


def strip_musicxml_header(xml: str) -> str:
    """Copy of the function from train.py (avoids torch import)."""
    xml = re.sub(r'<\?xml[^?]*\?>\s*', '', xml)
    xml = re.sub(r'<!DOCTYPE[^>]*>\s*', '', xml)
    # Strip <identification> but keep composer/lyricist (visible on page)
    xml = re.sub(r'\s*<rights>[^<]*</rights>', '', xml)
    xml = re.sub(r'\s*<encoding>.*?</encoding>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<creator type="arranger">[^<]*</creator>', '', xml)
    xml = re.sub(r'\s*<identification>\s*</identification>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<defaults>.*?</defaults>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<movement-title>tmp[^<]*</movement-title>', '', xml)
    xml = re.sub(r'\s*<score-instrument[^>]*>.*?</score-instrument>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<midi-instrument[^>]*>.*?</midi-instrument>', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<midi-device[^>]*/?>', '', xml)
    xml = re.sub(r'\s*<!--.*?-->', '', xml, flags=re.DOTALL)
    xml = re.sub(r'\s*<sound\b[^/]*/>', '', xml)
    xml = re.sub(
        r'\s*<direction[^>]*>\s*<direction-type>\s*<words\s*/>\s*</direction-type>\s*(?:<sound\b[^/]*/>\s*)?</direction>',
        '', xml, flags=re.DOTALL)
    xml = re.sub(
        r'\s*<direction[^>]*>\s*<direction-type>\s*</direction-type>\s*<sound\b[^/]*/>\s*</direction>',
        '', xml, flags=re.DOTALL)
    xml = xml.replace(' implicit="no"', '')
    return xml.strip()


def _extract_notes(root):
    """Extract a list of note dicts from an XML tree for comparison."""
    notes = []
    for note in root.iter("note"):
        d = {}
        pitch = note.find("pitch")
        if pitch is not None:
            d["step"] = pitch.findtext("step")
            d["alter"] = pitch.findtext("alter")
            d["octave"] = pitch.findtext("octave")
        rest = note.find("rest")
        if rest is not None:
            d["rest"] = True
            d["rest_measure"] = rest.get("measure")
        d["duration"] = note.findtext("duration")
        d["type"] = note.findtext("type")
        d["stem"] = note.findtext("stem")
        d["voice"] = note.findtext("voice")
        d["staff"] = note.findtext("staff")
        d["chord"] = note.find("chord") is not None
        d["grace"] = note.find("grace") is not None
        d["accidental"] = note.findtext("accidental")
        d["dots"] = len(note.findall("dot"))
        # Beams
        beams = {}
        for b in note.findall("beam"):
            beams[b.get("number", "1")] = (b.text or "").strip()
        d["beams"] = beams
        # Ties
        d["ties"] = [t.get("type") for t in note.findall("tie")]
        # Lyrics
        lyrics = []
        for ly in note.findall("lyric"):
            lyrics.append({
                "number": ly.get("number") or ly.get("name"),
                "syllabic": ly.findtext("syllabic"),
                "text": ly.findtext("text"),
            })
        d["lyrics"] = lyrics
        notes.append(d)
    return notes


def _extract_measures(root):
    """Extract measure numbers per part."""
    parts = {}
    for part in root.findall("part"):
        pid = part.get("id")
        parts[pid] = [m.get("number") for m in part.findall("measure")]
    return parts


def _extract_attributes(root):
    """Extract attributes from first measure of each part."""
    attrs = {}
    for part in root.findall("part"):
        pid = part.get("id")
        m = part.find("measure")
        if m is None:
            continue
        a = m.find("attributes")
        if a is None:
            continue
        attrs[pid] = {
            "divisions": a.findtext("divisions"),
            "fifths": a.findtext("key/fifths") if a.find("key") is not None else None,
            "beats": a.findtext("time/beats") if a.find("time") is not None else None,
            "beat-type": a.findtext("time/beat-type") if a.find("time") is not None else None,
        }
    return attrs


def roundtrip_check(cleaned_xml):
    """Run round-trip and return list of discrepancies (empty = success)."""
    errors = []

    try:
        mxc = xml_to_mxc(cleaned_xml)
    except Exception as e:
        return [f"xml_to_mxc failed: {e}"]

    try:
        result_xml = mxc_to_xml(mxc)
    except Exception as e:
        return [f"mxc_to_xml failed: {e}"]

    try:
        orig_root = ET.fromstring(cleaned_xml)
        result_root = ET.fromstring(result_xml)
    except ET.ParseError as e:
        return [f"XML parse error: {e}"]

    # Compare part structure
    orig_parts = _extract_measures(orig_root)
    result_parts = _extract_measures(result_root)
    if set(orig_parts.keys()) != set(result_parts.keys()):
        errors.append(f"Part IDs differ: {set(orig_parts.keys())} vs {set(result_parts.keys())}")
    for pid in orig_parts:
        if pid in result_parts:
            if len(orig_parts[pid]) != len(result_parts[pid]):
                errors.append(f"Part {pid}: measure count {len(orig_parts[pid])} vs {len(result_parts[pid])}")

    # Compare attributes
    orig_attrs = _extract_attributes(orig_root)
    result_attrs = _extract_attributes(result_root)
    for pid in orig_attrs:
        if pid in result_attrs:
            if orig_attrs[pid] != result_attrs[pid]:
                errors.append(f"Part {pid} attributes differ: {orig_attrs[pid]} vs {result_attrs[pid]}")

    # Compare notes
    orig_notes = _extract_notes(orig_root)
    result_notes = _extract_notes(result_root)
    if len(orig_notes) != len(result_notes):
        errors.append(f"Note count: {len(orig_notes)} vs {len(result_notes)}")
    else:
        for i, (on, rn) in enumerate(zip(orig_notes, result_notes)):
            for key in ("step", "alter", "octave", "rest", "duration", "type",
                        "stem", "chord", "grace", "accidental", "dots", "voice", "staff"):
                if on.get(key) != rn.get(key):
                    errors.append(f"Note {i} {key}: {on.get(key)!r} vs {rn.get(key)!r}")
            if on.get("beams") != rn.get("beams"):
                errors.append(f"Note {i} beams: {on['beams']} vs {rn['beams']}")
            if on.get("ties") != rn.get("ties"):
                errors.append(f"Note {i} ties: {on['ties']} vs {rn['ties']}")
            if len(on.get("lyrics", [])) != len(rn.get("lyrics", [])):
                errors.append(f"Note {i} lyric count: {len(on['lyrics'])} vs {len(rn['lyrics'])}")
            else:
                for j, (ol, rl) in enumerate(zip(on["lyrics"], rn["lyrics"])):
                    # Normalize non-breaking spaces to regular spaces for comparison
                    ol_norm = {k: (v.replace('\xa0', ' ') if isinstance(v, str) else v) for k, v in ol.items()}
                    rl_norm = {k: (v.replace('\xa0', ' ') if isinstance(v, str) else v) for k, v in rl.items()}
                    if ol_norm != rl_norm:
                        errors.append(f"Note {i} lyric {j}: {ol} vs {rl}")

    return errors


def main():
    from datasets import load_dataset
    ds = load_dataset('zzsi/openscore', 'pages_transcribed', split='train', streaming=True)

    n_tested = 0
    n_passed = 0
    n_failed = 0
    all_error_types = Counter()
    failed_ids = []
    target = 500

    for sample in ds:
        if sample.get('corpus') != 'lieder':
            continue

        sid = sample.get("score_id", f"sample_{n_tested}")
        page = sample.get("page", "?")

        try:
            cleaned = strip_musicxml_header(sample['musicxml'])
            errs = roundtrip_check(cleaned)
        except Exception as e:
            errs = [f"Exception: {e}"]

        n_tested += 1
        if errs:
            n_failed += 1
            if n_failed <= 10:
                print(f"FAIL {sid} page={page}:")
                for e in errs[:5]:
                    print(f"  {e}")
            for e in errs:
                # Categorize error type
                etype = e.split(":")[0] if ":" in e else e.split(" ")[0]
                all_error_types[etype] += 1
            failed_ids.append(sid)
        else:
            n_passed += 1

        if n_tested >= target:
            break

    print(f"\n{'='*60}")
    print(f"Results: {n_passed}/{n_tested} passed, {n_failed} failed")
    print(f"Pass rate: {100*n_passed/n_tested:.1f}%")
    if all_error_types:
        print(f"\nError type breakdown:")
        for etype, count in all_error_types.most_common(20):
            print(f"  {etype}: {count}")


if __name__ == "__main__":
    main()
