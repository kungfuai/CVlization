"""Unit tests for strip_musicxml_header."""
import re
import pytest


def strip_musicxml_header(xml: str) -> str:
    """Copy of the function from train.py (avoids torch import)."""
    xml = re.sub(r'<\?xml[^?]*\?>\s*', '', xml)
    xml = re.sub(r'<!DOCTYPE[^>]*>\s*', '', xml)
    xml = re.sub(r'\s*<identification>.*?</identification>', '', xml, flags=re.DOTALL)
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


# ── Things that should be STRIPPED ──────────────────────────────────────────

def test_strips_xml_declaration():
    xml = '<?xml version="1.0" encoding="utf-8"?>\n<score-partwise/>'
    assert strip_musicxml_header(xml) == '<score-partwise/>'


def test_strips_doctype():
    xml = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0">\n<score-partwise/>'
    assert strip_musicxml_header(xml) == '<score-partwise/>'


def test_strips_identification():
    xml = '<root>\n  <identification>\n    <creator>Test</creator>\n  </identification>\n  <note/>\n</root>'
    result = strip_musicxml_header(xml)
    assert '<identification>' not in result
    assert '<note/>' in result


def test_strips_defaults():
    xml = '<root>\n  <defaults>\n    <scaling><millimeters>7</millimeters></scaling>\n  </defaults>\n  <note/>\n</root>'
    result = strip_musicxml_header(xml)
    assert '<defaults>' not in result
    assert '<note/>' in result


def test_strips_temp_movement_title():
    xml = '<root>\n  <movement-title>tmppmimc8gc.xml</movement-title>\n  <note/>\n</root>'
    result = strip_musicxml_header(xml)
    assert '<movement-title>' not in result


def test_strips_score_instrument():
    xml = '<score-part id="P1">\n  <part-name>Voice</part-name>\n  <score-instrument id="Iabc123">\n    <instrument-name>Voice</instrument-name>\n  </score-instrument>\n</score-part>'
    result = strip_musicxml_header(xml)
    assert '<score-instrument' not in result
    assert '<instrument-name>' not in result
    assert '<part-name>Voice</part-name>' in result


def test_strips_midi_instrument():
    xml = '<score-part id="P1">\n  <midi-instrument id="Iabc">\n    <midi-channel>1</midi-channel>\n    <midi-program>69</midi-program>\n  </midi-instrument>\n</score-part>'
    result = strip_musicxml_header(xml)
    assert '<midi-instrument' not in result
    assert '<midi-channel>' not in result


def test_strips_midi_device():
    xml = '<measure>\n  <midi-device id="I123"/>\n  <note/>\n</measure>'
    result = strip_musicxml_header(xml)
    assert '<midi-device' not in result
    assert '<note/>' in result


def test_strips_xml_comments():
    xml = '<part id="P1">\n  <!--========================= Measure 1 ==========================-->\n  <measure number="1">\n    <note/>\n  </measure>\n</part>'
    result = strip_musicxml_header(xml)
    assert '<!--' not in result
    assert '<note/>' in result


def test_strips_sound_element():
    xml = '<measure>\n  <sound tempo="92" />\n  <note/>\n</measure>'
    result = strip_musicxml_header(xml)
    assert '<sound' not in result
    assert '<note/>' in result


def test_strips_direction_with_empty_words_and_sound():
    xml = '<measure>\n  <direction placement="above">\n    <direction-type>\n      <words />\n    </direction-type>\n    <sound tempo="92" />\n  </direction>\n  <note/>\n</measure>'
    result = strip_musicxml_header(xml)
    assert '<direction' not in result
    assert '<note/>' in result


def test_strips_implicit_no():
    xml = '<measure implicit="no" number="1">\n  <note/>\n</measure>'
    result = strip_musicxml_header(xml)
    assert 'implicit="no"' not in result
    assert 'number="1"' in result


# ── Things that should be KEPT ──────────────────────────────────────────────

def test_keeps_work_title():
    xml = '<work>\n  <work-title>Just for Today</work-title>\n</work>'
    assert '<work-title>Just for Today</work-title>' in strip_musicxml_header(xml)


def test_keeps_real_movement_title():
    xml = '<movement-title>The Old Fisherman</movement-title>'
    assert '<movement-title>The Old Fisherman</movement-title>' in strip_musicxml_header(xml)


def test_keeps_part_name():
    xml = '<score-part id="P1">\n  <part-name>Voice</part-name>\n  <part-abbreviation>Ob</part-abbreviation>\n</score-part>'
    result = strip_musicxml_header(xml)
    assert '<part-name>Voice</part-name>' in result
    assert '<part-abbreviation>Ob</part-abbreviation>' in result


def test_keeps_notes_with_pitch():
    xml = '<note>\n  <pitch>\n    <step>C</step>\n    <octave>4</octave>\n  </pitch>\n  <duration>10080</duration>\n  <type>quarter</type>\n</note>'
    result = strip_musicxml_header(xml)
    assert '<pitch>' in result
    assert '<step>C</step>' in result
    assert '<duration>10080</duration>' in result


def test_keeps_stem():
    xml = '<note>\n  <stem>up</stem>\n</note>'
    assert '<stem>up</stem>' in strip_musicxml_header(xml)


def test_keeps_beam():
    xml = '<note>\n  <beam number="1">begin</beam>\n</note>'
    assert '<beam number="1">begin</beam>' in strip_musicxml_header(xml)


def test_keeps_lyric():
    xml = '<note>\n  <lyric name="1" number="1">\n    <syllabic>single</syllabic>\n    <text>Lord,</text>\n  </lyric>\n</note>'
    result = strip_musicxml_header(xml)
    assert '<lyric' in result
    assert '<text>Lord,</text>' in result


def test_keeps_accidental():
    xml = '<note>\n  <accidental>natural</accidental>\n</note>'
    assert '<accidental>natural</accidental>' in strip_musicxml_header(xml)


def test_keeps_barline():
    xml = '<barline location="right">\n  <bar-style>light-light</bar-style>\n</barline>'
    result = strip_musicxml_header(xml)
    assert '<barline' in result
    assert '<bar-style>light-light</bar-style>' in result


def test_keeps_direction_with_visible_text():
    xml = '<direction placement="above">\n  <direction-type>\n    <words font-size="12" font-weight="bold">Andante</words>\n  </direction-type>\n</direction>'
    result = strip_musicxml_header(xml)
    assert '<direction' in result
    assert 'Andante' in result


def test_keeps_key_signature():
    xml = '<attributes>\n  <key>\n    <fifths>-1</fifths>\n  </key>\n</attributes>'
    result = strip_musicxml_header(xml)
    assert '<key>' in result
    assert '<fifths>-1</fifths>' in result


def test_keeps_time_signature():
    xml = '<attributes>\n  <time>\n    <beats>3</beats>\n    <beat-type>4</beat-type>\n  </time>\n</attributes>'
    result = strip_musicxml_header(xml)
    assert '<beats>3</beats>' in result


def test_keeps_clef():
    xml = '<attributes>\n  <clef>\n    <sign>G</sign>\n    <line>2</line>\n  </clef>\n</attributes>'
    result = strip_musicxml_header(xml)
    assert '<clef>' in result
    assert '<sign>G</sign>' in result


def test_keeps_rest():
    xml = '<note>\n  <rest measure="yes" />\n  <duration>30240</duration>\n</note>'
    result = strip_musicxml_header(xml)
    assert '<rest' in result
    assert '<duration>30240</duration>' in result
