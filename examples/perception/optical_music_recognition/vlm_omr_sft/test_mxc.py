"""Unit tests for MXC (MusicXML Compact) format."""
import pytest
import xml.etree.ElementTree as ET
from mxc import xml_to_mxc, mxc_to_xml


def _xml_equal(a: str, b: str) -> bool:
    """Compare two XML strings semantically (ignoring whitespace differences)."""
    def normalize(el):
        """Sort children and strip whitespace for comparison."""
        el.text = (el.text or "").strip() or None
        el.tail = (el.tail or "").strip() or None
        for child in el:
            normalize(child)
    tree_a = ET.fromstring(a)
    tree_b = ET.fromstring(b)
    normalize(tree_a)
    normalize(tree_b)
    return ET.tostring(tree_a, encoding="unicode") == ET.tostring(tree_b, encoding="unicode")


def _roundtrip(xml: str) -> str:
    """Convert XML → MXC → XML and return the result."""
    mxc = xml_to_mxc(xml)
    return mxc_to_xml(mxc)


# ── Forward: xml_to_mxc ─────────────────────────────────────────────────────

class TestXmlToMxc:
    def test_header(self):
        xml = '<score-partwise version="4.0"><work><work-title>Test</work-title></work></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert 'header work-title="Test"' in mxc

    def test_part_list(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Voice</part-name><part-abbreviation>Ob</part-abbreviation></score-part></part-list></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "P1 Voice Ob" in mxc

    def test_simple_rest(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><rest/><duration>10080</duration><type>quarter</type></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "R q 10080" in mxc

    def test_whole_measure_rest(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><rest measure="yes"/><duration>30240</duration></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "R whole 30240" in mxc

    def test_pitched_note(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type><stem>up</stem></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "N C4 q 10080 su" in mxc

    def test_flat_note(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>B</step><alter>-1</alter><octave>3</octave></pitch><duration>5040</duration><type>eighth</type></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "N Bb3 e 5040" in mxc

    def test_sharp_note(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>F</step><alter>1</alter><octave>5</octave></pitch><duration>5040</duration><type>eighth</type></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "N F#5 e 5040" in mxc

    def test_accidental(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>B</step><octave>3</octave></pitch><duration>5040</duration><type>eighth</type><accidental>natural</accidental></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "acc=natural" in mxc

    def test_beam(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>5040</duration><type>eighth</type><beam number="1">begin</beam></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "bm=begin" in mxc

    def test_lyric(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type><lyric number="1"><syllabic>single</syllabic><text>Lord,</text></lyric></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "L1:s:Lord," in mxc

    def test_attributes_inline(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><attributes><divisions>10080</divisions><key><fifths>-1</fifths></key><time><beats>3</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "M 1 div=10080 key=-1 time=3/4 clef=G2" in mxc

    def test_direction_with_text(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><direction placement="above"><direction-type><words font-size="12" font-weight="bold">Andante</words></direction-type></direction></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "dir @above" in mxc
        assert "Andante" in mxc

    def test_barline(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><barline location="right"><bar-style>light-light</bar-style></barline></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "bar=light-light" in mxc

    def test_chord(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type></note><note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "+N E4" in mxc

    def test_tie(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type><tie type="start"/></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "tie=start" in mxc

    def test_dot(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>15120</duration><type>quarter</type><dot/></note></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "dot" in mxc

    def test_backup_forward(self):
        xml = '<score-partwise version="4.0"><part id="P1"><measure number="1"><backup><duration>10080</duration></backup><forward><duration>5040</duration></forward></measure></part></score-partwise>'
        mxc = xml_to_mxc(xml)
        assert "bak 10080" in mxc
        assert "fwd 5040" in mxc


# ── Round-trip: XML → MXC → XML ─────────────────────────────────────────────

class TestRoundtrip:
    def test_simple_rest_measure(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list><part id="P1"><measure number="1"><note><rest measure="yes"/><duration>30240</duration></note></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))

    def test_pitched_note_with_stem(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type><stem>up</stem></note></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))

    def test_note_with_flat_and_lyric(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list><part id="P1"><measure number="6"><note><pitch><step>B</step><alter>-1</alter><octave>3</octave></pitch><duration>5040</duration><type>eighth</type><accidental>flat</accidental><stem>up</stem><beam number="1">begin</beam><lyric number="1"><syllabic>single</syllabic><text>for</text></lyric></note></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))

    def test_attributes(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list><part id="P1"><measure number="1"><attributes><divisions>10080</divisions><key><fifths>-1</fifths></key><time><beats>3</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes><note><rest measure="yes"/><duration>30240</duration></note></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))

    def test_two_parts(self):
        xml = '<score-partwise version="4.0"><work><work-title>Test</work-title></work><part-list><score-part id="P1"><part-name>Voice</part-name><part-abbreviation>V</part-abbreviation></score-part><score-part id="P2"><part-name>Piano</part-name><part-abbreviation>Pno</part-abbreviation></score-part></part-list><part id="P1"><measure number="1"><note><rest measure="yes"/><duration>30240</duration></note></measure></part><part id="P2"><measure number="1"><note><rest measure="yes"/><duration>30240</duration></note></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))

    def test_chord(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type></note><note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type></note></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))

    def test_barline_with_repeat(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list><part id="P1"><measure number="4"><barline location="right"><bar-style>light-heavy</bar-style><repeat direction="backward"/></barline></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))

    def test_backup_forward(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>10080</duration><type>quarter</type></note><backup><duration>10080</duration></backup><note><pitch><step>E</step><octave>3</octave></pitch><duration>10080</duration><type>quarter</type></note></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))

    def test_tie_and_dot(self):
        xml = '<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list><part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch><duration>15120</duration><type>quarter</type><dot/><tie type="start"/></note></measure></part></score-partwise>'
        assert _xml_equal(xml, _roundtrip(xml))


# ── Compression ratio ────────────────────────────────────────────────────────

class TestCompression:
    SAMPLE_XML = '''<score-partwise version="4.0">
  <work><work-title>Test Song</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Voice</part-name><part-abbreviation>V</part-abbreviation></score-part>
    <score-part id="P2"><part-name>Piano</part-name><part-abbreviation>Pno</part-abbreviation></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>10080</divisions>
        <key><fifths>-1</fifths></key>
        <time><beats>3</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <direction placement="above">
        <direction-type>
          <words font-size="12" font-weight="bold">Andante</words>
        </direction-type>
      </direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>10080</duration>
        <type>quarter</type>
        <stem>up</stem>
        <lyric number="1"><syllabic>single</syllabic><text>Lord,</text></lyric>
      </note>
      <note>
        <pitch><step>B</step><alter>-1</alter><octave>3</octave></pitch>
        <duration>5040</duration>
        <type>eighth</type>
        <accidental>flat</accidental>
        <stem>up</stem>
        <beam number="1">begin</beam>
        <lyric number="1"><syllabic>single</syllabic><text>for</text></lyric>
      </note>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>5040</duration>
        <type>eighth</type>
        <stem>up</stem>
        <beam number="1">end</beam>
        <lyric number="1"><syllabic>begin</syllabic><text>to</text></lyric>
      </note>
    </measure>
    <measure number="2">
      <note><rest measure="yes"/><duration>30240</duration></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes>
        <divisions>10080</divisions>
        <key><fifths>-1</fifths></key>
        <time><beats>3</beats><beat-type>4</beat-type></time>
        <clef><sign>F</sign><line>4</line></clef>
      </attributes>
      <note>
        <pitch><step>F</step><octave>3</octave></pitch>
        <duration>10080</duration>
        <type>quarter</type>
        <stem>down</stem>
      </note>
      <note>
        <pitch><step>E</step><octave>3</octave></pitch>
        <duration>10080</duration>
        <type>quarter</type>
        <stem>down</stem>
      </note>
      <note>
        <pitch><step>F</step><octave>3</octave></pitch>
        <duration>10080</duration>
        <type>quarter</type>
        <stem>down</stem>
      </note>
    </measure>
    <measure number="2">
      <note><rest measure="yes"/><duration>30240</duration></note>
    </measure>
  </part>
</score-partwise>'''

    def test_compression_at_least_3x(self):
        mxc = xml_to_mxc(self.SAMPLE_XML)
        ratio = len(self.SAMPLE_XML) / len(mxc)
        assert ratio > 3, f"Compression ratio {ratio:.1f}x is below 3x target"

    def test_roundtrip_sample(self):
        assert _xml_equal(self.SAMPLE_XML, _roundtrip(self.SAMPLE_XML))
