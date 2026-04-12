const pptxgen = require('pptxgenjs');
const path = require('path');
const html2pptx = require(path.join(process.env.HOME, '.claude/skills/pptx/scripts/html2pptx'));

const slideDir = __dirname;

async function build() {
  const pptx = new pptxgen();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'KungFu AI';
  pptx.title = 'Optical Music Recognition with VLMs';

  const slides = [
    's01-title.html',
    's02-challenge.html',
    's03-pipeline.html',
    's04-bottleneck.html',
    's05-mxc.html',
    's06-comparison.html',
    's07-breakthrough.html',
    's08-example.html',
    's09-lora.html',
    's10-findings.html',
    's11-next.html',
    's12-resources.html',
  ];

  const slideResults = {};
  for (const file of slides) {
    console.log(`Processing ${file}...`);
    const result = await html2pptx(path.join(slideDir, file), pptx);
    slideResults[file] = result;
  }

  const hdr = { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 };
  const best = { bold: true, color: 'E41159' };

  // Slide 6: Model comparison table
  const s6 = slideResults['s06-comparison.html'];
  if (s6.placeholders.length > 0) {
    const p = s6.placeholders[0];
    s6.slide.addTable([
      [
        { text: 'Model', options: hdr },
        { text: 'Size', options: hdr },
        { text: 'Pitched-only Sim', options: hdr },
        { text: 'Rhythm', options: hdr },
        { text: 'eval_loss', options: hdr },
      ],
      ['DeepSeek-OCR-2', '3B', '0%', '4%', '0.315'],
      ['Gemma-3 4B', '4.4B', '1%', '11%', '0.216'],
      ['Ministral-3', '3.9B', '0%', '10%', '0.109'],
      ['Qwen3-VL 8B', '8B', '16%', '23%', '0.166'],
      ['Qwen3-VL 32B', '32B', '23%', '31%', '0.147'],
      [
        { text: 'Qwen3.5-9B r=32', options: best },
        { text: '9.5B', options: best },
        { text: '35%', options: best },
        { text: '46%', options: best },
        { text: '0.149', options: best },
      ],
    ], {
      x: p.x, y: p.y, w: p.w, h: p.h,
      border: { pt: 1, color: 'D9D5D2' },
      align: 'center',
      valign: 'middle',
      fontSize: 10,
      colW: [2.2, 0.8, 1.8, 1.2, 1.2],
    });
  }

  // Slide 9: LoRA rank table
  const s9 = slideResults['s09-lora.html'];
  if (s9.placeholders.length > 0) {
    const p = s9.placeholders[0];
    s9.slide.addTable([
      [
        { text: 'LoRA Rank', options: hdr },
        { text: 'Trainable Params', options: hdr },
        { text: 'Pitch Similarity', options: hdr },
        { text: 'Rhythm Similarity', options: hdr },
      ],
      ['r=16', '51M', '33%', '32%'],
      [
        { text: 'r=32', options: best },
        { text: '102M', options: best },
        { text: '36%', options: best },
        { text: '46%', options: best },
      ],
      ['r=64', '204M', '32%', '46%'],
    ], {
      x: p.x, y: p.y, w: p.w, h: p.h,
      border: { pt: 1, color: 'D9D5D2' },
      align: 'center',
      valign: 'middle',
      fontSize: 11,
      colW: [1.8, 2.2, 1.8, 1.8],
    });
  }

  const outPath = path.join(slideDir, 'omr-mxc-results.pptx');
  await pptx.writeFile({ fileName: outPath });
  console.log(`Created: ${outPath}`);
}

build().catch(console.error);
