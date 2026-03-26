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
    'slide01-title.html',
    'slide02-vintage-scan.html',
    'slide03-approaches.html',
    'slide04-zero-shot.html',
    'slide05-datasets.html',
    'slide06-challenge.html',
    'slide07-pipeline.html',
    'slide07b-pipeline-detail.html',
    'slide07c-data-example.html',
    'slide07d-data-challenges.html',
    'slide08-bottleneck.html',
    'slide09-mxc.html',
    'slide10-comparison.html',
    'slide11-breakthrough.html',
    'slide12-example.html',
    'slide13-lora.html',
    'slide14-findings.html',
    'slide15-next.html',
    'slide16-resources.html',
  ];

  const slideResults = {};
  for (const file of slides) {
    console.log(`Processing ${file}...`);
    const result = await html2pptx(path.join(slideDir, file), pptx);
    slideResults[file] = result;
  }

  const hdr = { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 };
  const best = { bold: true, color: 'E41159' };

  // Add table to slide 10 (model comparison)
  const s6 = slideResults['slide10-comparison.html'];
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

  // Add table to slide 13 (LoRA rank)
  const s9 = slideResults['slide13-lora.html'];
  if (s9.placeholders.length > 0) {
    const p = s9.placeholders[0];
    s9.slide.addTable([
      [
        { text: 'LoRA Rank', options: hdr },
        { text: 'Trainable Params', options: hdr },
        { text: 'Pitch Similarity', options: hdr },
        { text: 'Rhythm Similarity', options: hdr },
      ],
      ['r=16', '51M (0.54%)', '33%', '32%'],
      [
        { text: 'r=32', options: best },
        { text: '102M (1.07%)', options: best },
        { text: '36%', options: best },
        { text: '46%', options: best },
      ],
      ['r=64', '204M (2.12%)', '32%', '46%'],
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
