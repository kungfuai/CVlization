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

  // Add table to slide 10 (model comparison)
  const s6 = slideResults['slide10-comparison.html'];
  if (s6.placeholders.length > 0) {
    const p = s6.placeholders[0];
    s6.slide.addTable([
      [
        { text: 'Model', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
        { text: 'Size', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
        { text: 'Pitched-only Sim', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
        { text: 'Rhythm', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
        { text: 'eval_loss', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
      ],
      ['DeepSeek-OCR-2', '3B', '0%', '4%', '0.315'],
      ['Gemma-3 4B', '4.4B', '1%', '11%', '0.216'],
      ['Ministral-3', '3.9B', '0%', '10%', '0.109'],
      ['Qwen3-VL 8B', '8B', '16%', '23%', '0.166'],
      ['Qwen3-VL 32B', '32B', '23%', '31%', '0.147'],
      [
        { text: 'Qwen3.5-9B r=32', options: { bold: true, color: 'E41159' } },
        { text: '9.5B', options: { bold: true, color: 'E41159' } },
        { text: '35%', options: { bold: true, color: 'E41159' } },
        { text: '46%', options: { bold: true, color: 'E41159' } },
        { text: '0.149', options: { bold: true, color: 'E41159' } },
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
        { text: 'LoRA Rank', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
        { text: 'Trainable Params', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
        { text: 'Pitch Similarity', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
        { text: 'Rhythm Similarity', options: { fill: { color: 'E41159' }, color: 'FFFFFF', bold: true, fontSize: 11 } },
      ],
      ['r=16', '51M (0.54%)', '33%', '32%'],
      [
        { text: 'r=32', options: { bold: true, color: 'E41159' } },
        { text: '102M (1.07%)', options: { bold: true, color: 'E41159' } },
        { text: '36%', options: { bold: true, color: 'E41159' } },
        { text: '46%', options: { bold: true, color: 'E41159' } },
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
