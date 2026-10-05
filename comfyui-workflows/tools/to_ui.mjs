// Converts a workflow spec (API prompt + layout) into a drag-and-drop ComfyUI UI workflow by
// building it inside the real ComfyUI frontend, then round-trips it to prove the UI file
// produces exactly the same API prompt.
//
// Usage: node to_ui.mjs <spec.json> <out_ui.json> [comfy_url]
//   spec.json = { "prompt": {...API format...},
//                 "widths": { "<node id>": px },
//                 "rows": [[ block, ... ], ...] }
//   block = { "title", "color", "columns": [["<node id>", ...], ...] }   -> a group
//         | { "note": { "title", "text", "width", "height" } }           -> a free-standing Markdown note
//   Nodes are stacked top-to-bottom inside each column using their real rendered sizes,
//   columns left-to-right inside a group, blocks left-to-right inside a row, rows top-to-bottom.
// Requires a running ComfyUI with every custom node pack the workflow uses, and Playwright.
import { chromium } from 'playwright';
import fs from 'node:fs';

const [specPath, outPath, url = 'http://127.0.0.1:8188/'] = process.argv.slice(2);
if (!specPath || !outPath) {
  console.error('Usage: node to_ui.mjs <spec.json> <out_ui.json> [comfy_url]');
  process.exit(2);
}
const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
const page = await browser.newPage();
const pageErrors = [];
page.on('pageerror', (e) => pageErrors.push(e.message));
await page.goto(url, { waitUntil: 'networkidle' });
await page.waitForFunction(() => window.app && window.app.graph, null, { timeout: 60000 });

const result = await page.evaluate(async (spec) => {
  const app = window.app;
  const LG = window.LiteGraph;
  await app.loadApiJson(spec.prompt, 'workflow');
  const graph = app.graph;
  const byId = (id) => graph.getNodeById(id) ?? graph.getNodeById(Number(id));

  const TITLE = LG.NODE_TITLE_HEIGHT, GAP_X = 40, GAP_Y = 24, PAD = 24, HEADER = 56, BLOCK_GAP = 90, ROW_GAP = 110;
  const placed = new Set();
  for (const [id, w] of Object.entries(spec.widths ?? {})) {
    const n = byId(id);
    if (!n) throw new Error(`widths references missing node ${id}`);
    n.size = [Math.max(w, n.computeSize()[0]), n.size[1]];
  }

  let rowY = 0;
  for (const row of spec.rows) {
    let blockX = 0, rowH = 0;
    for (const block of row) {
      if (block.note) {
        const n = LG.createNode('MarkdownNote');
        n.title = block.note.title;
        n.widgets.find((w) => w.name === 'text').value = block.note.text;
        n.color = '#432';
        n.bgcolor = '#653';
        graph.add(n);
        n.pos = [blockX, rowY + TITLE];
        n.size = [block.note.width, block.note.height];
        blockX += block.note.width + BLOCK_GAP;
        rowH = Math.max(rowH, block.note.height + TITLE);
        continue;
      }
      let colX = blockX + PAD, groupH = 0;
      for (const column of block.columns) {
        const nodes = column.map((id) => {
          const n = byId(id);
          if (!n) throw new Error(`group "${block.title}" references missing node ${id}`);
          if (placed.has(String(id))) throw new Error(`node ${id} placed twice`);
          placed.add(String(id));
          return n;
        });
        const colW = Math.max(...nodes.map((n) => n.size[0]));
        let y = rowY + HEADER + TITLE;
        for (const n of nodes) {
          n.size = [colW, n.size[1]];
          n.pos = [colX, y];
          y += n.size[1] + TITLE + GAP_Y;
        }
        groupH = Math.max(groupH, y - GAP_Y - TITLE - rowY);
        colX += colW + GAP_X;
      }
      const grp = new LG.LGraphGroup(block.title);
      grp.pos = [blockX, rowY];
      grp.size = [colX - GAP_X + PAD - blockX, groupH + PAD];
      if (block.color) grp.color = block.color;
      graph.add(grp);
      blockX += grp.size[0] + BLOCK_GAP;
      rowH = Math.max(rowH, grp.size[1]);
    }
    rowY += rowH + ROW_GAP;
  }
  const unplaced = Object.keys(spec.prompt).filter((id) => !placed.has(String(id)));
  if (unplaced.length) throw new Error(`nodes missing from layout: ${unplaced.join(', ')}`);

  const ui = graph.serialize();
  // Round trip: reload the serialized UI workflow from scratch and convert it back to an API prompt.
  await app.loadGraphData(structuredClone(ui), true, true, 'roundtrip');
  const { output } = await app.graphToPrompt();
  return { ui, roundtrip: output };
}, spec);

await browser.close();

// Compare the round-tripped prompt with the source prompt (inputs only; _meta titles are cosmetic).
const sortKeys = (o) => Object.fromEntries(Object.entries(o).sort(([a], [b]) => a.localeCompare(b)));
const norm = (p) => Object.fromEntries(Object.entries(p).map(([id, n]) => [String(id), { class_type: n.class_type, inputs: sortKeys(n.inputs) }]));
const want = norm(spec.prompt), got = norm(result.roundtrip);
const problems = [];
for (const id of new Set([...Object.keys(want), ...Object.keys(got)])) {
  const a = JSON.stringify(want[id]), b = JSON.stringify(got[id]);
  if (a !== b) problems.push(`node ${id}:\n  spec:      ${a}\n  roundtrip: ${b}`);
}
if (pageErrors.length) problems.push('page errors: ' + pageErrors.join(' | '));
if (problems.length) {
  console.error('Round-trip mismatch, UI workflow NOT written:\n' + problems.join('\n'));
  process.exit(1);
}
fs.writeFileSync(outPath, JSON.stringify(result.ui, null, 2));
console.log(`OK: ${outPath} (${result.ui.nodes.length} nodes, ${result.ui.links.length} links, round-trip identical)`);
