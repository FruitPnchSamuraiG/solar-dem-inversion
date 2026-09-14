const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class Select {
  options = []; selectedIndex = -1; listeners = {};
  get value() { return this.options[this.selectedIndex]?.value || ''; }
  set value(value) { this.selectedIndex = this.options.findIndex(x => x.value === value); }
  appendChild(option) { this.options.push(option); if (this.selectedIndex < 0) this.selectedIndex = 0; }
  replaceChildren() { this.options = []; this.selectedIndex = -1; }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  change(value) { this.value = value; this.listeners.change(); }
}

async function check(entries, query) {
  const elements = Object.fromEntries(['select1', 'select2', 'date', 'viewMode'].map(id => [id, new Select()]));
  elements['comparison-table'] = {};
  elements.viewMode.appendChild({value: 'dems'});
  const ctx = vm.createContext({
    URLSearchParams,
    Option: function(text, value) { this.text = text; this.value = value; },
    document: {getElementById: id => elements[id]},
    window: {location: {search: query, pathname: '/webapp/compare.html'}},
    history: {replaceState() {}},
    fetch: async () => ({json: async () => entries}),
  });
  const file = path.join(__dirname, '../student_package/visuals_pipeline/webapp/compare.js');
  const source = fs.readFileSync(file, 'utf8').replace(/setupZoomViewer\(\);\s*initCompare\(\);\s*$/, '');
  vm.runInContext(source, ctx);
  vm.runInContext('render = () => {};', ctx);
  await vm.runInContext('initCompare()', ctx);
  return elements;
}

(async () => {
  const entries = [];
  for (const track of ['bp', 'enet']) {
    for (const run of ['solver', 'mlp6_h232', 'supervised']) {
      entries.push(`${track}_${run}/20150923_180356`);
      if (run !== 'supervised') entries.push(`${track}_${run}/20151022_053428`);
    }
  }
  const e = await check(entries, '?solver=bp_solver&model=bp_supervised&date=20151022_053428');
  assert.equal(e.select2.value, 'bp_supervised');
  assert.equal(e.date.value, '20150923_180356');
  assert.equal(e.date.options.length, 1);
  assert.deepEqual(e.select2.options.map(x => x.value), ['bp_mlp6_h232', 'bp_supervised']);
  e.select1.change('enet_solver');
  assert.equal(e.select2.value, 'enet_supervised');
  e.select2.change('enet_mlp6_h232');
  assert.equal(e.date.options.length, 2);
  e.date.change('20151022_053428');
  e.select1.change('bp_solver');
  assert.equal(e.select2.value, 'bp_mlp6_h232');
  assert.equal(e.date.value, '20151022_053428');
  const old = await check(entries.filter(x => !x.includes('_supervised')), '?solver=enet_solver&model=bp_supervised');
  assert.equal(old.select2.value, 'enet_mlp6_h232');
  console.log('Viewer dropdown checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
