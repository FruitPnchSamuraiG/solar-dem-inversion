const path = "../results/test/";
let modelList = [];

function prettyName(entry) {
  const [run, timestamp] = entry.split("/");
  const track = run.startsWith("bp_") ? "BP" : "ENet";
  const label = run.endsWith("_solver") ? `${track} solver reference` : `Unsupervised ${track} mlp6 (h232)`;
  return `${label} — ${timestamp}`;
}

async function initCompare() {
  const select1 = document.getElementById("select1");
  const select2 = document.getElementById("select2");
  const viewMode = document.getElementById("viewMode");
  const table = document.getElementById("comparison-table");
  const res = await fetch("models.json");
  modelList = await res.json();

  const references = modelList.filter(model => model.includes("_solver/"));
  const models = modelList.filter(model => model.includes("_mlp6_"));
  for (const model of references) {
    const opt1 = new Option(prettyName(model), model);
    select1.appendChild(opt1);
  }
  for (const model of models) {
    const opt2 = new Option(prettyName(model), model);
    select2.appendChild(opt2);
  }

  // Match the deployed viewer.html convention: a shareable URL chooses both
  // assets and the initial pane, while still falling back cleanly for a fresh
  // visit.  Entries are relative result folders such as "bp_solver/201509...".
  const query = new URLSearchParams(window.location.search);
  const choose = (select, requested, fallback) => {
    const values = [...select.options].map(option => option.value);
    const index = requested ? values.indexOf(requested) : fallback;
    select.selectedIndex = index >= 0 ? index : fallback;
  };
  choose(select1, query.get("model1"), 0);
  choose(select2, query.get("model2"), 0);
  if ([...viewMode.options].some(option => option.value === query.get("mode"))) {
    viewMode.value = query.get("mode");
  }

  [select1, select2, viewMode].forEach(select =>
    select.addEventListener("change", () => {
      const params = new URLSearchParams({model1: select1.value, model2: select2.value, mode: viewMode.value});
      history.replaceState(null, "", `${window.location.pathname}?${params}`);
      render(select1.value, select2.value, viewMode.value, table);
    })
  );

  render(select1.value, select2.value, viewMode.value, table);
}

function render(leftModel, rightModel, mode, table) {
  table.innerHTML = "";

  const descriptors = {
    metrics: ["mae", "mse"],
    dems: ["mean_logt.png", "std_logt.png", ...Array.from({ length: 18 }, (_, i) => `dem_${i}.png`)],
    aia: [0, 1, 2, 3, 4, 5].map(i => `aia_${i}_resynth.png`),
    jpdfs: [0, 1, 2, 3, 4, 5].map(i => `aia_${i}_resynth_jpdf.png`)
  };

   const aiaLabels = ["94 Å", "131 Å", "171 Å", "193 Å", "211 Å", "335 Å"];
   const demLabels = [
    "Mean Log T",
    "Std Log T",
    ...Array.from({ length: 18 }, (_, i) => `logT = ${(5.5 + 0.1 * i).toFixed(1)}`)
    ];


  if (mode === "metrics") {
    Promise.all([leftModel, rightModel].map(m => fetch(`${path}${m}/metrics.json`).then(r => r.json())))
      .then(([left, right]) => {
        const rows = ["mae", "mse"].map((metric) => {
          return `
            <tr>
              <td class="pr-4 text-sm text-gray-500">${metric}</td>
              <td class="border px-4 py-2 text-center">${left[metric]}</td>
              <td class="border px-4 py-2 text-center">${right[metric]}</td>
              <td></td>
            </tr>`;
        });
        table.innerHTML = rows.join("");
      });
    return;
  }

  const files = descriptors[mode];
  for (let i = 0; i < files.length; i++) {
    const name = files[i];
    const row = document.createElement("tr");

    row.innerHTML = `
      <td class="text-sm text-gray-600">
        ${name}
        <div class="text-xs text-gray-400 mt-1">
            ${
            mode === "aia" || mode === "jpdfs"
                ? aiaLabels[i] || ""
                : mode === "dems"
                ? demLabels[i] || ""
                : ""
            }
        </div>
        </td>
      <td><a target="_blank" id="a_${i}_1"><img id="img_${i}_1" width=400 height=400 class="border shadow" /></a></td>
      <td><a target="_blank" id="a_${i}_2"><img id="img_${i}_2" width=400 height=400 class="border shadow" /></a></td>
    `;

    table.appendChild(row);

    const leftPath = `${path}${leftModel}/${name}`;
    const rightPath = `${path}${rightModel}/${name}`;

    document.getElementById(`img_${i}_1`).src = leftPath;
    document.getElementById(`img_${i}_2`).src = rightPath;
    document.getElementById(`a_${i}_1`).href = leftPath;
    document.getElementById(`a_${i}_2`).href = rightPath;
  }
}
initCompare();
