const path = "../results/test/";

const RUN_LABELS = {
  bp_solver: "BP solver reference",
  enet_solver: "ENet solver reference",
  bp_mlp6_h232: "BP mlp6 (h232, 176k)",
  enet_mlp6_h232: "ENet mlp6 (h232, 176k)",
};

async function initCompare() {
  const select1 = document.getElementById("select1");
  const select2 = document.getElementById("select2");
  const date = document.getElementById("date");
  const viewMode = document.getElementById("viewMode");
  const table = document.getElementById("comparison-table");
  const res = await fetch("models.json");
  const entries = await res.json();
  const available = new Set(entries);

  const runs = [...new Set(entries.map(entry => entry.split("/")[0]))];
  const references = runs.filter(run => run.endsWith("_solver"));
  const models = runs.filter(run => run.includes("_mlp6_"));
  const dates = [...new Set(entries.map(entry => entry.split("/")[1]))].sort();
  references.forEach(run => select1.appendChild(new Option(RUN_LABELS[run] || run, run)));
  models.forEach(run => select2.appendChild(new Option(RUN_LABELS[run] || run, run)));
  dates.forEach(stamp => date.appendChild(new Option(stamp, stamp)));

  // One shared date makes a direct side-by-side comparison unambiguous.
  const query = new URLSearchParams(window.location.search);
  const choose = (select, requested, fallback) => {
    const values = [...select.options].map(option => option.value);
    const index = requested ? values.indexOf(requested) : fallback;
    select.selectedIndex = index >= 0 ? index : fallback;
  };
  choose(select1, query.get("solver"), 0);
  choose(select2, query.get("model"), 0);
  choose(date, query.get("date"), 0);
  if ([...viewMode.options].some(option => option.value === query.get("mode"))) {
    viewMode.value = query.get("mode");
  }

  const renderSelected = () => {
    const left = `${select1.value}/${date.value}`;
    const right = `${select2.value}/${date.value}`;
    if (!available.has(left) || !available.has(right)) {
      table.innerHTML = `<tr><td class="text-center text-red-700 py-8" colspan="3">Assets for this model/date combination have not been generated.</td></tr>`;
      return;
    }
    const params = new URLSearchParams({solver: select1.value, model: select2.value, date: date.value, mode: viewMode.value});
    history.replaceState(null, "", `${window.location.pathname}?${params}`);
    render(left, right, viewMode.value, table);
  };
  [select1, select2, date, viewMode].forEach(select =>
    select.addEventListener("change", () => {
      renderSelected();
    })
  );

  renderSelected();
}

function render(leftModel, rightModel, mode, table) {
  table.innerHTML = "";

  const descriptors = {
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
