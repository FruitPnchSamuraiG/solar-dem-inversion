const path = "../results/test/";
let modelList = [];

async function initCompare() {
  const select1 = document.getElementById("select1");
  const select2 = document.getElementById("select2");
  const viewMode = document.getElementById("viewMode");
  const table = document.getElementById("comparison-table");
  const res = await fetch("models.json");
  modelList = await res.json();

  for (const model of modelList) {
    const opt1 = new Option(model, model);
    const opt2 = new Option(model, model);
    // print options
    console.log(`Adding option for ${model}`);
    select1.appendChild(opt1);
    select2.appendChild(opt2);
  }

  select1.selectedIndex = 0;
  select2.selectedIndex = 1;

  [select1, select2, viewMode].forEach(select =>
    select.addEventListener("change", () =>
      render(select1.value, select2.value, viewMode.value, table)
    )
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
