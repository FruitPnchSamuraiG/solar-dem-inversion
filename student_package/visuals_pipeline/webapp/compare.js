const path = "../results/test/";

const RUN_LABELS = {
  bp_solver: "BP solver reference",
  enet_solver: "ENet solver reference",
  bp_mlp6_h232: "Label-free MLP6 (176k)",
  enet_mlp6_h232: "Label-free MLP6 (176k)",
  bp_supervised: "Supervised model",
  enet_supervised: "Supervised model",
};

const RESULTS = {
  bp: {
    title: "BP reference",
    rows: [
      ["Full", 4.191, 20.16, 0.0845, 0.906, 14.34, 0.1334],
      ["Bright", 40.901, 19.82, 0.0785, 8.657, 8.92, 0.0546],
      ["Quiet", 0.0583, 20.53, 0.0853, 0.0335, 20.13, 0.1430],
    ],
  },
  enet: {
    title: "ENet reference (alpha=0.001, L1 ratio=0.5)",
    rows: [
      ["Full", 0.588, 13.45, 0.1176, 0.318, 18.00, 0.1210],
      ["Bright", 4.239, 8.44, 0.0717, 1.926, 14.47, 0.0607],
      ["Quiet", 0.159, 17.23, 0.1230, 0.1296, 20.66, 0.1281],
    ],
  },
};

let openZoomViewer;

function setupZoomViewer() {
  const modal = document.getElementById("zoom-modal");
  const canvas = document.getElementById("zoom-canvas");
  const image = document.getElementById("zoom-image");
  const title = document.getElementById("zoom-title");
  const close = document.getElementById("zoom-close");
  let scale = 1;
  let x = 0;
  let y = 0;
  let dragging = false;
  let pointerX = 0;
  let pointerY = 0;

  const apply = () => {
    image.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
  };
  const fit = () => {
    if (!image.naturalWidth || !image.naturalHeight) return;
    const padding = 12;
    scale = Math.min(
      (canvas.clientWidth - padding * 2) / image.naturalWidth,
      (canvas.clientHeight - padding * 2) / image.naturalHeight,
    );
    x = (canvas.clientWidth - image.naturalWidth * scale) / 2;
    y = (canvas.clientHeight - image.naturalHeight * scale) / 2;
    apply();
  };
  const hide = () => {
    modal.hidden = true;
    image.removeAttribute("src");
  };

  openZoomViewer = (src, label) => {
    title.textContent = label;
    modal.hidden = false;
    image.onload = fit;
    image.src = src;
  };

  close.addEventListener("click", hide);
  modal.addEventListener("click", event => {
    if (event.target === modal) hide();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !modal.hidden) hide();
  });
  window.addEventListener("resize", () => {
    if (!modal.hidden) fit();
  });
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const cursorX = event.clientX - rect.left;
    const cursorY = event.clientY - rect.top;
    const imageX = (cursorX - x) / scale;
    const imageY = (cursorY - y) / scale;
    const factor = event.deltaY < 0 ? 1.18 : 1 / 1.18;
    scale = Math.max(0.05, Math.min(12, scale * factor));
    x = cursorX - imageX * scale;
    y = cursorY - imageY * scale;
    apply();
  }, { passive: false });
  canvas.addEventListener("pointerdown", event => {
    dragging = true;
    pointerX = event.clientX;
    pointerY = event.clientY;
    canvas.classList.add("dragging");
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (!dragging) return;
    x += event.clientX - pointerX;
    y += event.clientY - pointerY;
    pointerX = event.clientX;
    pointerY = event.clientY;
    apply();
  });
  canvas.addEventListener("pointerup", event => {
    dragging = false;
    canvas.classList.remove("dragging");
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointercancel", () => {
    dragging = false;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("dblclick", fit);
}

async function initCompare() {
  const select1 = document.getElementById("select1");
  const date = document.getElementById("date");
  const viewMode = document.getElementById("viewMode");
  const table = document.getElementById("comparison-table");
  const res = await fetch("models.json", {cache: "no-store"});
  if (!res.ok) throw new Error(`Cannot load model list (HTTP ${res.status}).`);
  const entries = await res.json();
  const available = new Set(entries);

  const runs = [...new Set(entries.map(entry => entry.split("/")[0]))];
  const references = runs.filter(run => run.endsWith("_solver"));
  references.forEach(run => select1.appendChild(new Option(RUN_LABELS[run] || run, run)));

  // One shared date makes a direct side-by-side comparison unambiguous.
  const query = new URLSearchParams(window.location.search);
  const choose = (select, requested, fallback) => {
    const values = [...select.options].map(option => option.value);
    const index = requested ? values.indexOf(requested) : fallback;
    select.selectedIndex = index >= 0 ? index : fallback;
  };
  choose(select1, query.get("solver"), 0);
  const selectedRuns = () => {
    const prefix = select1.value.replace(/_solver$/, "");
    return [select1.value, `${prefix}_mlp6_h232`, `${prefix}_supervised`];
  };
  const refreshDates = (requested) => {
    const dates = entries
      .filter(entry => entry.startsWith(`${select1.value}/`))
      .map(entry => entry.split("/")[1])
      .filter(stamp => selectedRuns().every(run => available.has(`${run}/${stamp}`)));
    date.replaceChildren();
    [...new Set(dates)].sort().forEach(stamp => date.appendChild(new Option(stamp, stamp)));
    choose(date, requested, 0);
  };
  refreshDates(query.get("date"));
  if ([...viewMode.options].some(option => option.value === query.get("mode"))) {
    viewMode.value = query.get("mode");
  }

  const renderSelected = () => {
    const selected = selectedRuns().map(run => `${run}/${date.value}`);
    if (!selected.every(run => available.has(run))) {
      table.innerHTML = `<tr><td class="text-center text-red-700 py-8" colspan="4">No common date is available for the solver and both models. Generate and stage all three asset sets.</td></tr>`;
      return;
    }
    const params = new URLSearchParams({solver: select1.value, date: date.value, mode: viewMode.value});
    history.replaceState(null, "", `${window.location.pathname}?${params}`);
    render(selected, viewMode.value, table);
  };
  select1.addEventListener("change", () => {
    const previousDate = date.value;
    refreshDates(previousDate);
    renderSelected();
  });
  [date, viewMode].forEach(select => select.addEventListener("change", renderSelected));

  renderSelected();
}

function setupLinkedNavigation(table) {
  const panels = [...table.querySelectorAll(".linked-panel")];
  let scale = 1, x = 0, y = 0;
  const apply = () => {
    x = Math.max(1 - scale, Math.min(0, x));
    y = Math.max(1 - scale, Math.min(0, y));
    panels.forEach(panel => {
      panel.querySelector("img").style.transform =
        `translate(${100 * x}%, ${100 * y}%) scale(${scale})`;
    });
  };
  const reset = () => { scale = 1; x = 0; y = 0; apply(); };
  document.getElementById("reset-linked-zoom").onclick = reset;
  panels.forEach(panel => {
    let drag = null;
    panel.addEventListener("wheel", event => {
      if (!event.shiftKey) return;
      event.preventDefault();
      const rect = panel.getBoundingClientRect();
      const px = (event.clientX - rect.left) / rect.width;
      const py = (event.clientY - rect.top) / rect.height;
      const next = Math.max(1, Math.min(12, scale * (event.deltaY < 0 ? 1.18 : 1 / 1.18)));
      x = px - (px - x) * next / scale;
      y = py - (py - y) * next / scale;
      scale = next;
      apply();
    }, {passive: false});
    panel.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      drag = {x: event.clientX, y: event.clientY};
      panel.setPointerCapture(event.pointerId);
    });
    panel.addEventListener("pointermove", event => {
      if (!drag) return;
      const rect = panel.getBoundingClientRect();
      x += (event.clientX - drag.x) / rect.width;
      y += (event.clientY - drag.y) / rect.height;
      drag = {x: event.clientX, y: event.clientY};
      apply();
    });
    const stop = event => {
      drag = null;
      if (panel.hasPointerCapture(event.pointerId)) panel.releasePointerCapture(event.pointerId);
    };
    panel.addEventListener("pointerup", stop);
    panel.addEventListener("pointercancel", stop);
    panel.addEventListener("lostpointercapture", () => { drag = null; });
    panel.addEventListener("dblclick", reset);
  });
}

function render(models, mode, table) {
  table.innerHTML = "";
  const resultsContent = document.getElementById("results-content");
  resultsContent.replaceChildren();
  resultsContent.hidden = mode !== "results";
  table.hidden = mode === "results";
  document.getElementById("reset-linked-zoom").parentElement.hidden = mode === "results";
  const columns = models.map(model => ({
    model, label: RUN_LABELS[model.split("/")[0]], subtitle: "", observed: false,
  }));
  if (mode === "aia") {
    columns.shift();
    columns.forEach(column => { column.subtitle = "reconstruction vs observed AIA"; });
    columns.unshift({model: models[0], label: "Observed AIA", subtitle: "preprocessed measurement", observed: true});
  } else if (mode === "jpdfs") {
    columns.shift();
    columns.forEach(column => { column.subtitle = "reconstruction vs observed AIA"; });
  } else {
    columns.forEach(column => { column.subtitle = "DEM estimate"; });
  }
  document.getElementById("view-description").textContent = {
    dems: "Three DEM estimates at the same date and solar location.",
    aia: "Measured AIA followed by label-free and supervised reconstructions from their DEMs, using the observed-image colour scale for each channel.",
    jpdfs: "Horizontal axis: observed AIA. Vertical axis: reconstructed AIA. Colour shows pixel counts; the diagonal marks agreement. Axes are logarithmic. Linked zoom matches image positions; plot axis limits may differ.",
    results: "Shared-test DEM metrics and selected-date AIA reconstruction errors. Lower is better for every metric.",
  }[mode];
  if (mode === "results") {
    renderResults(models, resultsContent);
    return;
  }
  const header = table.createTHead().insertRow();
  for (const column of [{label: "", subtitle: ""}, ...columns]) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.className = "text-center align-bottom px-2 font-medium";
    cell.innerHTML = column.label
      ? `<div>${column.label}</div><div class="text-xs font-normal text-gray-500 mt-1">${column.subtitle}</div>`
      : "";
    header.appendChild(cell);
  }
  const body = table.createTBody();

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
    `;

    body.appendChild(row);
    const label = mode === "aia" || mode === "jpdfs"
      ? aiaLabels[i]
      : demLabels[i];
    for (const column of columns) {
      const cell = row.insertCell();
      cell.className = "px-1";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "block text-sm underline mt-2";
      button.textContent = "Open full image";
      const title = `${column.label} — ${label}`;
      button.setAttribute("aria-label", `Zoom ${title}`);
      const img = document.createElement("img");
      const filename = column.observed ? `aia_${i}.png` : name;
      const src = `${path}${column.model}/${filename}`;
      img.src = src;
      img.alt = title;
      img.width = 400;
      img.height = 400;
      img.className = "w-full border shadow";
      img.draggable = false;
      const panel = document.createElement("div");
      panel.className = "linked-panel";
      panel.appendChild(img);
      cell.appendChild(panel);
      button.addEventListener("click", () => openZoomViewer(src, title));
      cell.appendChild(button);
    }
  }
  setupLinkedNavigation(table);
}

function makeResultsTable(title, headers, rows) {
  const section = document.createElement("section");
  section.className = "results-section";
  const heading = document.createElement("h2");
  heading.className = "serif text-2xl text-center mb-3";
  heading.textContent = title;
  section.appendChild(heading);
  const table = document.createElement("table");
  table.className = "results-table";
  const thead = table.createTHead().insertRow();
  headers.forEach(header => {
    const cell = document.createElement("th");
    cell.className = "border-b border-gray-300 px-3 py-2 text-center font-medium";
    cell.textContent = header;
    thead.appendChild(cell);
  });
  const body = table.createTBody();
  rows.forEach(values => {
    const row = body.insertRow();
    values.forEach((value, index) => {
      const cell = row.insertCell();
      cell.className = "border-b border-gray-200 px-3 py-2 text-center";
      cell.textContent = index === 0 ? value : value;
    });
  });
  section.appendChild(table);
  return section;
}

async function renderResults(models, container) {
  container.innerHTML = "";
  const generation = Symbol();
  container.resultsGeneration = generation;
  const intro = document.createElement("section");
  intro.className = "results-section";
  intro.innerHTML = `
    <h2 class="serif text-3xl mb-4">Evaluation results</h2>
    <p class="mb-4">The DEM tables summarize the full shared test set: 153 timestamps and 48,960 blocks, including five solver targets per spatial block. They do not change with the selected viewer date. Predictions are compared with the corresponding BP or ENet solver reference, not a directly measured true DEM.</p>
    <dl class="space-y-3 mb-5">
      <div><dt class="font-bold">DEM MSE ↓</dt><dd>Mean squared difference between predicted and reference DEM values, averaged across valid pixels and 18 temperature bins. Large errors receive more weight.</dd></div>
      <div><dt class="font-bold">EM relative error (%) ↓</dt><dd>Sum of absolute errors in each pixel’s total emission, divided by the sum of reference emission, multiplied by 100. This is a ratio of totals, not an average of pixel percentages.</dd></div>
      <div><dt class="font-bold">W1 (dex) ↓</dt><dd>Average temperature-distribution distance between DEM curves normalized to unit total emission. Lower values indicate closer thermal shapes. Only pixels with positive emission in both curves are included.</dd></div>
      <div><dt class="font-bold">AIA MAE and MSE ↓</dt><dd>Mean absolute and mean squared differences between reconstructed and observed AIA brightness. MAE is in DN/s and MSE in (DN/s)². The frame diagnostics below average across six channels and image pixels.</dd></div>
    </dl>
    <p>Full includes all valid pixels. Bright means at least one AIA channel reaches its fixed brightness threshold; Quiet is the remaining valid population. Lower is better for all metrics. Supervised models learn solver labels; label-free MLP6 models learn physical objectives.</p>
    <p class="mt-3">Full-test AIA reconstruction evaluation is pending. Until it is available, the AIA section below shows selected-frame diagnostics only.</p>`;
  container.appendChild(intro);
  for (const prefix of ["bp", "enet"]) {
    const result = RESULTS[prefix];
    const rows = result.rows.flatMap(row => [
      ["Supervised", row[0], row[4].toFixed(4), row[5].toFixed(2), row[6].toFixed(4)],
      ["Label-free MLP6", row[0], row[1].toFixed(4), row[2].toFixed(2), row[3].toFixed(4)],
    ]);
    container.appendChild(makeResultsTable(
      `${result.title} — full shared test set`,
      ["Model", "Pixels", "DEM MSE ↓", "EM error (%) ↓", "W1 (dex) ↓"],
      rows,
    ));
  }
  const date = document.getElementById("date").value;
  const section = makeResultsTable(
    `AIA reconstruction — selected frame only: ${date}`,
    ["Track", "Model", "AIA MAE ↓", "AIA MSE ↓"],
    [],
  );
  const note = document.createElement("p");
  note.className = "text-gray-600 my-3";
  note.textContent = "These are saved full-frame diagnostics, not test-set averages or a shared finite-pixel-mask evaluation. Nonfinite or inaccessible metrics are shown as unavailable.";
  section.insertBefore(note, section.querySelector("table"));
  container.appendChild(section);
  const tailNote = document.createElement("section");
  tailNote.className = "results-section leading-relaxed";
  tailNote.innerHTML = `<h2 class="serif text-2xl mb-3">Interpreting large DEM errors</h2>
    <p>Squared error is strongly concentrated in a small upper tail, especially among bright pixels. For label-free BP, the worst 1% of pixels contribute about 97.80% of total squared error; for label-free ENet, 66.75%. These errors remain part of the reported means—they are not discarded.</p>
    <p class="mt-3">The approximate median per-pixel DEM squared error (summed over 18 bins) is 0.02618 for label-free BP versus 0.11022 for supervised BP: a lower median despite a higher mean MSE. For ENet, the corresponding medians are 1.39589 versus 1.12525, so supervised ENet has the lower median as well as the lower mean MSE. Divide these median sums by 18 to express them as per-pixel MSE. A lower median does not remove the importance of large tail errors.</p>`;
  container.appendChild(tailNote);
  const runs = ["bp_mlp6_h232", "bp_supervised", "enet_mlp6_h232", "enet_supervised"];
  const rows = runs.map(run => {
    const row = section.querySelector("tbody").insertRow();
    [run.startsWith("bp_") ? "BP" : "ENet", RUN_LABELS[run], "Loading…", "Loading…"].forEach(value => {
      row.insertCell().textContent = value;
    });
    return row;
  });
  await Promise.all(runs.map(async (run, index) => {
    let values = ["Unavailable", "Unavailable"];
    try {
      const response = await fetch(`${path}${run}/${date}/metrics.json`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const metrics = await response.json();
      values = [metrics.mae, metrics.mse].map(value =>
        typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "Unavailable");
    } catch (_) {}
    if (container.resultsGeneration !== generation || !section.isConnected) return;
    rows[index].cells[2].textContent = values[0];
    rows[index].cells[3].textContent = values[1];
  }));
}
setupZoomViewer();
initCompare().catch(error => {
  document.getElementById("view-description").textContent =
    `Viewer could not load: ${error.message}`;
});
