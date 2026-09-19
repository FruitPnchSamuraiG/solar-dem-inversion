const path = "../results/test/";

const RUN_LABELS = {
  bp_solver: "BP solver reference",
  enet_solver: "ENet solver reference",
  bp_mlp6_h232: "Label-free MLP6 (176k)",
  enet_mlp6_h232: "Label-free MLP6 (176k)",
  bp_supervised: "Supervised model",
  enet_supervised: "Supervised model",
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
  const res = await fetch("models.json");
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
    aia: "Measured AIA followed by our and Samuel’s reconstructions from their DEMs, using the observed-image colour scale for each channel.",
    jpdfs: "Horizontal axis: observed AIA. Vertical axis: reconstructed AIA. Colour shows pixel counts; the diagonal marks agreement. Axes are logarithmic. Linked zoom matches image positions; plot axis limits may differ.",
  }[mode];
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
setupZoomViewer();
initCompare();
