document.addEventListener("DOMContentLoaded", () => {
  const benchmarkResults = [
    { model: "HRSCD-S2", family: "CNN", second: { oa: 85.49, fscd: 49.22, miou: 64.43, sek: 10.69 }, landsat: { oa: 86.06, fscd: 36.52, miou: 74.92, sek: 2.89 } },
    { model: "HRSCD-S3", family: "CNN", second: { oa: 84.62, fscd: 51.62, miou: 66.33, sek: 11.97 }, landsat: { oa: 91.47, fscd: 75.86, miou: 79.79, sek: 35.57 } },
    { model: "HRSCD-S4", family: "CNN", second: { oa: 86.62, fscd: 58.21, miou: 71.15, sek: 18.80 }, landsat: { oa: 92.17, fscd: 77.37, miou: 81.07, sek: 38.09 } },
    { model: "ChangeMask", family: "CNN", second: { oa: 86.93, fscd: 59.74, miou: 71.46, sek: 19.50 }, landsat: { oa: 92.93, fscd: 79.74, miou: 81.46, sek: 40.50 } },
    { model: "SSCD-1", family: "CNN", second: { oa: 87.19, fscd: 61.22, miou: 72.60, sek: 21.86 }, landsat: { oa: 93.20, fscd: 80.53, miou: 81.89, sek: 41.77 } },
    { model: "Bi-SRNet", family: "CNN", second: { oa: 87.84, fscd: 62.61, miou: 73.41, sek: 23.22 }, landsat: { oa: 93.80, fscd: 82.01, miou: 82.94, sek: 44.27 } },
    { model: "TED", family: "CNN", second: { oa: 87.39, fscd: 60.34, miou: 72.79, sek: 22.17 }, landsat: { oa: 94.39, fscd: 83.63, miou: 84.79, sek: 48.33 } },
    { model: "SMNet", family: "Transformer", second: { oa: 86.68, fscd: 60.34, miou: 71.95, sek: 20.29 }, landsat: { oa: 94.53, fscd: 84.12, miou: 85.65, sek: 51.14 } },
    { model: "SCanNet", family: "Transformer", second: { oa: 87.86, fscd: 63.66, miou: 73.42, sek: 23.94 }, landsat: { oa: 96.04, fscd: 85.62, miou: 86.37, sek: 52.63 } },
    { model: "ChangeMamba", family: "Mamba", second: { oa: 88.12, fscd: 64.03, miou: 73.68, sek: 24.11 }, landsat: { oa: 96.08, fscd: 86.61, miou: 86.91, sek: 53.66 } },
    { model: "Mamba-FCS", family: "Mamba", second: { oa: 88.62, fscd: 65.78, miou: 74.07, sek: 25.50 }, landsat: { oa: 96.25, fscd: 89.27, miou: 88.81, sek: 60.26 } },
  ];

  const metricLabels = {
    oa: "OA (%)",
    fscd: "F_SCD (%)",
    miou: "mIoU (%)",
    sek: "SeK (%)",
  };

  function modelColor(row) {
    if (row.model === "Mamba-FCS") {
      return "#3273dc";
    }
    if (row.family === "Mamba") {
      return "#6f8fd8";
    }
    if (row.family === "Transformer") {
      return "#9a9a9a";
    }
    return "#c9c9c9";
  }

  function niceCeil(value) {
    const step = value <= 40 ? 5 : 10;
    return Math.ceil((value + step * 0.35) / step) * step;
  }

  function renderMetricPlot(targetId, datasetKey, metricKey) {
    const target = document.getElementById(targetId);
    if (!target) {
      return;
    }

    const width = 620;
    const height = 360;
    const margin = { top: 24, right: 18, bottom: 104, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const mambaValue = benchmarkResults.find((row) => row.model === "Mamba-FCS")[datasetKey][metricKey];
    const yMax = niceCeil(mambaValue);
    const tickStep = yMax <= 40 ? 5 : 10;
    const ticks = [];
    for (let tick = 0; tick <= yMax; tick += tickStep) {
      ticks.push(tick);
    }

    const points = benchmarkResults.map((row, index) => {
      const value = row[datasetKey][metricKey];
      const x = margin.left + (chartWidth / (benchmarkResults.length - 1)) * index;
      const y = margin.top + chartHeight - (value / yMax) * chartHeight;
      const labelY = margin.top + chartHeight + 18;
      const radius = row.model === "Mamba-FCS" ? 5.5 : 4.5;

      return `
        <g>
          <title>${row.model}: ${value.toFixed(2)} ${metricLabels[metricKey]}</title>
          <line x1="${x.toFixed(2)}" y1="${y.toFixed(2)}" x2="${x.toFixed(2)}" y2="${margin.top + chartHeight}" class="dot-stem"></line>
          <circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${radius}" fill="${modelColor(row)}"></circle>
          <text x="${x.toFixed(2)}" y="${(y - 8).toFixed(2)}" text-anchor="middle" class="dot-value">${value.toFixed(2)}</text>
          <text x="${x.toFixed(2)}" y="${labelY}" text-anchor="end" transform="rotate(-45 ${x.toFixed(2)} ${labelY})" class="x-label">${row.model}</text>
        </g>
      `;
    }).join("");

    const grid = ticks.map((tick) => {
      const y = margin.top + chartHeight - (tick / yMax) * chartHeight;
      return `
        <g>
          <line x1="${margin.left}" y1="${y.toFixed(2)}" x2="${margin.left + chartWidth}" y2="${y.toFixed(2)}" class="grid-line"></line>
          <text x="${margin.left - 10}" y="${(y + 4).toFixed(2)}" text-anchor="end" class="tick-label">${tick}</text>
        </g>
      `;
    }).join("");

    target.innerHTML = `
      <svg class="metric-svg" viewBox="0 0 ${width} ${height}" aria-label="${metricLabels[metricKey]} comparison">
        ${grid}
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + chartHeight}" class="axis-line"></line>
        <line x1="${margin.left}" y1="${margin.top + chartHeight}" x2="${margin.left + chartWidth}" y2="${margin.top + chartHeight}" class="axis-line"></line>
        <text x="${margin.left - 38}" y="${margin.top + chartHeight / 2}" text-anchor="middle" transform="rotate(-90 ${margin.left - 38} ${margin.top + chartHeight / 2})" class="axis-title">${metricLabels[metricKey]}</text>
        ${points}
      </svg>
    `;
  }

  function updatePlots() {
    const select = document.getElementById("metric-select");
    const metricKey = select ? select.value : "sek";
    renderMetricPlot("plot-second", "second", metricKey);
    renderMetricPlot("plot-landsat", "landsat", metricKey);
  }

  const metricSelect = document.getElementById("metric-select");
  if (metricSelect) {
    metricSelect.addEventListener("change", updatePlots);
    updatePlots();
  }

  const burgers = Array.from(document.querySelectorAll(".navbar-burger"));

  burgers.forEach((burger) => {
    burger.addEventListener("click", () => {
      const targetId = burger.dataset.target;
      const target = targetId ? document.getElementById(targetId) : null;

      burger.classList.toggle("is-active");
      if (target) {
        target.classList.toggle("is-active");
      }
    });
  });

  document.querySelectorAll(".navbar-menu a[href^='#']").forEach((link) => {
    link.addEventListener("click", () => {
      document.querySelectorAll(".navbar-burger, .navbar-menu").forEach((node) => {
        node.classList.remove("is-active");
      });
    });
  });
});
