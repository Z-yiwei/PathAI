"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const data = PROJECT_DATA;
const names = { events: "Events", image: "Image", both: "Image + events" };
const colors = { events: "#afabff", image: "#efa3c5", both: "#6ce9d3" };
let currentRun = "a";

function activate(group, button) {
  group.querySelectorAll("button").forEach((item) => {
    const active = item === button;
    item.classList.toggle("active", active);
    item.setAttribute("aria-pressed", String(active));
  });
}

function makeDots(target, count, training) {
  for (let i = 0; i < count; i++) {
    const dot = document.createElement("i");
    if (i >= training) dot.className = "val";
    target.append(dot);
  }
}
makeDots($("#mayo-dots"), 50, 40);
makeDots($("#tcga-dots"), 10, 6);
$("#tcga-dots").classList.add("tcga");

function showPatient(index) {
  const patient = data.patients[index];
  $("#raw-record").textContent = JSON.stringify(
    {
      patient_id: patient.patient_id,
      events: patient.events,
    },
    null,
    2,
  );
  $("#patient-title").textContent = patient.patient_id;
  $("#patient-summary").textContent =
    patient.source === "mayo"
      ? "A synthetic Mayo training record. The full event sequence supplies next-event targets; an available report supplies auxiliary semantic supervision during fine-tuning."
      : `${patient.events.length} events in the full record; ${patient.input_events.length} events through resection. The recorded-outcome label is ${patient.label}.`;
  $("#data-events").replaceChildren();
  patient.events.forEach((event) => {
    const token = document.createElement("span");
    token.className =
      "token" +
      (event === "DEATH" || event === "EVENT_PROGRESSION" ? " outcome" : "");
    token.textContent = event;
    $("#data-events").append(token);
  });
}
$("#patient-select").addEventListener("change", (event) =>
  showPatient(Number(event.target.value)),
);
showPatient(0);

const eventMeanings = {
  DEMO_SEX_F: "Female demographic token.",
  AGE_60S: "Age group in the sixties.",
  SMOKE_NEVER: "Never-smoker category.",
  DIAG_C34: "Lung cancer diagnosis category.",
  STAGE_T2: "Recorded primary-tumor stage token.",
  STAGE_N0: "Recorded node-negative stage token.",
  STAGE_M1: "Recorded distant-metastasis stage token.",
  MUT_EGFR: "Recorded EGFR mutation token.",
  MUT_TP53: "Recorded TP53 mutation token.",
  PROC_RESECTION: "The first resection is the final event in the model input.",
  EVENT_PROGRESSION:
    "A later progression event. Excluded from the TCGA model input.",
  DEATH:
    "A later death event. Defines the positive outcome label, outside the TCGA model input.",
};
const example = data.patients[0];
example.events.forEach((event, index) => {
  const button = document.createElement("button");
  button.className =
    "token" + (index >= example.input_events.length ? " outcome" : "");
  button.textContent = event;
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", () => {
    activate($("#timeline"), button);
    $("#event-detail").replaceChildren();
    const title = document.createElement("strong");
    title.textContent = event + " · ";
    $("#event-detail").append(title, eventMeanings[event]);
    if (index < example.ids.length) {
      $("#event-detail").append(
        ` Vocabulary ID ${example.ids[index]}. Retained as input.`,
      );
    }
  });
  $("#timeline").append(button);
});
$("#timeline").children[9].click();
$("#token-ids").textContent = "[" + example.ids.join(", ") + "]";

const slide = data.slide;
const [slideWidth, slideHeight] = slide.dimensions;
$("#slide-dimensions").textContent =
  `${slideWidth.toLocaleString()} × ${slideHeight.toLocaleString()} px`;
$("#slide-overlay").setAttribute("viewBox", `0 0 ${slideWidth} ${slideHeight}`);
$("#slide-overlay").innerHTML = slide.coords
  .map(
    ([x, y], index) =>
      `<rect x="${x}" y="${y}" width="${slide.footprint}" height="${slide.footprint}" fill="${index < 12 ? "#ff558b" : "#26c4ad"}" fill-opacity=".65" stroke="${index < 12 ? "#b8235a" : "#005e55"}" stroke-width="110"><title>Sampled patch ${index + 1}</title></rect>`,
  )
  .join("");
$("#show-regions").addEventListener("change", (event) => {
  $("#slide-overlay").style.visibility = event.target.checked
    ? "visible"
    : "hidden";
});
for (let index = 0; index < 12; index++) {
  const button = document.createElement("button");
  const position = `${((index % 4) * 100) / 3}% ${Math.floor(index / 4) * 50}%`;
  button.style.backgroundPosition = position;
  button.setAttribute("aria-label", `Show tissue patch ${index + 1}`);
  button.setAttribute("aria-pressed", "false");
  button.addEventListener("click", () => {
    activate($("#patch-picker"), button);
    $("#patch-preview").style.backgroundPosition = position;
    $("#patch-preview").setAttribute(
      "aria-label",
      `Actual H&E tissue patch ${index + 1}`,
    );
    const [x, y] = slide.coords[index];
    $("#patch-caption").textContent =
      `Patch ${index + 1} · level-0 location (${x.toLocaleString()}, ${y.toLocaleString()}) · model input 224 × 224 px`;
    [...$("#slide-overlay").children].forEach((rect, i) => {
      rect.setAttribute("stroke-width", i === index ? "750" : "110");
    });
  });
  $("#patch-picker").append(button);
}
$("#patch-picker").firstElementChild.click();

$("#report-text").textContent = data.report.clean;
$$("#report-tabs button").forEach((button) =>
  button.addEventListener("click", () => {
    activate($("#report-tabs"), button);
    $("#report-text").textContent = data.report[button.dataset.report];
  }),
);
$("#chunk-description").textContent =
  `This report contains ${data.report.tokens} text tokens, split into ${data.report.chunks} contiguous chunks of up to 75 tokens. Each receives start/end tokens for PLIP.`;
for (let i = 0; i < data.report.chunks; i++) {
  const block = document.createElement("i");
  block.title = `Chunk ${i + 1}: ${Math.min(75, data.report.tokens - 75 * i)} text tokens`;
  $("#chunks").append(block);
}
$("#chunks").setAttribute(
  "aria-label",
  `${data.report.chunks} report chunks; all ${data.report.tokens} text tokens covered`,
);

const descriptions = {
  events:
    "Event embeddings enter the original GPT. Each event sees itself and earlier events. The last non-padding event supplies the patient state.",
  image:
    "The projected image is the only input token. It passes through the same GPT blocks; its final state supplies the patient representation.",
  both: "The projected image comes first. Every later event can attend to the image and earlier events. The final resection token summarizes this combined context.",
};
function showMode(mode) {
  const sequence =
    mode === "image"
      ? ["IMG"]
      : mode === "events"
        ? ["E1", "E2", "E3", "E4"]
        : ["IMG", "E1", "E2", "E3", "E4"];
  $("#model-tokens").innerHTML = sequence
    .map(
      (token) =>
        `<span class="${token === "IMG" ? "image-token" : ""}">${token === "IMG" ? "IMAGE · 64D" : token}</span>`,
    )
    .join("");
  $("#mode-description").textContent = descriptions[mode];
  const grid = $("#attention-grid");
  grid.style.gridTemplateColumns = `30px repeat(${sequence.length}, 1fr)`;
  grid.setAttribute(
    "aria-label",
    `${names[mode]} causal mask. Each query attends to itself and preceding positions.`,
  );
  grid.innerHTML =
    "<span></span>" + sequence.map((token) => `<span>${token}</span>`).join("");
  sequence.forEach((query, row) => {
    grid.insertAdjacentHTML("beforeend", `<span>${query}</span>`);
    sequence.forEach((key, col) => {
      const visible = col <= row;
      const cell = document.createElement("div");
      cell.className =
        "cell" +
        (visible ? " visible" : "") +
        (visible && key === "IMG" ? " image-cell" : "");
      cell.title = `${query} → ${key}: ${visible ? "visible" : "masked"}`;
      grid.append(cell);
    });
  });
}
$$("#mode-tabs button").forEach((button) =>
  button.addEventListener("click", () => {
    activate($("#mode-tabs"), button);
    showMode(button.dataset.mode);
  }),
);
showMode("both");

const lossDescriptions = {
  outcome:
    "Outcome BCE uses the six TCGA training labels. Inverse-frequency class weights balance positive and negative examples. The outcome head predicts a logit; BCEWithLogits combines the sigmoid and binary cross-entropy computation. Validation BCE is unweighted.",
  events:
    "Event CE = Mayo next-event cross-entropy + TCGA next-event cross-entropy. Padding targets are ignored. The joint-input model conditions TCGA event prediction on the image prefix; the single-input baselines retain event-only auxiliary language modeling.",
  report:
    "Report loss = Mayo report cosine loss + TCGA report cosine loss. Each averages 1 − cosine similarity across available training reports. A missing or unreadable report is masked. Mayo report predictions use events; TCGA report predictions use the selected outcome input mode.",
};
$$(".loss-equation button").forEach((button) =>
  button.addEventListener("click", () => {
    activate($(".loss-equation"), button);
    $("#loss-detail").textContent = lossDescriptions[button.dataset.loss];
  }),
);
$(".loss-equation button").click();
const pretraining = data.pretraining;
$("#pretrain-drop").textContent =
  `${pretraining[0].val_lm.toFixed(2)} → ${pretraining.at(-1).val_lm.toFixed(3)}`;
function showBce() {
  const probability = Number($("#probability").value) / 100;
  $("#probability-value").textContent = probability.toFixed(2);
  $("#bce-value").textContent = (-Math.log(probability)).toFixed(3);
}
$("#probability").addEventListener("input", showBce);
showBce();

function drawChart() {
  const field = $("#chart-select").value;
  const run = data.runs[currentRun];
  const isPretraining = field === "pretraining";
  const titles = {
    eval_bce: "Validation BCE",
    train_bce: "Training BCE",
    report_loss: "Training report cosine loss",
    pretraining: "Original GPT: Mayo pretraining",
  };
  $("#chart-title").textContent = titles[field];
  const series = isPretraining
    ? [
        {
          name: "Training",
          color: colors.both,
          points: pretraining.map((r) => [r.step, r.train_lm]),
        },
        {
          name: "Validation",
          color: colors.image,
          points: pretraining.map((r) => [r.step, r.val_lm]),
        },
      ]
    : run.history.map((mode) => ({
        name: names[mode.mode],
        color: colors[mode.mode],
        points: mode.trace.map((r) => [r.step, r[field]]),
        selected: field === "eval_bce" ? mode.selected_step : null,
      }));
  const values = series.flatMap((line) => line.points.map((point) => point[1]));
  const maxX = isPretraining ? 300 : 100;
  const spread = Math.max(...values) - Math.min(...values);
  const minY = Math.max(0, Math.min(...values) - spread * 0.14);
  const maxY = Math.max(...values) + spread * 0.12;
  const width = 850,
    height = 330,
    left = 58,
    right = 20,
    top = 20,
    bottom = 44;
  const plotWidth = width - left - right,
    plotHeight = height - top - bottom;
  const x = (value) => left + (value / maxX) * plotWidth;
  const y = (value) => top + ((maxY - value) / (maxY - minY)) * plotHeight;
  let svg = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${titles[field]} for Run ${currentRun.toUpperCase()}"><title>${titles[field]}: actual logged experiment values</title>`;
  for (let i = 0; i <= 4; i++) {
    const value = minY + ((maxY - minY) * i) / 4;
    const position = y(value);
    svg += `<line x1="${left}" y1="${position}" x2="${width - right}" y2="${position}" stroke="#2b3749" stroke-dasharray="3 5"/><text x="${left - 12}" y="${position + 4}" fill="#9eaec2" font-size="11" font-family="monospace" text-anchor="end">${value.toFixed(2)}</text>`;
  }
  for (let i = 0; i <= 5; i++) {
    const value = (maxX * i) / 5;
    svg += `<text x="${x(value)}" y="${height - 22}" fill="#9eaec2" font-size="11" font-family="monospace" text-anchor="middle">${value}</text>`;
  }
  svg += `<text x="${left + plotWidth / 2}" y="${height - 2}" fill="#9eaec2" font-size="11" text-anchor="middle">${isPretraining ? "Pretraining" : "Fine-tuning"} step</text>`;
  series.forEach((line) => {
    const d = line.points
      .map(
        ([step, value], index) =>
          `${index ? "L" : "M"}${x(step).toFixed(2)},${y(value).toFixed(2)}`,
      )
      .join(" ");
    svg += `<path d="${d}" stroke="${line.color}" stroke-width="2.7" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;
    line.points.forEach(([step, value]) => {
      const selected = line.selected === step;
      svg += `<circle cx="${x(step)}" cy="${y(value)}" r="${selected ? 6 : 3}" fill="${line.color}" ${selected ? 'stroke="#edf2fa" stroke-width="1.5"' : 'fill-opacity=".7"'}><title>${line.name}: step ${step}, value ${value.toFixed(4)}${selected ? " · selected checkpoint" : ""}</title></circle>`;
    });
  });
  $("#chart").innerHTML = svg + "</svg>";
  $("#chart-legend").innerHTML = series
    .map(
      (line) =>
        `<span><i style="background:${line.color}"></i>${line.name}</span>`,
    )
    .join("");
  $("#chart").setAttribute(
    "aria-label",
    `${titles[field]}. ${series.map((line) => `${line.name}: ${line.points[0][1].toFixed(3)} at start to ${line.points.at(-1)[1].toFixed(3)} at final logged step`).join(". ")}`,
  );
}
function showRun() {
  const run = data.runs[currentRun];
  $("#metric-cards").innerHTML = Object.keys(names)
    .map((mode) => {
      const metric = run.metrics[mode];
      const selected = run.history.find(
        (record) => record.mode === mode,
      ).selected_step;
      return `<article class="metric-card ${mode === "both" ? "winner" : ""}"><h3>${names[mode]}${mode === "both" ? "<span>BEST ACC. + BCE</span>" : ""}</h3><div class="metric-values"><div><strong>${Math.round(metric.accuracy * 100)}%</strong><small>ACCURACY ↑</small></div><div><strong>${metric.bce.toFixed(4)}</strong><small>BCE ↓</small></div></div><p>AUROC ${metric.auroc.toFixed(2)} · selected step ${selected}</p></article>`;
    })
    .join("");
  const patients = [...new Set(run.predictions.map((row) => row.patient_id))];
  $("#prediction-table").innerHTML = patients
    .map((patient) => {
      const rows = run.predictions.filter((row) => row.patient_id === patient);
      const label = Number(rows[0].label);
      return `<tr><td>${patient}</td><td>${label}</td>${Object.keys(names)
        .map((mode) => {
          const row = rows.find((record) => record.mode === mode);
          const probability = Number(row.probability);
          const correct = Number(probability >= 0.5) === label;
          return `<td class="${correct ? "correct" : "incorrect"}">${probability.toFixed(3)} ${correct ? "✓" : "×"}</td>`;
        })
        .join("")}</tr>`;
    })
    .join("");
  const gradients = run.history.find(
    (record) => record.mode === "both",
  ).initial_gradients;
  const components = {
    gpt_attention: "Original GPT · first attention QKV",
    image_projection: "Image projection · 768 → 64",
    report_head: "Report head · 64 → 512",
  };
  $("#gradient-table").innerHTML = Object.entries(gradients)
    .map(
      ([key, value]) =>
        `<tr><td>${components[key]}</td><td>${value.toFixed(6)}</td></tr>`,
    )
    .join("");
  $("#full-curves").src = `../results/run_${currentRun}/learning_curves.png`;
  $("#full-curves").alt =
    `All four logged training curves for Run ${currentRun.toUpperCase()}`;
  $("#curve-link").href = $("#full-curves").getAttribute("src");
  drawChart();
}
$$("#run-tabs button").forEach((button) =>
  button.addEventListener("click", () => {
    activate($("#run-tabs"), button);
    currentRun = button.dataset.run;
    showRun();
  }),
);
$("#chart-select").addEventListener("change", drawChart);
showRun();

const sections = $$("main > section");
let scheduled = false;
function updateNavigation() {
  const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
  $("#progress").style.width =
    `${maxScroll > 0 ? (window.scrollY / maxScroll) * 100 : 0}%`;
  const active =
    sections
      .filter((section) => section.getBoundingClientRect().top <= 180)
      .at(-1) || sections[0];
  $$(".rail a").forEach((link) => {
    const selected = link.hash === "#" + active.id;
    link.classList.toggle("active", selected);
    if (selected) link.setAttribute("aria-current", "step");
    else link.removeAttribute("aria-current");
  });
  scheduled = false;
}
window.addEventListener(
  "scroll",
  () => {
    if (!scheduled) {
      requestAnimationFrame(updateNavigation);
      scheduled = true;
    }
  },
  { passive: true },
);
window.addEventListener("resize", updateNavigation);
updateNavigation();
