/* Scribix (transcriptor local) — lógica de la interfaz.
 *
 * La página es un shell de dos paneles: la biblioteca a la izquierda (lista
 * larga, scroll propio, búsqueda y filtros) y a la derecha una de dos vistas,
 * "Nuevo trabajo" o "Trabajo".  Antes todo vivía en una sola columna y con
 * 50 trabajos guardados había que recorrer la lista entera para llegar al
 * formulario.
 */

let SOURCE = "url";
let MODE = "transcribe";
let INFO = null;
let ITERATE_JOB_ID = null;

// --- Estado de la biblioteca -------------------------------------------- //
let JOBS = [];              // último listado del servidor, sin filtrar
let QUERY = "";
let FILTER = "all";
let SORT = "recent";
let SELECTED_JOB = null;    // id resaltado en la lista
let EXPANDED_JOB = null;    // id con el cajón de acciones abierto
let VIEW = "new";

// ----------------------------------------------------------------------- //
// Diálogo reutilizable                                                      //
// ----------------------------------------------------------------------- //

const DIALOG_QUEUE = [];
let DIALOG_ACTIVE = null;

function presentNextDialog() {
  if (DIALOG_ACTIVE || !DIALOG_QUEUE.length) return;
  DIALOG_ACTIVE = DIALOG_QUEUE.shift();
  const options = DIALOG_ACTIVE.options;
  const dialog = document.getElementById("app-dialog");
  const kind = ["warning", "danger"].includes(options.kind)
    ? options.kind : "info";
  dialog.className = `app-dialog is-${kind}`;
  document.getElementById("app-dialog-icon").textContent = {
    info: "i", warning: "!", danger: "×",
  }[kind];
  document.getElementById("app-dialog-title").textContent =
    options.title || (kind === "danger" ? "Confirma esta acción" : "Aviso");
  document.getElementById("app-dialog-message").textContent = options.message || "";

  const detail = document.getElementById("app-dialog-detail");
  detail.textContent = options.detail || "";
  detail.classList.toggle("hidden", !options.detail);

  const cancel = document.getElementById("app-dialog-cancel");
  cancel.textContent = options.cancelLabel || "";
  cancel.classList.toggle("hidden", !options.cancelLabel);
  const confirm = document.getElementById("app-dialog-confirm");
  confirm.textContent = options.confirmLabel || "Entendido";
  dialog.showModal();
  confirm.focus();
}

function finishDialog(accepted) {
  const dialog = document.getElementById("app-dialog");
  if (!dialog.open || !DIALOG_ACTIVE) return;
  const active = DIALOG_ACTIVE;
  DIALOG_ACTIVE = null;
  dialog.close();
  active.resolve(accepted);
  presentNextDialog();
}

function showAppDialog(options) {
  return new Promise(resolve => {
    DIALOG_QUEUE.push({options, resolve});
    presentNextDialog();
  });
}

function showAlert(message, options = {}) {
  return showAppDialog({
    title: options.title || "No se pudo completar",
    message,
    detail: options.detail,
    kind: options.kind || "warning",
    confirmLabel: options.confirmLabel || "Entendido",
  });
}

function showConfirm(options) {
  return showAppDialog({
    ...options,
    kind: options.kind || "danger",
    cancelLabel: options.cancelLabel || "Volver",
    confirmLabel: options.confirmLabel || "Confirmar",
  });
}

document.getElementById("app-dialog-confirm").addEventListener("click", () => {
  finishDialog(true);
});
document.getElementById("app-dialog-cancel").addEventListener("click", () => {
  finishDialog(false);
});
document.getElementById("app-dialog").addEventListener("cancel", event => {
  event.preventDefault();
  // Escape equivale a volver en una confirmación y a cerrar en un aviso.
  finishDialog(!DIALOG_ACTIVE?.options.cancelLabel);
});

// ----------------------------------------------------------------------- //
// Estructura: vistas y barra lateral                                        //
// ----------------------------------------------------------------------- //

function setView(view) {
  VIEW = view;
  for (const name of ["new", "job"]) {
    document.getElementById(`view-${name}`).classList.toggle("hidden", name !== view);
    document.getElementById(`switch-${name}`)
      .setAttribute("aria-selected", name === view ? "true" : "false");
  }
  document.querySelector(".main").scrollTop = 0;
  closeSidebar();
}

// La vista de trabajo solo se puede elegir cuando hay algo que enseñar.
function enableJobView(enabled) {
  document.getElementById("switch-job").disabled = !enabled;
}

function openSidebar() {
  document.getElementById("sidebar").classList.add("open");
  document.getElementById("scrim").classList.remove("hidden");
  document.getElementById("sidebar-toggle").setAttribute("aria-expanded", "true");
}

function closeSidebar() {
  document.getElementById("sidebar").classList.remove("open");
  document.getElementById("scrim").classList.add("hidden");
  document.getElementById("sidebar-toggle").setAttribute("aria-expanded", "false");
}

function toggleSidebar() {
  const open = document.getElementById("sidebar").classList.contains("open");
  if (open) closeSidebar(); else openSidebar();
}

// ----------------------------------------------------------------------- //
// Formulario                                                                //
// ----------------------------------------------------------------------- //

function syncGoLabel() {
  document.getElementById("go").textContent = ITERATE_JOB_ID
    ? "Reanalizar usando la voz guardada"
    : MODE === "audio_only" ? "Descargar audio"
    : MODE === "video_only" ? "Descargar video"
    : MODE === "ultra" ? "Sacar letra ultra precisa"
    : MODE === "lyrics" ? "Sacar letra" : "Transcribir";
}

function setSource(s) {
  // Elegir una fuente explícitamente significa iniciar un trabajo distinto.
  ITERATE_JOB_ID = null;
  SOURCE = s;
  document.querySelectorAll(".tab[data-src]").forEach(t => {
    const on = t.dataset.src === s;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
    t.setAttribute("tabindex", on ? "0" : "-1");
  });
  document.getElementById("pane-url").classList.toggle("hidden", s !== "url");
  document.getElementById("pane-file").classList.toggle("hidden", s !== "file");
  syncAudioOpts();
  syncGoLabel();
}

const SEPARATES = m => m === "lyrics" || m === "ultra";

// El formato de audio solo aplica cuando bajamos de YouTube. En los modos de
// letra el backend usa siempre el original (separa mejor la voz).
function syncAudioOpts() {
  document.getElementById("audio-opts").classList.toggle(
    "hidden", SOURCE !== "url" || SEPARATES(MODE) || MODE === "video_only");
  document.getElementById("video-opts").classList.toggle(
    "hidden", SOURCE !== "url" || MODE !== "video_only");
  document.getElementById("separator-opts").classList.toggle(
    "hidden", !SEPARATES(MODE));
  document.getElementById("ultra-opts").classList.toggle(
    "hidden", MODE !== "ultra");
}

// El modo ultra corre con faster-whisper, que guarda sus pesos aparte: sin
// esto la lista diría "descargado" para un modelo que el modo ultra no tiene.
function syncModelLabels() {
  const ultra = MODE === "ultra";
  for (const option of document.getElementById("model").options) {
    const model = INFO?.models?.find(item => item.name === option.value);
    if (!model) continue;
    const engine = effectiveEngine();
    const cached = ultra
      ? engine === "mlx-whisper" ? model.mlx_cached
      : engine === "faster-whisper" ? model.faster_cached
      : model.cached
      : model.cached;
    const description = model.parameters
      ? `${model.label} · ${model.parameters} · ${model.quality}`
      : (model.label || model.name);
    option.textContent = description
      + (cached ? "  ✓ descargado" : `  ⬇ ${model.download || "requiere descarga"}`);
  }
  onModelChange();
}

function effectiveEngine() {
  if (MODE !== "ultra") return "openai-whisper";
  const selected = document.getElementById("asr_engine")?.value || "auto";
  return selected === "auto" ? (INFO?.auto_engine || "openai-whisper") : selected;
}

function onModelChange() {
  if (!INFO) return;
  const name = document.getElementById("model").value;
  const model = INFO.models?.find(item => item.name === name);
  if (!model) return;
  const engine = effectiveEngine();
  const artifact = model.artifacts?.[engine] || name;
  if (!model.parameters) {
    document.getElementById("model-hint").textContent =
      `${model.label || name}. Reinicia el servidor para cargar los detalles `
      + "completos del modelo y del motor.";
    return;
  }
  document.getElementById("model-hint").textContent =
    `${model.label} · ${model.parameters} parámetros · ${model.download} · `
    + `${model.speed} · calidad ${model.quality.toLowerCase()}. ${model.hint} `
    + `Modelo exacto: ${artifact}.`;
}

function onEngineChange() {
  if (!INFO) return;
  const selected = document.getElementById("asr_engine").value || "auto";
  const engine = INFO.asr_engines?.find(item => item.value === selected);
  const actual = effectiveEngine();
  document.getElementById("engine-hint").textContent =
    `${engine?.hint || ""} Motor efectivo: ${actual}.`;
  syncModelLabels();
}

function onPrecisionChange() {
  const value = document.getElementById("precision_level").value;
  const spec = (INFO?.precision_levels || []).find(p => p.value === value);
  document.getElementById("precision-hint").textContent = spec ? spec.hint : "";
}

function onTranslateChange() {
  const wants = document.getElementById("translate_to").value;
  const hint = document.getElementById("translator-hint");
  const translator = INFO?.translator || {};
  if (!wants) {
    hint.textContent = "Sin traducción: Lyrixa recibe la letra original y, si "
      + "pides romaji, su transliteración.";
    hint.className = "hint";
  } else if (translator.available) {
    hint.textContent = `Traduce en tu máquina con ${translator.model}. `
      + "Los idiomas de escritura no latina pasan por inglés (ruso → inglés → destino).";
    hint.className = "hint";
  } else {
    hint.textContent = "⚠️ No hay traductor local. Arranca Ollama y baja un "
      + "modelo multilingüe: ollama pull qwen3.5";
    hint.className = "warn";
  }
}

function onFormatChange() {
  const sel = document.getElementById("audio_format");
  const spec = (INFO?.audio_formats || []).find(f => f.value === sel.value);
  // Los formatos sin pérdida no tienen bitrate que elegir.
  document.getElementById("quality-wrap").classList.toggle(
    "hidden", !spec || spec.lossless);
  document.getElementById("fmt-hint").textContent = spec ? spec.hint : "";
}

function onVideoFormatChange() {
  const value = document.getElementById("video_format").value;
  const spec = (INFO?.video_formats || []).find(item => item.value === value);
  document.getElementById("video-fmt-hint").textContent = spec ? spec.hint : "";
}

function onSeparationChange() {
  const value = document.getElementById("separation_quality").value;
  const spec = (INFO?.separation_profiles || []).find(p => p.value === value);
  const acceleration = INFO?.separator_acceleration;
  const hardware = spec?.value === "ultra" || spec?.value === "precise"
    ? ` Motor actual: ${acceleration?.mode || "desconocido"}`
      + (acceleration?.accelerated ? " acelerado." : " (sin aceleración).")
    : "";
  document.getElementById("separation-hint").textContent =
    (spec ? spec.hint : "") + hardware;
}

function setMode(m) {
  MODE = m;
  if (m === "video_only" && SOURCE !== "url") setSource("url");
  document.querySelectorAll(".tab[data-mode]").forEach(t => {
    const on = t.dataset.mode === m;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
    t.setAttribute("tabindex", on ? "0" : "-1");
  });
  // El panel agrupado se presenta como el panel del tab activo.
  document.getElementById("mode-panel")
    .setAttribute("aria-labelledby", "mode-tab-" + m);
  // Las descargas directas no necesitan Whisper ni selector de idioma.
  document.getElementById("opts").classList.toggle(
    "hidden", m === "audio_only" || m === "video_only");
  if (m !== "ultra") ITERATE_JOB_ID = null;
  syncGoLabel();
  const hints = {
    lyrics: "Aísla la voz con el motor elegido y luego la transcribe. " +
            "BS-RoFormer recupera mejor las voces débiles; Demucs es más rápido.",
    ultra: "Aísla la voz, detecta dónde se canta y transcribe varias veces " +
           "con ajustes distintos. Después vota palabra por palabra y puede " +
           "revisar solo las zonas dudosas en el audio original. Detecta el " +
           "idioma tramo por tramo, así que un mix con canciones en japonés y " +
           "en inglés se transcribe y se traduce bien en las dos partes.",
    video_only: "Descarga el video y su audio en el mejor flujo disponible " +
                "hasta la resolución elegida. Función local para uso personal.",
  };
  const hint = document.getElementById("mode-hint");
  hint.textContent = hints[m] || "";
  hint.classList.toggle("hidden", !hints[m]);
  if (INFO && document.getElementById("asr_engine").options.length) {
    onEngineChange();
  } else {
    syncModelLabels();
  }
  syncAudioOpts();
}

// Tabs accesibles: flechas mueven la selección (activación automática, que
// reusa el onclick existente → misma lógica de estado), Enter/Espacio activan,
// Home/End saltan al primero/último. No reescribe setSource/setMode.
function wireTablist(tablistEl) {
  const tabs = () => Array.from(tablistEl.querySelectorAll(".tab"));
  tablistEl.addEventListener("keydown", ev => {
    const items = tabs();
    const i = items.indexOf(document.activeElement);
    if (i < 0) return;
    let next = null;
    if (ev.key === "ArrowRight" || ev.key === "ArrowDown") next = items[(i + 1) % items.length];
    else if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") next = items[(i - 1 + items.length) % items.length];
    else if (ev.key === "Home") next = items[0];
    else if (ev.key === "End") next = items[items.length - 1];
    else if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      items[i].click();
      return;
    }
    if (!next) return;
    ev.preventDefault();
    next.focus();
    next.click();
  });
}

// Qué mostrar a la derecha de la barra en las etapas que no son transcripción.
const STAGE_LABELS = {
  downloading: "descargando",
  downloading_model: "descargando modelo",
  downloading_separator_model: "descargando modelo de separación",
  separating: "aislando la voz",
  analyzing: "buscando la voz",
  references: "preparando guía del mix",
  detecting_language: "detectando idioma",
  transcribing: "transcribiendo",
  romanizing: "pasando a romaji",
  translating: "traduciendo",
  postprocessing: "convirtiendo",
};

// Cómo presentar cada archivo del resultado. El orden manda en la fila.
const FILE_KINDS = [
  ["video", "🎞️", "Video"],
  ["audio", "⬇️", "Audio"],
  ["vocals", "🎤", "Voz aislada"],
  ["instrumental", "🎹", "Instrumental sin voz"],
  ["lyrics", "📝", "Letra legible (original · romaji · traducción)"],
  ["txt", "📄", "Texto plano"],
  ["srt", "🎬", "Subtítulos SRT"],
  ["vtt", "🌐", "Subtítulos VTT"],
  ["tsv", "📊", "Tabla TSV"],
  ["json", "{ }", "JSON con tiempos"],
  ["guide", "🧭", "Guía del mix y referencias"],
  ["lyrixa", "🎵", "Proyecto Lyrixa"],
];

// ----------------------------------------------------------------------- //
// Formatos                                                                  //
// ----------------------------------------------------------------------- //

function fmt(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function fmtBytes(bytes) {
  let value = Math.max(0, Number(bytes) || 0);
  const units = ["B", "KB", "MB", "GB", "TB"];
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024; unit++;
  }
  return `${value.toFixed(unit >= 3 ? 1 : 0)} ${units[unit]}`;
}

// Igual que fmt pero con milisegundos (importante para sincronizar letras).
function fmtMs(sec) {
  sec = Math.max(0, sec || 0);
  const whole = Math.floor(sec);
  const ms = Math.floor((sec - whole) * 1000);
  return fmt(whole) + "." + String(ms).padStart(3, "0");
}

// "hace 3 h" dice más que una fecha completa cuando la lista está ordenada
// por reciente; a partir de una semana la fecha vuelve a ser lo útil.
function fmtAgo(iso) {
  const then = Date.parse(iso || "");
  if (!Number.isFinite(then)) return "";
  const minutes = Math.round((Date.now() - then) / 60000);
  if (minutes < 1) return "ahora";
  if (minutes < 60) return `hace ${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `hace ${hours} h`;
  const days = Math.round(hours / 24);
  if (days === 1) return "ayer";
  if (days < 7) return `hace ${days} d`;
  return new Date(then).toLocaleDateString("es", {day: "numeric", month: "short"});
}

async function loadTracklistFile(input) {
  const file = input.files?.[0];
  if (!file) return;
  document.getElementById("tracklist_text").value = await file.text();
  await previewTracklist();
}

async function previewTracklist() {
  const text = document.getElementById("tracklist_text").value.trim();
  const target = document.getElementById("tracklist-preview");
  if (!text) {
    target.textContent = "";
    return;
  }
  const fd = new FormData();
  fd.append("tracklist_text", text);
  fd.append("duration", "0");
  const response = await fetch("/api/tracklists/preview", {method: "POST", body: fd});
  const data = await response.json();
  if (!response.ok) {
    target.textContent = "⚠️ " + (data.error || "No pude leer la lista.");
    return;
  }
  const sections = data.sections.map(section =>
    `${section.name}: ${section.tracks} canciones · última en ${fmt(section.last)}`
  );
  target.textContent = `✅ ${sections.join("\n")}\nLa sección correcta se elegirá por la duración real del mix.`;
}

// ----------------------------------------------------------------------- //
// Biblioteca                                                                //
// ----------------------------------------------------------------------- //

const JOB_STATUS = {
  running: "en curso", completed: "terminado",
  interrupted: "interrumpido", cancelled: "cancelado",
  error: "error", saved: "guardado",
};

const FILTERS = [
  ["all", "Todos"],
  ["completed", "Terminados"],
  ["interrupted", "Interrumpidos"],
  ["cancelled", "Cancelados"],
  ["error", "Con error"],
  ["saved", "Guardados"],
];

async function loadSavedJobs() {
  try {
    const response = await fetch("/api/jobs", {cache: "no-store"});
    const data = await response.json();
    JOBS = data.jobs || [];
  } catch (_) {
    JOBS = null;                 // distinto de "lista vacía": no hubo respuesta
  }
  renderJobs();
}

function visibleJobs() {
  const needle = QUERY.trim().toLowerCase();
  let rows = (JOBS || []).filter(job => {
    if (FILTER !== "all" && job.status !== FILTER) return false;
    if (!needle) return true;
    // El id cuenta: es lo que se pega desde un enlace o desde la carpeta.
    return `${job.title || ""} ${job.id}`.toLowerCase().includes(needle);
  });
  const key = {
    recent: job => -Date.parse(job.updatedAt || job.createdAt || 0),
    oldest: job => Date.parse(job.updatedAt || job.createdAt || 0),
    largest: job => -(job.diskBytes || 0),
    longest: job => -(job.duration || 0),
    title: job => (job.title || job.id).toLowerCase(),
  }[SORT];
  return rows.sort((a, b) => {
    const ka = key(a), kb = key(b);
    return ka < kb ? -1 : ka > kb ? 1 : 0;
  });
}

function renderFilters() {
  const box = document.getElementById("job-filters");
  box.innerHTML = "";
  for (const [value, label] of FILTERS) {
    const total = value === "all"
      ? (JOBS || []).length
      : (JOBS || []).filter(job => job.status === value).length;
    if (!total && value !== "all" && FILTER !== value) continue;
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.setAttribute("aria-pressed", FILTER === value ? "true" : "false");
    chip.textContent = label;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = total;
    chip.appendChild(count);
    chip.onclick = () => { FILTER = value; renderJobs(); };
    box.appendChild(chip);
  }
}

function renderJobs() {
  const box = document.getElementById("jobs-list");
  renderFilters();
  box.innerHTML = "";

  if (JOBS === null) {
    box.innerHTML = "<div class='empty'><strong>Sin conexión</strong>"
      + "No pude leer la carpeta de trabajos.</div>";
    document.getElementById("job-count").textContent = "";
    return;
  }
  const rows = visibleJobs();
  document.getElementById("job-count").textContent = JOBS.length
    ? `${rows.length} de ${JOBS.length}` : "";

  if (!JOBS.length) {
    box.innerHTML = "<div class='empty'><strong>Todavía no hay nada</strong>"
      + "Los trabajos que proceses aparecerán aquí.</div>";
    return;
  }
  if (!rows.length) {
    box.innerHTML = "<div class='empty'><strong>Ningún resultado</strong>"
      + "Prueba con otro texto o quita el filtro.</div>";
    return;
  }
  for (const job of rows) box.appendChild(jobRow(job));
}

function jobRow(job) {
  const row = document.createElement("div");
  row.className = "job" + (job.id === SELECTED_JOB ? " selected" : "");

  const open = document.createElement("button");
  open.className = "job-open";
  open.type = "button";
  open.onclick = () => openSavedJob(job.id);

  const title = document.createElement("span");
  title.className = "job-title";
  title.textContent = job.title || job.id;

  const when = document.createElement("span");
  when.className = "job-time";
  when.textContent = fmtAgo(job.updatedAt || job.createdAt);

  const meta = document.createElement("span");
  meta.className = "job-meta";
  const status = document.createElement("span");
  status.className = `status is-${job.status}`;
  status.textContent = JOB_STATUS[job.status] || job.status;
  meta.appendChild(status);
  const bits = [fmt(job.duration)];
  if (job.iteration) bits.push(`it. ${job.iteration + 1}`);
  if (job.diskBytes) bits.push(fmtBytes(job.diskBytes));
  if (job.reference) bits.push(`🧭 ${job.reference.found}/${job.reference.total}`);
  for (const bit of bits) {
    const sep = document.createElement("span");
    sep.className = "sep";
    sep.textContent = "·";
    const span = document.createElement("span");
    span.textContent = bit;
    meta.append(sep, span);
  }
  open.append(title, when, meta);

  const more = document.createElement("button");
  more.className = "icon-button job-more";
  more.type = "button";
  more.textContent = "⋯";
  more.title = "Acciones";
  more.setAttribute("aria-label", `Acciones de ${job.title || job.id}`);
  more.setAttribute("aria-expanded", EXPANDED_JOB === job.id ? "true" : "false");
  more.onclick = () => {
    EXPANDED_JOB = EXPANDED_JOB === job.id ? null : job.id;
    renderJobs();
  };
  row.append(open, more);

  // Cajón de acciones en la propia fila en vez de un menú flotante: no hay
  // que posicionar nada, se cierra con el mismo botón y el foco no se pierde.
  if (EXPANDED_JOB === job.id) row.appendChild(jobActions(job));
  return row;
}

function jobActions(job) {
  const actions = document.createElement("div");
  actions.className = "job-actions";
  const add = (label, handler, className) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    if (className) button.className = className;
    button.onclick = handler;
    actions.appendChild(button);
    return button;
  };
  add("Abrir", () => openSavedJob(job.id));
  if (job.status === "running") {
    add("■ Cancelar", () => cancelJob(job.id), "danger");
  }
  if (job.recoverable) add("↻ Volver a procesar", () => prepareSavedIteration(job.id));
  if (job.files?.json) add("🌐 Traducir / romaji", () => translateSavedJob(job.id));
  if (job.links?.openInLyrixa && job.files?.lyrixa) {
    add("🎛️ Lyrixa", () => window.open(job.links.openInLyrixa, "_blank", "noopener"));
  }
  const remove = add("🗑️ Eliminar", () => deleteSavedJob(job), "danger");
  remove.disabled = job.status === "running";
  return actions;
}

async function deleteSavedJob(job) {
  const size = job.diskBytes ? ` (${fmtBytes(job.diskBytes)})` : "";
  const confirmed = await showConfirm({
    title: "Eliminar trabajo",
    message: `¿Quieres eliminar permanentemente “${job.title || job.id}”${size}?`,
    detail: "Se borrarán el audio, los stems, los resultados y todas sus iteraciones. Esta acción no se puede deshacer.",
    confirmLabel: "Eliminar definitivamente",
  });
  if (!confirmed) return;
  const response = await fetch(`/api/jobs/${job.id}`, {method: "DELETE"});
  const data = await response.json();
  if (!response.ok) {
    return showAlert(data.error || "No pude eliminar el trabajo.", {
      title: "No se eliminó el trabajo",
    });
  }
  if (ITERATE_JOB_ID === job.id) {
    ITERATE_JOB_ID = null;
    syncGoLabel();
  }
  if (localStorage.getItem("job_id") === job.id) {
    localStorage.removeItem("job_id");
  }
  if (SELECTED_JOB === job.id) SELECTED_JOB = null;
  EXPANDED_JOB = null;
  await loadSavedJobs();
}

function setCancelButton(jobId, visible) {
  const button = document.getElementById("cancel-job");
  button.dataset.jobId = visible ? (jobId || "") : "";
  button.disabled = false;
  button.textContent = "■ Cancelar trabajo";
  button.classList.toggle("hidden", !visible);
}

async function cancelJob(jobId) {
  if (!jobId) return;
  const confirmed = await showConfirm({
    title: "Cancelar trabajo",
    message: "¿Quieres detener el trabajo que está en curso?",
    detail: "Scribix seguirá abierto y conservará los audios o stems que ya estén completos.",
    kind: "warning",
    confirmLabel: "Detener trabajo",
    cancelLabel: "Seguir procesando",
  });
  if (!confirmed) return;
  const button = document.getElementById("cancel-job");
  if (button.dataset.jobId === jobId) {
    button.disabled = true;
    button.textContent = "Cancelando…";
  }
  let response, data;
  try {
    response = await fetch(`/api/jobs/${jobId}/cancel`, {method: "POST"});
    data = await response.json();
  } catch (_) {
    if (button.dataset.jobId === jobId) setCancelButton(jobId, true);
    return showAlert("No pude contactar con Scribix para cancelar el trabajo.", {
      title: "Scribix no responde",
    });
  }
  if (!response.ok) {
    if (button.dataset.jobId === jobId) setCancelButton(jobId, true);
    return showAlert(data.error || "No pude cancelar el trabajo.", {
      title: "No se canceló el trabajo",
    });
  }
  if (button.dataset.jobId === jobId) setStage("Cancelando el trabajo…", true);
  await loadSavedJobs();
}

function cancelActiveJob() {
  return cancelJob(document.getElementById("cancel-job").dataset.jobId);
}

async function openSavedJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`, {cache: "no-store"});
  const job = await response.json();
  if (!response.ok) {
    return showAlert(job.error || "No pude abrir el trabajo.", {
      title: "No se pudo abrir",
    });
  }
  SELECTED_JOB = jobId;
  renderJobs();
  enableJobView(true);
  setView("job");
  resetPanels();
  clearSegmentView();
  document.getElementById("err").textContent = job.error ? "⚠️ " + job.error : "";
  document.getElementById("job-name").textContent = job.title || jobId;
  setBadges({model: job.settings?.model || null, device: INFO.device,
             title: job.title, duration: job.duration});
  showActualEngine(job.precision?.engine || job.settings?.asrEngine);
  setStage(`${JOB_STATUS[job.status] || job.status} · iteración ${job.iteration + 1}`, false);
  setProgress(job.status === "completed" ? 1 : 0);
  showFiles(job.id, job.files || {}, job.links);
  if (job.reference) showReference({...job.reference, section: job.section, stage: "loaded"});
  setCancelButton(job.id, job.status === "running");
  if (job.status === "running") {
    localStorage.setItem("job_id", job.id);
    listen(job.id);
  }
}

async function prepareSavedIteration(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`, {cache: "no-store"});
  const job = await response.json();
  if (!response.ok) {
    return showAlert(job.error || "No pude cargar el trabajo.", {
      title: "No se pudo cargar",
    });
  }
  setMode("ultra");
  ITERATE_JOB_ID = jobId;
  SELECTED_JOB = jobId;
  EXPANDED_JOB = null;
  renderJobs();
  syncGoLabel();
  const settings = job.settings || {};
  const modelSelect = document.getElementById("model");
  const savedModel = [...modelSelect.options].find(
    option => option.value === settings.model && !option.disabled);
  const cachedUltra = INFO.models?.find(model => model.ultra_cached);
  if (savedModel) modelSelect.value = savedModel.value;
  else if (cachedUltra) modelSelect.value = cachedUltra.name;
  if (settings.precisionLevel) {
    document.getElementById("precision_level").value = settings.precisionLevel;
  }
  if (settings.asrEngine) {
    document.getElementById("asr_engine").value = settings.asrEngine;
  }
  document.getElementById("rescue_original").checked =
    settings.rescueOriginal !== false;
  document.getElementById("translate_to").value = settings.translateTo || "";
  document.getElementById("romaji").checked = Boolean(settings.romaji);
  document.getElementById("fetch_lyrics").checked = settings.fetchLyrics !== false;
  onPrecisionChange();
  onTranslateChange();
  document.getElementById("tracklist_text").value = job.tracklistText || "";
  if (job.tracklistText) await previewTracklist();
  const state = {
    title: job.title, duration: job.duration, recoverable: job.recoverable,
  };
  showRecovery(jobId, state);
  document.getElementById("err").textContent = job.recoverable
    ? "✅ Trabajo cargado. Ajusta las opciones y vuelve a procesar; la salida anterior se archivará."
    : "⚠️ Este trabajo no conserva voz aislada ni asr-input.wav.";
  setView("new");
}

async function translateSavedJob(jobId) {
  const target = document.getElementById("translate_to").value;
  const romanize = document.getElementById("romaji").checked;
  if (!target && !romanize) {
    setMode("ultra");
    setView("new");
    document.getElementById("translate_to").focus();
    return showAlert("Elige en el formulario un idioma de traducción o activa romaji.", {
      title: "Falta una opción de idioma",
      kind: "info",
    });
  }
  const fd = new FormData();
  fd.append("translate_to", target);
  fd.append("romaji", romanize ? "1" : "0");
  let response, data;
  try {
    response = await fetch(`/api/jobs/${jobId}/translate`, {
      method: "POST", body: fd,
    });
    data = await response.json();
  } catch (_) {
    return showAlert("No se pudo contactar con el servidor.", {
      title: "Scribix no responde",
    });
  }
  if (!response.ok) {
    return showAlert(data.error || "No pude iniciar la traducción.", {
      title: "No se inició la traducción",
    });
  }
  const job = await (await fetch(`/api/jobs/${jobId}`)).json();
  SELECTED_JOB = jobId;
  EXPANDED_JOB = null;
  renderJobs();
  enableJobView(true);
  setView("job");
  document.getElementById("err").textContent = "";
  document.getElementById("job-name").textContent = job.title || jobId;
  document.getElementById("go").disabled = true;
  resetPanels();
  setBadges({model: null, device: INFO.device, title: job.title,
             duration: job.duration});
  setStage("Traduciendo el resultado guardado…", true);
  setProgress(0);
  localStorage.setItem("job_id", jobId);
  listen(jobId);
}

// ----------------------------------------------------------------------- //
// Catálogos del servidor                                                    //
// ----------------------------------------------------------------------- //

async function loadInfo() {
  INFO = await (await fetch("/api/info")).json();
  document.getElementById("device-badge").textContent = INFO.device;
  const ms = document.getElementById("model");
  INFO.models.forEach(m => {
    const o = document.createElement("option");
    o.value = m.name;
    o.dataset.cached = m.cached ? "1" : "";
    o.dataset.ultraCached = m.ultra_cached ? "1" : "";
    if (m.name === "medium") o.selected = true;
    ms.appendChild(o);
  });
  syncModelLabels();
  const ls = document.getElementById("language");
  INFO.languages.forEach(l => {
    const o = document.createElement("option");
    o.value = l.value; o.textContent = l.label;
    ls.appendChild(o);
  });
  const fs = document.getElementById("audio_format");
  INFO.audio_formats.forEach(f => {
    const o = document.createElement("option");
    o.value = f.value; o.textContent = f.label;
    if (f.value === "mp3") o.selected = true;   // el más compatible
    fs.appendChild(o);
  });
  const qs = document.getElementById("audio_quality");
  INFO.audio_qualities.forEach(q => {
    const o = document.createElement("option");
    o.value = q; o.textContent = q.replace("K", " kbps");
    if (q === "192K") o.selected = true;
    qs.appendChild(o);
  });
  const vfs = document.getElementById("video_format");
  (INFO.video_formats || []).forEach(format => {
    const option = document.createElement("option");
    option.value = format.value;
    option.textContent = format.label;
    if (format.value === "mp4") option.selected = true;
    vfs.appendChild(option);
  });
  const vqs = document.getElementById("video_quality");
  (INFO.video_qualities || []).forEach(quality => {
    const option = document.createElement("option");
    option.value = quality.value;
    option.textContent = quality.label;
    if (quality.value === "1080") option.selected = true;
    vqs.appendChild(option);
  });
  const ss = document.getElementById("separation_quality");
  INFO.separation_profiles.forEach(p => {
    const o = document.createElement("option");
    o.value = p.value;
    o.textContent = p.label + (p.available ? "" : " — no instalado");
    o.disabled = !p.available;
    if (p.value === "precise" && p.available) o.selected = true;
    ss.appendChild(o);
  });
  if (!ss.value || ss.selectedOptions[0]?.disabled) {
    const fallback = INFO.separation_profiles.find(p => p.available);
    if (fallback) ss.value = fallback.value;
  }
  const ps = document.getElementById("precision_level");
  (INFO.precision_levels || []).forEach(p => {
    const o = document.createElement("option");
    o.value = p.value; o.textContent = p.label;
    if (p.value === "balanced") o.selected = true;
    ps.appendChild(o);
  });
  const engines = document.getElementById("asr_engine");
  (INFO.asr_engines || []).forEach(engine => {
    const o = document.createElement("option");
    o.value = engine.value;
    o.textContent = engine.label + (engine.available ? "" : " — no instalado");
    o.disabled = !engine.available;
    if (engine.value === "auto") o.selected = true;
    engines.appendChild(o);
  });
  const ts = document.getElementById("translate_to");
  const none = document.createElement("option");
  none.value = ""; none.textContent = "Sin traducción";
  ts.appendChild(none);
  (INFO.target_languages || []).forEach(l => {
    const o = document.createElement("option");
    o.value = l.value; o.textContent = l.label;
    ts.appendChild(o);
  });
  ts.value = "";
  onEngineChange();
  onModelChange();
  onPrecisionChange();
  onTranslateChange();
  onSeparationChange();
  onFormatChange();
  onVideoFormatChange();
  syncAudioOpts();
}

// ----------------------------------------------------------------------- //
// Paneles del trabajo en curso                                              //
// ----------------------------------------------------------------------- //

const VERDICTS = {
  ok: ["verdict-ok", "✅ Limpia"],
  bleed: ["verdict-warn", "⚠️ Queda instrumental"],
  clipping: ["verdict-warn", "⚠️ Satura"],
  instrumental: ["verdict-warn", "⚠️ Casi no hay canto"],
  silent: ["verdict-bad", "⛔ Voz en silencio"],
  empty: ["verdict-bad", "⛔ Pista vacía"],
  unknown: ["", "· sin medir"],
};

function showStem(d) {
  const [cls, label] = VERDICTS[d.verdict] || VERDICTS.unknown;
  const where = d.stage === "final" ? "pista que recibe Whisper"
    : `pasada ${d.pass_number} del separador`;
  const html =
    `<div class="metric"><span class="${cls}">${label}</span>` +
    `<cite>· ${where}</cite></div><br>` +
    `<span class="metric"><cite>Canto</cite> <b>${(d.speechCoverage * 100).toFixed(0)}%</b>` +
    ` <cite>(${fmt(d.speechSeconds)})</cite></span>` +
    `<span class="metric"><cite>Nivel</cite> <b>${d.vocalDbfs} dBFS</b></span>` +
    `<span class="metric"><cite>Fuga instrumental</cite> <b>${d.bleedRatio}</b></span>`;
  const body = document.getElementById("stem-body");
  body.innerHTML = html + (d.notes || []).map(() => `<p class="hint"></p>`).join("");
  (d.notes || []).forEach((note, i) => {
    body.querySelectorAll("p")[i].textContent = note;
  });
  document.getElementById("stem-panel").classList.remove("hidden");
}

function pct(value) {
  return value == null ? "—" : (value * 100).toFixed(1) + "%";
}

function renderSegment(d) {
  // Una fila de letra en la vista. La usan tanto el stream en vivo como el
  // `snapshot` de reconexión: una recarga pinta exactamente igual.
  const row = document.createElement("div");
  // Menos del 80% de acuerdo entre pasadas = la línea que hay que escuchar.
  const shaky = d.agreement != null && d.agreement < 0.8;
  row.className = shaky ? "seg shaky" : "seg";
  row.innerHTML = `<span class="t">${fmtMs(d.start)} → ${fmtMs(d.end)}</span>` +
                  `<span class="x"></span>` +
                  (d.referenceScore ? `<span class="ref">🧭 ${(d.referenceScore * 100).toFixed(0)}%</span>` : "") +
                  (shaky ? `<span class="ag">${pct(d.agreement)}</span>` : "");
  row.querySelector(".x").textContent = d.text;
  if (d.track?.title) row.title = `${d.track.artist || ""} — ${d.track.title}`;
  document.getElementById("segments").appendChild(row);
}

function showRound(d) {
  const body = document.getElementById("rounds-body");
  // Una fila por número de pasadas: la ronda se reescribe, no se duplica.
  let row = body.querySelector(`tr[data-passes="${d.passes}"]`);
  if (!row) {
    row = document.createElement("tr");
    row.dataset.passes = d.passes;
    body.appendChild(row);
  }
  row.innerHTML =
    `<td>${d.passes}</td><td>${d.words}</td><td>${d.dropped}</td>` +
    `<td>${d.looped ?? 0}</td>` +
    `<td>${pct(d.meanAgreement)}</td><td>${pct(d.unanimous)}</td>` +
    `<td>${pct(d.stability)}</td>`;
  document.getElementById("rounds-panel").classList.remove("hidden");
  document.getElementById("rounds-note").textContent = d.fallback
    ? "⚠️ Las pasadas no se pudieron alinear de forma segura. Se conservó la "
      + "lectura principal en vez de borrar toda la letra; revisa las líneas en ámbar."
    : "Descartadas = palabras que una pasada oyó y la mayoría no: casi siempre " +
    "texto inventado. Bucles = repeticiones descontroladas en las que todas " +
    "las pasadas se engancharon a la vez, así que el voto no podía verlas. " +
    "Estabilidad = cuánto coincide esta ronda con la anterior.";
}

function showPrecision(d) {
  showActualEngine(d.engine);
  const context = `motor ${d.engine} · ${d.windows} ventanas de voz ` +
    `(${fmt(d.speechSeconds)} de canto) · ${d.droppedSlots} palabras ` +
    `descartadas por falta de acuerdo` +
    (d.loopedWords ? ` y ${d.loopedWords} recortadas de bucles` : "") +
    (d.rescuedWindows ? ` · ${d.rescuedWindows} zonas rescatadas con ${d.rescueModel}` : "") + ".";
  let verdict;
  if (d.consensusFallback) {
    verdict = "⚠️ El voto no pudo alinear las pasadas; se conservó la lectura "
      + "principal para no perder toda la letra. Revisa las líneas en ámbar · ";
  } else if (d.converged) {
    verdict = `✅ Convergió con ${d.passes} pasadas · `;
  } else if (!d.checked) {
    // Nivel Rápido: una sola ronda, así que no hay con qué comparar.
    verdict = `✅ ${d.passes} pasadas votadas (sin comprobar convergencia) · `;
  } else {
    verdict = `⚠️ Se agotaron las ${d.passes} pasadas sin llegar al objetivo ` +
      `de estabilidad; revisa las líneas en ámbar · `;
  }
  document.getElementById("rounds-note").textContent = verdict + context;

  // Un mix de una hora no tiene un solo idioma: si lo tuviera, forzarlo es más
  // estable; si no, cada tramo se transcribe en el suyo.
  const langs = d.languages;
  if (!langs || !langs.dominant) return;
  const note = document.createElement("p");
  note.className = "hint";
  const parts = Object.entries(langs.seconds || {})
    .map(([code, secs]) => `${code} ${fmt(secs)}`).join(" · ");
  note.textContent = langs.multilingual
    ? `🌍 Mezcla multilingüe: ${parts}. Cada tramo se transcribió en su propio `
      + `idioma y se traduce con su propia ruta.`
    : `🌍 Un solo idioma: ${langs.dominant} (${(langs.share * 100).toFixed(0)}% `
      + `del canto).`;
  const panel = document.getElementById("rounds-panel");
  panel.querySelector(".lang-note")?.remove();
  note.classList.add("lang-note");
  panel.appendChild(note);
}

function showActualEngine(engine) {
  document.getElementById("actual-engine")?.remove();
  if (!engine) return;
  const actual = engine === "auto" ? (INFO?.auto_engine || engine) : engine;
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.id = "actual-engine";
  badge.textContent = `Motor real ${actual}`;
  document.getElementById("badges").appendChild(badge);
}

function showTranslation(d) {
  const body = document.getElementById("translation-body");
  const rows = [];
  if (d.romajiBackend && d.romajiBackend !== "none") {
    rows.push(`<span class="metric"><cite>Romaji</cite> <b>${esc(d.romajiBackend)}</b></span>`);
  }
  if (d.routeLabel) {
    rows.push(`<span class="metric"><cite>Ruta</cite> <b>${esc(d.routeLabel)}</b></span>`);
  }
  if (d.backend) {
    rows.push(`<span class="metric"><cite>Motor</cite> <b>${esc(d.backend)}</b></span>`);
  }
  body.innerHTML = rows.join("") || "<cite>Sin traducción.</cite>";
  if (d.error) {
    const p = document.createElement("p");
    p.className = "warn";
    p.textContent = "⚠️ " + d.error;
    body.appendChild(p);
  } else if (d.targetLanguage) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "La traducción viaja en el .lyrixa.json como capa propia "
      + "(layer-backing), con los mismos tiempos que la letra original.";
    body.appendChild(p);
  }
  document.getElementById("translation-panel").classList.remove("hidden");
}

function showReference(d) {
  const body = document.getElementById("reference-body");
  const total = d.total ?? 0;
  const found = d.found ?? 0;
  const synced = d.synced ?? 0;
  const failed = d.failed ?? 0;
  const noMatch = d.noMatch ?? Math.max(0, total - found - failed);
  const matched = d.matchedLines;
  body.innerHTML = "";
  const summary = document.createElement("p");
  summary.className = "hint";
  summary.style.marginTop = "0";
  summary.textContent = `${d.section || "Tracklist"} · ${total} canciones · `
    + `${found} letras encontradas (${synced} sincronizadas)`
    + (noMatch ? ` · ${noMatch} sin coincidencia` : "")
    + (failed ? ` · ${failed} búsquedas fallidas` : "")
    + (matched != null ? ` · ${matched} líneas confirmadas con ortografía exacta` : "");
  body.appendChild(summary);
  if (failed) {
    const firstFailure = (d.tracks || []).find(track => track.lookup?.error);
    const warning = document.createElement("p");
    warning.className = "warn";
    warning.textContent = "⚠️ LRCLIB no respondió correctamente"
      + (firstFailure ? `: ${firstFailure.lookup.error}` : ". Se reintentará automáticamente.");
    body.appendChild(warning);
  }
  if (d.stage === "prepared") {
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = "Cada ventana de Whisper recibe solo la canción de ese "
      + "tramo, y solo si la referencia pasó los filtros de artista y duración.";
    body.appendChild(note);
  }
  document.getElementById("reference-panel").classList.remove("hidden");
}

// ----------------------------------------------------------------------- //
// Lanzar un trabajo                                                         //
// ----------------------------------------------------------------------- //

async function start() {
  // "Volver a iterar" debe conservar el mismo job y entrar por /resume.
  // Antes esta función creaba otro ID y repetía la separación completa.
  if (ITERATE_JOB_ID) return resumeSavedJob(ITERATE_JOB_ID);
  const model = document.getElementById("model").value;
  const language = document.getElementById("language").value;
  const fd = new FormData();
  fd.append("source_type", SOURCE);
  fd.append("model", model);
  fd.append("language", language);
  fd.append("mode", MODE);
  fd.append("audio_format", document.getElementById("audio_format").value);
  fd.append("audio_quality", document.getElementById("audio_quality").value);
  fd.append("video_format", document.getElementById("video_format").value);
  fd.append("video_quality", document.getElementById("video_quality").value);
  fd.append("separation_quality", document.getElementById("separation_quality").value);
  fd.append("precision_level", document.getElementById("precision_level").value);
  fd.append("asr_engine", document.getElementById("asr_engine").value);
  fd.append("rescue_original", document.getElementById("rescue_original").checked ? "1" : "0");
  fd.append("translate_to", MODE === "ultra"
    ? document.getElementById("translate_to").value : "");
  fd.append("romaji", document.getElementById("romaji").checked ? "1" : "0");
  fd.append("tracklist_text", MODE === "ultra"
    ? document.getElementById("tracklist_text").value : "");
  fd.append("fetch_lyrics", document.getElementById("fetch_lyrics").checked ? "1" : "0");

  if (MODE === "audio_only" || MODE === "video_only") {
    // Las descargas directas funcionan con un link de YouTube.
    const url = document.getElementById("url").value.trim();
    if (!url) {
      const kind = MODE === "video_only" ? "video" : "audio";
      return showAlert(`Pega un link de YouTube para descargar el ${kind}.`, {
        title: "Falta el enlace",
        kind: "info",
      });
    }
    fd.append("url", url);
  } else if (SOURCE === "url") {
    const url = document.getElementById("url").value.trim();
    if (!url) {
      return showAlert("Pega un link de YouTube para comenzar.", {
        title: "Falta el enlace",
        kind: "info",
      });
    }
    fd.append("url", url);
  } else {
    const f = document.getElementById("file").files[0];
    if (!f) {
      return showAlert("Elige un archivo de audio o video para comenzar.", {
        title: "Falta el archivo",
        kind: "info",
      });
    }
    fd.append("file", f);
  }

  document.getElementById("go").disabled = true;
  enableJobView(true);
  setView("job");
  document.getElementById("job-name").textContent = "Trabajo nuevo";
  clearSegmentView();
  document.getElementById("downloads").classList.add("hidden");
  document.getElementById("downloads").innerHTML = "";
  document.getElementById("err").textContent = "";
  resetPanels();
  setBadges({ model: ["audio_only", "video_only"].includes(MODE) ? null : model,
              device: INFO.device });
  setStage("Iniciando…", true);
  setProgress(0);

  let payload;
  try {
    const res = await fetch("/api/jobs", { method: "POST", body: fd });
    payload = await res.json();
  } catch (_) {
    // El servidor cayó a mitad del POST (o se fue la red): sin esto el botón
    // "Empezar" quedaba inutilizable hasta recargar la pestaña.
    document.getElementById("err").textContent =
      "⚠️ No se pudo contactar con el servidor. ¿Sigue corriendo?";
    setStage("Error", false);
    document.getElementById("go").disabled = false;
    return;
  }
  if (!payload.job_id) {
    document.getElementById("err").textContent =
      "⚠️ " + (payload.error || "No se pudo iniciar el trabajo.");
    setStage("Error", false);
    document.getElementById("go").disabled = false;
    return;
  }
  const { job_id } = payload;
  SELECTED_JOB = job_id;
  localStorage.setItem("job_id", job_id);  // para retomar si se recarga la pestaña
  listen(job_id);
}

function esc(s) {
  // Textura 100% defensiva: títulos de video, mensajes de stage y nombres de
  // motor vienen del servidor/SSE y aterrizan en `innerHTML`. Escaparlos aquí
  // en un solo punto evita que un título like `<img src=x onerror=…>` ejecute
  // JS contra esta API local (que incluye DELETE /api/jobs/{id}).
  return String(s ?? "").replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setBadges({ model, device, title, duration }) {
  const b = document.getElementById("badges");
  // Todo lo que entra aquí (título de YouTube, motor, device) puede llevar
  // HTML: `esc` lo vuelve texto antes de tocar `innerHTML`.
  let html = model
    ? `<span class="badge"><span class="dot"></span>Modelo <b>${esc(model)}</b></span>`
    : "";
  html += `<span class="badge">Dispositivo <b>${esc(device)}</b></span>`;
  if (duration) html += `<span class="badge">Duración <b>${esc(fmt(duration))}</b></span>`;
  b.innerHTML = html;
  if (title) document.getElementById("job-name").textContent = title;
}

function setStage(text, spinning) {
  // `text` puede traer el título del video o un mensaje crudo del servidor.
  const stage = document.getElementById("stage");
  stage.textContent = "";
  if (spinning) {
    const spin = document.createElement("span");
    spin.className = "spin";
    spin.textContent = "⏳";
    stage.append(spin, " ");
  }
  stage.append(text ?? "");
}

function resetPanels() {
  for (const id of ["stem-panel", "rounds-panel", "translation-panel", "reference-panel"]) {
    document.getElementById(id).classList.add("hidden");
  }
  document.getElementById("rounds-body").innerHTML = "";
  document.getElementById("stem-body").innerHTML = "";
  document.getElementById("translation-body").innerHTML = "";
  document.getElementById("reference-body").innerHTML = "";
}

function setProgress(p) {
  const pct = p == null ? null : Math.round(p * 100);
  document.getElementById("bar").style.width = (pct == null ? 3 : pct) + "%";
  document.getElementById("pct").textContent = pct == null ? "…" : pct + "%";
}

function finishJob() {
  // Trabajo terminado (bien o mal): dejamos de retomarlo al recargar.
  localStorage.removeItem("job_id");
  ITERATE_JOB_ID = null;
  syncGoLabel();
  document.getElementById("go").disabled = false;
  document.getElementById("recovery-actions").classList.add("hidden");
  setCancelButton(null, false);
}

function showFiles(jobId, files, links) {
  const dl = document.getElementById("downloads");
  dl.classList.remove("hidden");
  dl.innerHTML = "";
  // El backend arma este enlace porque es el único que sabe dónde vive
  // Lyrixa y bajo qué dirección se ve a sí mismo. Componerlo aquí obligaría
  // a la interfaz a conocer puertos, que es justo lo que `config` evita.
  if (links && links.openInLyrixa && files.lyrixa) {
    const open = document.createElement("a");
    open.href = links.openInLyrixa;
    open.target = "_blank";
    open.rel = "noopener";
    open.className = "open-in-lyrixa";
    open.textContent = "🎛️ Abrir en Lyrixa";
    open.title = "Abre Lyrixa con este trabajo cargado: proyecto y audio, sin "
               + "descargar nada. Necesita Lyrixa corriendo.";
    dl.appendChild(open);
  }
  for (const [kind, icon, label] of FILE_KINDS) {
    const descriptor = files[kind];
    if (!descriptor) continue;
    const name = typeof descriptor === "string" ? descriptor : descriptor.name;
    const url = typeof descriptor === "string"
      ? `/download/${jobId}/${encodeURIComponent(name)}`
      : descriptor.url;
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.title = name;
    if (kind === "lyrixa") {
      a.title = "En Lyrixa: More → Import project (.lyrixa.json). Después carga el mismo audio.";
    }
    const ext = name.includes(".") ? name.split(".").pop() : "";
    a.textContent = `${icon} ${label}` + (kind === "audio" ? ` (.${ext})` : "");
    const small = document.createElement("b");
    small.textContent = name;
    a.appendChild(small);
    dl.appendChild(a);
    if (kind === "vocals" || kind === "instrumental") {
      const player = document.createElement("div");
      player.className = "stem-player";
      const playerLabel = document.createElement("label");
      const playerName = `${icon} Escuchar ${label.toLowerCase()}`;
      playerLabel.textContent = playerName;
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = url;
      // <audio> no es etiquetable: el <label> visible no asocia con for;
      // el reproductor se anuncia con su propio nombre.
      audio.setAttribute("aria-label", playerName);
      player.append(playerLabel, audio);
      dl.appendChild(player);
    }
  }
}

function showRecovery(jobId, state) {
  enableJobView(true);
  // No hay proceso vivo que cancelar. Esta vista puede aparecer mientras el
  // botón aún dice “Cancelando…” tras perderse la conexión.
  setCancelButton(jobId, false);
  clearSegmentView();
  document.getElementById("downloads").classList.add("hidden");
  resetPanels();
  setMode("ultra");
  const cachedUltra = INFO.models?.find(model => model.ultra_cached);
  if (cachedUltra) document.getElementById("model").value = cachedUltra.name;
  setBadges({ model: null, device: INFO.device, title: state.title,
              duration: state.duration });
  setProgress(0);
  setStage("La sesión de macOS cerró este proceso", false);
  document.getElementById("err").textContent = state.recoverable
    ? "✅ La separación quedó guardada. Puedes retomar sin repetirla. Revisa el modelo, idioma, precisión y traducción en el formulario."
    : "⚠️ No quedó un checkpoint suficiente para retomar este trabajo.";
  document.getElementById("go").disabled = false;

  const actions = document.getElementById("recovery-actions");
  actions.innerHTML = "";
  if (state.recoverable) {
    const resume = document.createElement("button");
    resume.className = "primary";
    resume.textContent = "▶️ Retomar desde el audio guardado";
    resume.onclick = () => resumeSavedJob(jobId);
    actions.appendChild(resume);
  }
  const discard = document.createElement("button");
  discard.textContent = "Descartar aviso";
  discard.onclick = () => {
    localStorage.removeItem("job_id");
    actions.classList.add("hidden");
    setView("new");
  };
  actions.appendChild(discard);
  actions.classList.remove("hidden");
}

async function resumeSavedJob(jobId) {
  const fd = new FormData();
  fd.append("model", document.getElementById("model").value);
  fd.append("language", document.getElementById("language").value);
  fd.append("precision_level", document.getElementById("precision_level").value);
  fd.append("asr_engine", document.getElementById("asr_engine").value);
  fd.append("rescue_original", document.getElementById("rescue_original").checked ? "1" : "0");
  fd.append("translate_to", document.getElementById("translate_to").value);
  fd.append("romaji", document.getElementById("romaji").checked ? "1" : "0");
  fd.append("tracklist_text", document.getElementById("tracklist_text").value);
  fd.append("fetch_lyrics", document.getElementById("fetch_lyrics").checked ? "1" : "0");

  const actions = document.getElementById("recovery-actions");
  actions.classList.add("hidden");
  document.getElementById("err").textContent = "";
  document.getElementById("go").disabled = true;
  enableJobView(true);
  setView("job");
  setStage("Retomando desde asr-input.wav…", true);
  let res, payload;
  try {
    res = await fetch(`/api/jobs/${jobId}/resume`, {method: "POST", body: fd});
    payload = await res.json();
  } catch (_) {
    document.getElementById("err").textContent =
      "⚠️ No se pudo contactar con el servidor para retomar.";
    setStage("No se pudo retomar", false);
    document.getElementById("go").disabled = false;
    actions.classList.remove("hidden");
    return;
  }
  if (!res.ok) {
    document.getElementById("err").textContent =
      "⚠️ " + (payload.error || "No se pudo retomar el trabajo.");
    setStage("No se pudo retomar", false);
    document.getElementById("go").disabled = false;
    actions.classList.remove("hidden");
    return;
  }
  SELECTED_JOB = jobId;
  localStorage.setItem("job_id", jobId);
  listen(jobId);
}

async function interruptedState(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}/state`, {cache: "no-store"});
    if (res.status === 404) return {status: "missing", recoverable: false};
    return await res.json();
  } catch (_) {
    return null;  // servidor realmente desconectado: EventSource debe reintentar
  }
}

// ----------------------------------------------------------------------- //
// Stream de eventos                                                         //
// ----------------------------------------------------------------------- //

let ACTIVE_EVENTS = null;
// Último `id:` SSE oído en esta sesión de página, del trabajo escuchado.
// Al reconectar (corte de red o pestaña reabierta) vale como `?since=`: así
// no reenvían desde el evento 0 miles de segmentos ya pintados. Va unido a
// la vista de segmentos: si la vista se limpia, el único punto de reanudo
// honesto es el principio del anillo del servidor (clearSegmentView). Si al
// pedir since el servidor ya no lo retiene, manda `reset` y vaciamos la
// vista antes del replay.
let LAST_EVENT = { job: null, id: 0 };

// Única forma de vaciar la vista de segmentos: olvida también el since.
function clearSegmentView() {
  LAST_EVENT = { job: null, id: 0 };
  document.getElementById("segments").innerHTML = "";
}

function listen(jobId) {
  setCancelButton(jobId, true);
  // EventSource se reconecta solo tras un corte (dormir la Mac, WiFi, etc.) y,
  // gracias a los `id:` del servidor, retoma justo donde se quedó, sin duplicar.
  // Solo puede haber una escucha viva: si arranca otro trabajo (o se reabre la
  // pestaña sobre otro job), la sesión vieja debe morir o sus eventos seguirían
  // pintando paneles ajenos en la vista nueva.
  if (ACTIVE_EVENTS) ACTIVE_EVENTS.close();
  // Si ya tenemos eventos pintados de este job (reconexión en la misma
  // página) pedimos solo lo que falta; si no, el anillo completo.
  const resume = (LAST_EVENT.job === jobId && LAST_EVENT.id > 0) ? LAST_EVENT.id : 0;
  LAST_EVENT = { job: jobId, id: resume };
  const es = new EventSource(
    `/api/jobs/${jobId}/events` + (resume ? `?since=${resume}` : ""));
  ACTIVE_EVENTS = es;
  // Cada mensaje trae `id:` (ev.lastEventId): lo apuntamos para el future
  // `?since=` y para que la reconexión automática de EventSource (cabecera
  // Last-Event-ID) coincida con lo que realmente hemos pintado.
  const on = (name, fn) => es.addEventListener(name, e => {
    const n = Number(e.lastEventId);
    if (Number.isFinite(n) && n > LAST_EVENT.id) LAST_EVENT.id = n;
    fn(e);
  });
  let duration = 0;
  let tries = 0;
  let checkingState = false;

  es.addEventListener("open", () => { tries = 0; });  // reconectó bien

  // El anillo del servidor podó eventos más antiguos que nuestro `since`:
  // llega el resumen acumulado (letra, stems, rondas retirados + el último
  // valor de cada panel) y la cola reanuda justo después. Reconstruimos todo.
  on("snapshot", e => {
    const s = JSON.parse(e.data);
    const L = s.latest || {};
    resetPanels();
    document.getElementById("segments").innerHTML = "";
    for (const d of (s.rounds || [])) showRound(d);
    for (const d of (s.stems || [])) showStem(d);
    for (const d of (s.segments || [])) renderSegment(d);
    const box = document.getElementById("segments");
    box.scrollTop = box.scrollHeight;
    const lastSeg = (s.segments || []).slice(-1)[0];
    if (lastSeg) document.getElementById("clock").textContent =
      `${fmt(lastSeg.end)}${duration ? " / " + fmt(duration) : ""}`;
    if (L.meta) { duration = L.meta.duration || duration; setBadges(L.meta); }
    if (L.progress) {
      setProgress(L.progress.progress);
      document.getElementById("clock").textContent =
        STAGE_LABELS[L.progress.stage] || "";
    }
    if (L.status) {
      setStage(L.status.message || L.status.stage, true);
      if (L.status.engine) showActualEngine(L.status.engine);
    }
    if (L.consensus) showRound(L.consensus);
    if (L.precision) showPrecision(L.precision);
    if (L.translation) showTranslation(L.translation);
    if (L.reference) showReference(L.reference);
  });

  on("status", e => {
    const d = JSON.parse(e.data);
    setStage(d.message || d.stage, true);
    if (d.model) setBadges({ model: d.model, device: d.device, duration: d.duration });
    if (d.engine) showActualEngine(d.engine);
  });
  on("meta", e => {
    const d = JSON.parse(e.data);
    duration = d.duration;
    setBadges({ model: d.model, device: d.device, title: d.title, duration: d.duration });
  });
  // Avance de las etapas largas que no son Whisper (descarga, Demucs, conversión).
  on("progress", e => {
    const d = JSON.parse(e.data);
    setProgress(d.progress);
    document.getElementById("clock").textContent = STAGE_LABELS[d.stage] || "";
  });
  on("stem", e => showStem(JSON.parse(e.data)));
  on("consensus", e => showRound(JSON.parse(e.data)));
  on("precision", e => showPrecision(JSON.parse(e.data)));
  on("translation", e => showTranslation(JSON.parse(e.data)));
  on("reference", e => showReference(JSON.parse(e.data)));
  on("rescue", e => {
    const d = JSON.parse(e.data);
    if (d.error) {
      document.getElementById("err").textContent =
        "⚠️ El rescate opcional no se pudo completar: " + d.error;
    }
  });
  on("segment", e => {
    const d = JSON.parse(e.data);
    if (d.progress != null) setProgress(d.progress);
    document.getElementById("clock").textContent =
      `${fmt(d.end)}${duration ? " / " + fmt(duration) : ""}`;
    renderSegment(d);
    const box = document.getElementById("segments");
    box.scrollTop = box.scrollHeight;
  });
  on("done", e => {
    const d = JSON.parse(e.data);
    setStage("✅ Listo — " + (d.title || ""), false);
    setProgress(1);
    document.getElementById("clock").textContent = "";
    showFiles(d.job_id, d.files, d.links);
    SELECTED_JOB = d.job_id;
    loadSavedJobs();
    es.close();
    finishJob();
  });
  on("cancelled", e => {
    const d = JSON.parse(e.data);
    setStage("Trabajo cancelado", false);
    document.getElementById("err").textContent = "ℹ️ "
      + (d.message || "El trabajo se canceló y Scribix sigue disponible.");
    document.getElementById("clock").textContent = "";
    es.close();
    finishJob();
    loadSavedJobs();
  });
  on("error", async e => {
    if (e.data) {
      // Error real de la app (p. ej. yt-dlp/whisper falló).
      let msg = "Ocurrió un error.";
      try { msg = JSON.parse(e.data).message; } catch (_) {}
      document.getElementById("err").textContent = "⚠️ " + msg;
      setStage("Error", false);
      es.close();
      finishJob();
      loadSavedJobs();
      return;
    }
    // Sin datos = se cayó la conexión. EventSource reintenta solo.
    tries++;
    if (!checkingState) {
      checkingState = true;
      const state = await interruptedState(jobId);
      checkingState = false;
      if (state?.status === "interrupted" || state?.status === "missing") {
        es.close();
        showRecovery(jobId, state);
        return;
      }
    }
    if (tries > 6) {
      document.getElementById("err").textContent =
        "⚠️ No se pudo reconectar. El trabajo pudo terminar o el servidor se cerró.";
      setStage("Sin conexión", false);
      es.close();
      finishJob();
      return;
    }
    setStage("🔌 Conexión perdida, reintentando… (la transcripción sigue en el servidor)", true);
  });
}

// Al cargar distinguimos entre una caída de red y un reinicio real del servidor.
async function resumeIfAny() {
  const jobId = localStorage.getItem("job_id");
  if (!jobId) return;
  const state = await interruptedState(jobId);
  if (state?.status === "interrupted" || state?.status === "missing") {
    showRecovery(jobId, state);
    setView("job");
    return;
  }
  SELECTED_JOB = jobId;
  renderJobs();
  enableJobView(true);
  setView("job");
  clearSegmentView();
  document.getElementById("downloads").classList.add("hidden");
  document.getElementById("err").textContent = "";
  document.getElementById("go").disabled = true;
  resetPanels();
  setStage("Retomando el trabajo en curso…", true);
  setProgress(0);
  listen(jobId);
}

// ----------------------------------------------------------------------- //
// Arranque                                                                  //
// ----------------------------------------------------------------------- //

function wireChrome() {
  document.querySelectorAll(".tabs[role=tablist]").forEach(wireTablist);

  const search = document.getElementById("job-search");
  search.addEventListener("input", () => { QUERY = search.value; renderJobs(); });
  search.addEventListener("keydown", ev => {
    if (ev.key !== "Escape") return;
    search.value = ""; QUERY = ""; renderJobs();
  });

  const sort = document.getElementById("job-sort");
  sort.addEventListener("change", () => { SORT = sort.value; renderJobs(); });

  // "/" enfoca la búsqueda desde cualquier sitio que no sea un campo de texto:
  // con la lista larga, llegar al buscador con el ratón es el gesto que más
  // se repite.
  document.addEventListener("keydown", ev => {
    if (ev.key !== "/" || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    ev.preventDefault();
    openSidebar();
    search.focus();
    search.select();
  });
  document.addEventListener("keydown", ev => {
    if (ev.key === "Escape") closeSidebar();
  });
}

async function boot() {
  wireChrome();
  setView("new");
  enableJobView(false);
  try {
    await loadInfo();
  } catch (_) {
    // Sin /api/info no hay modelos ni idiomas: lo decimos en pantalla en vez
    // de dejar una página muda que el usuario debe adivinar por qué.
    document.getElementById("boot-error").textContent =
      "⚠️ No hay servidor. Arranca `python3 server.py` y recarga la página.";
    return;
  }
  await loadSavedJobs();
  await resumeIfAny();
}

boot();
