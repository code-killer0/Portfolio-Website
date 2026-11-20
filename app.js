(() => {
  const canvas = document.querySelector("#ml-canvas");
  const epochValue = document.querySelector("#epoch-value");
  const accuracyValue = document.querySelector("#accuracy-value");
  const lossValue = document.querySelector("#loss-value");
  const modeCaption = document.querySelector("#mode-caption");
  const trainButton = document.querySelector("#train-button");
  const datasetButtons = [...document.querySelectorAll(".dataset-button")];
  let context;
  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let model;
  let samples = [];
  let mode = "xor";
  let epoch = 0;
  let running = false;
  let animationFrame = 0;
  let lastFrameAt = 0;
  let frames = 0;
  const totalEpochs = 640;
  const hiddenUnits = 8;
  const inputWeightRange = Math.sqrt(2 / 10);

  function randomNormal() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function createSamples(dataset) {
    const result = [];
    if (dataset === "xor") {
      const corners = [
        [-0.62, -0.62, 0],
        [-0.62, 0.62, 1],
        [0.62, -0.62, 1],
        [0.62, 0.62, 0],
      ];
      for (let i = 0; i < 240; i += 1) {
        const [cx, cy, label] = corners[i % corners.length];
        result.push({ x: cx + randomNormal() * 0.22, y: cy + randomNormal() * 0.22, label });
      }
    } else {
      for (let i = 0; i < 240; i += 1) {
        const label = i % 2;
        const angle = Math.random() * Math.PI;
        const noiseX = randomNormal() * 0.055;
        const noiseY = randomNormal() * 0.055;
        const x = label === 0 ? Math.cos(angle) : 1 - Math.cos(angle);
        const y = label === 0 ? Math.sin(angle) : 0.48 - Math.sin(angle);
        result.push({ x: (x - 0.5) / 1.6 + noiseX, y: y / 1.15 + noiseY, label });
      }
    }
    return result;
  }

  function freshModel() {
    const between = (range) => (Math.random() * 2 - 1) * range;
    return {
      w1: Array.from({ length: hiddenUnits }, () => [between(inputWeightRange), between(inputWeightRange)]),
      b1: Array.from({ length: hiddenUnits }, () => between(0.08)),
      w2: Array.from({ length: hiddenUnits }, () => between(Math.sqrt(2 / (hiddenUnits + 1)))),
      b2: between(0.08),
    };
  }

  function sigmoid(value) {
    if (value < -30) return 0;
    if (value > 30) return 1;
    return 1 / (1 + Math.exp(-value));
  }

  function forward(x, y) {
    const hidden = new Array(hiddenUnits);
    for (let j = 0; j < hiddenUnits; j += 1) {
      hidden[j] = Math.tanh(model.w1[j][0] * x + model.w1[j][1] * y + model.b1[j]);
    }
    let logit = model.b2;
    for (let j = 0; j < hiddenUnits; j += 1) logit += model.w2[j] * hidden[j];
    return { hidden, output: sigmoid(logit) };
  }

  function trainOne(sample) {
    const { hidden, output } = forward(sample.x, sample.y);
    const outputDelta = output - sample.label;
    const hiddenDeltas = new Array(hiddenUnits);
    for (let j = 0; j < hiddenUnits; j += 1) {
      hiddenDeltas[j] = outputDelta * model.w2[j] * (1 - hidden[j] * hidden[j]);
    }
    const learningRate = 0.045;
    for (let j = 0; j < hiddenUnits; j += 1) {
      model.w2[j] -= learningRate * outputDelta * hidden[j];
    }
    model.b2 -= learningRate * outputDelta;
    for (let j = 0; j < hiddenUnits; j += 1) {
      model.w1[j][0] -= learningRate * hiddenDeltas[j] * sample.x;
      model.w1[j][1] -= learningRate * hiddenDeltas[j] * sample.y;
      model.b1[j] -= learningRate * hiddenDeltas[j];
    }
  }

  function measure() {
    let correct = 0;
    let loss = 0;
    for (const sample of samples) {
      const probability = Math.min(1 - 1e-7, Math.max(1e-7, forward(sample.x, sample.y).output));
      if ((probability >= 0.5 ? 1 : 0) === sample.label) correct += 1;
      loss -= sample.label * Math.log(probability) + (1 - sample.label) * Math.log(1 - probability);
    }
    const accuracy = (correct / samples.length) * 100;
    const meanLoss = loss / samples.length;
    epochValue.textContent = String(epoch).padStart(3, "0");
    accuracyValue.innerHTML = `${accuracy.toFixed(1)}<span>%</span>`;
    lossValue.textContent = meanLoss.toFixed(3);
  }

  function trainEpoch() {
    for (let i = samples.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }
    for (const sample of samples) trainOne(sample);
    epoch += 1;
    measure();
  }

  function setCanvasSize() {
    if (!canvas || !context) return;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width < 10 || bounds.height < 10) return;
    width = bounds.width;
    height = bounds.height;
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    draw();
  }

  function draw() {
    if (!context || width < 10 || height < 10) return;
    const plot = { left: 20, top: 32, right: width - 10, bottom: height - 15 };
    plot.width = plot.right - plot.left;
    plot.height = plot.bottom - plot.top;
    context.clearRect(0, 0, width, height);

    const columns = Math.max(42, Math.round(plot.width / 4));
    const rows = Math.max(30, Math.round(plot.height / 4));
    const cellWidth = plot.width / columns + 0.65;
    const cellHeight = plot.height / rows + 0.65;
    for (let row = 0; row < rows; row += 1) {
      const y = 1.15 - ((row + 0.5) / rows) * 2.3;
      for (let column = 0; column < columns; column += 1) {
        const x = -1.15 + ((column + 0.5) / columns) * 2.3;
        const probability = forward(x, y).output;
        const isPositive = probability >= 0.5;
        const certainty = Math.abs(probability - 0.5) * 2;
        const alpha = (0.09 + certainty * 0.2).toFixed(3);
        context.fillStyle = isPositive ? `rgba(199,255,94,${alpha})` : `rgba(162,138,255,${alpha})`;
        context.fillRect(plot.left + column * plot.width / columns, plot.top + row * plot.height / rows, cellWidth, cellHeight);
      }
    }

    for (const sample of samples) {
      const px = plot.left + ((sample.x + 1.15) / 2.3) * plot.width;
      const py = plot.top + ((1.15 - sample.y) / 2.3) * plot.height;
      if (px < plot.left - 2 || px > plot.right + 2 || py < plot.top - 2 || py > plot.bottom + 2) continue;
      context.beginPath();
      context.arc(px, py, 2.05, 0, Math.PI * 2);
      context.fillStyle = sample.label ? "#d9ff9a" : "#c5b7ff";
      context.fill();
      context.lineWidth = 0.75;
      context.strokeStyle = "rgba(8,12,10,.74)";
      context.stroke();
    }
  }

  function frame(timestamp) {
    if (!running) return;
    if (timestamp - lastFrameAt < 45) {
      animationFrame = window.requestAnimationFrame(frame);
      return;
    }
    lastFrameAt = timestamp;
    for (let i = 0; i < 5 && epoch < totalEpochs; i += 1) trainEpoch();
    frames += 1;
    if (frames % 2 === 0 || epoch >= totalEpochs) draw();
    if (epoch >= totalEpochs) {
      running = false;
      trainButton.innerHTML = '<span aria-hidden="true">↻</span> RETRAIN';
      trainButton.removeAttribute("aria-busy");
      return;
    }
    animationFrame = window.requestAnimationFrame(frame);
  }

  function startTraining(nextMode = mode) {
    window.cancelAnimationFrame(animationFrame);
    running = false;
    mode = nextMode;
    samples = createSamples(mode);
    model = freshModel();
    epoch = 0;
    frames = 0;
    lastFrameAt = 0;
    modeCaption.textContent = mode === "xor" ? "XOR · 2 CLASSES" : "TWO MOONS · 2 CLASSES";
    datasetButtons.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    measure();
    draw();
    running = true;
    trainButton.innerHTML = '<span aria-hidden="true">↻</span> TRAINING';
    trainButton.setAttribute("aria-busy", "true");
    animationFrame = window.requestAnimationFrame(frame);
  }

  if (canvas) {
    context = canvas.getContext("2d", { alpha: true });
    datasetButtons.forEach((button) => button.addEventListener("click", () => startTraining(button.dataset.mode)));
    trainButton.addEventListener("click", () => startTraining(mode));
    window.addEventListener("resize", setCanvasSize, { passive: true });
    if ("ResizeObserver" in window) new ResizeObserver(setCanvasSize).observe(canvas);
    startTraining("xor");
    setCanvasSize();
  }

  const filterButtons = [...document.querySelectorAll(".filter-chip")];
  const cards = [...document.querySelectorAll(".project-card")];
  const emptyState = document.querySelector("#filter-empty");
  filterButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const filter = button.dataset.filter;
      filterButtons.forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("is-active", active);
        candidate.setAttribute("aria-pressed", String(active));
      });
      let visible = 0;
      cards.forEach((card) => {
        const categories = card.dataset.category.split(" ");
        const show = filter === "all" || categories.includes(filter);
        card.hidden = !show;
        if (show) visible += 1;
      });
      emptyState.hidden = visible > 0;
    });
  });

  const menuToggle = document.querySelector(".menu-toggle");
  const navigation = document.querySelector("#primary-nav");
  if (menuToggle && navigation) {
    menuToggle.addEventListener("click", () => {
      const expanded = menuToggle.getAttribute("aria-expanded") === "true";
      menuToggle.setAttribute("aria-expanded", String(!expanded));
      menuToggle.setAttribute("aria-label", expanded ? "Open navigation" : "Close navigation");
      navigation.classList.toggle("is-open", !expanded);
    });
    navigation.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
      menuToggle.setAttribute("aria-expanded", "false");
      menuToggle.setAttribute("aria-label", "Open navigation");
      navigation.classList.remove("is-open");
    }));
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && navigation.classList.contains("is-open")) {
        menuToggle.setAttribute("aria-expanded", "false");
        menuToggle.setAttribute("aria-label", "Open navigation");
        navigation.classList.remove("is-open");
        menuToggle.focus();
      }
    });
  }

  const revealTargets = [...document.querySelectorAll(
    ".proof-metric, .startup-card, .section-heading, .project-card, .learning-copy, .lab-card, .experience-item, .principle-card, .toolkit-group, .track-card, .achievement-list > div, .community-lines > div, .contact-main, .contact-links"
  )];
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (revealTargets.length && "IntersectionObserver" in window && !prefersReducedMotion) {
    document.querySelectorAll(".project-grid, .proof-strip, .experience-list, .principle-grid, .toolkit-grid, .track-grid, .achievement-list, .community-lines").forEach((group) => {
      [...group.children].forEach((item, index) => item.style.setProperty("--reveal-delay", `${Math.min(index, 5) * 95}ms`));
    });
    revealTargets.forEach((target) => target.classList.add("scroll-reveal"));
    document.documentElement.classList.add("reveal-ready");
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -48px 0px" });
    revealTargets.forEach((target) => revealObserver.observe(target));
  }

  const year = document.querySelector("#current-year");
  if (year) year.textContent = String(new Date().getFullYear());
})();
