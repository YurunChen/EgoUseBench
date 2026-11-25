function initializeReviewImages() {
  if (window.reviewImageEvents) window.reviewImageEvents.abort();
  window.reviewImageEvents = new AbortController();
  const signal = window.reviewImageEvents.signal;
  document.querySelector(".lightbox")?.remove();
  document.body.classList.remove("lightbox-open");
  document.querySelectorAll("[data-answer-details]").forEach(function (control) {
    control.addEventListener("click", function () {
      const shouldOpen = control.getAttribute("data-answer-details") === "open";
      document.querySelectorAll(".answer-workbench .entity-evidence").forEach(function (details) {
        details.open = shouldOpen;
      });
    });
  });

  const thumbs = Array.from(document.querySelectorAll(".frame-thumb"));
  if (!thumbs.length) {
    return;
  }

  const frames = thumbs.map(function (thumb) {
    return {
      src: thumb.getAttribute("data-src"),
      caption: thumb.getAttribute("data-caption") || "",
    };
  });

  const lightbox = document.createElement("div");
  lightbox.className = "lightbox";
  lightbox.hidden = true;
  lightbox.innerHTML =
    '<div class="lightbox-stage">' +
      '<button type="button" class="lightbox-nav" data-dir="-1" aria-label="上一帧">‹</button>' +
      '<img alt="">' +
      '<button type="button" class="lightbox-nav" data-dir="1" aria-label="下一帧">›</button>' +
    "</div>" +
    '<div class="lightbox-bar">' +
      "<p></p>" +
      '<button type="button" class="lightbox-close" aria-label="关闭">关闭</button>' +
    "</div>";
  document.body.appendChild(lightbox);

  const image = lightbox.querySelector("img");
  const caption = lightbox.querySelector(".lightbox-bar p");
  const prevButton = lightbox.querySelector('[data-dir="-1"]');
  const nextButton = lightbox.querySelector('[data-dir="1"]');
  let index = 0;
  let startX = null;

  function show(nextIndex) {
    index = (nextIndex + frames.length) % frames.length;
    image.src = frames[index].src;
    image.alt = frames[index].caption;
    caption.textContent = frames[index].caption + "（" + (index + 1) + "/" + frames.length + "）";
    prevButton.disabled = frames.length < 2;
    nextButton.disabled = frames.length < 2;
  }

  function open(nextIndex) {
    show(nextIndex);
    lightbox.hidden = false;
    document.body.classList.add("lightbox-open");
  }

  function close() {
    lightbox.hidden = true;
    document.body.classList.remove("lightbox-open");
  }

  thumbs.forEach(function (thumb, thumbIndex) {
    thumb.addEventListener("click", function () {
      open(thumbIndex);
    });
  });
  prevButton.addEventListener("click", function () {
    show(index - 1);
  });
  nextButton.addEventListener("click", function () {
    show(index + 1);
  });
  lightbox.querySelector(".lightbox-close").addEventListener("click", close);
  lightbox.addEventListener("click", function (event) {
    if (event.target === lightbox) {
      close();
    }
  });
  document.addEventListener("keydown", function (event) {
    if (lightbox.hidden) {
      return;
    }
    if (event.key === "Escape") {
      close();
    } else if (event.key === "ArrowLeft") {
      show(index - 1);
    } else if (event.key === "ArrowRight") {
      show(index + 1);
    }
  }, { signal: signal });

  lightbox.addEventListener("pointerdown", function (event) {
    if (event.target.closest("button")) {
      return;
    }
    startX = event.clientX;
  });
  lightbox.addEventListener("pointerup", function (event) {
    if (startX === null) {
      return;
    }
    const delta = event.clientX - startX;
    startX = null;
    if (delta > 60) {
      show(index - 1);
    } else if (delta < -60) {
      show(index + 1);
    }
  });
  lightbox.addEventListener("pointercancel", function () {
    startX = null;
  });
}
initializeReviewImages();
