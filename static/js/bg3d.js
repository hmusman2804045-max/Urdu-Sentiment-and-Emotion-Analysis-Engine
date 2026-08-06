/* =============================================================
   bg3d.js — ambient motion layer

   Three cooperating effects driven by ONE pointer listener and ONE
   rAF loop:
     1. a harmonic particle wave (Three.js)
     2. slow embers drifting up through the same 3D volume
     3. the DOM cursor spotlight + cursor-lit glass rims

   The spotlight is deliberately handled here rather than in main.js:
   it shares the pointer state and the frame loop with the WebGL layer,
   so the page pays for one listener and one rAF instead of two.

   Every piece degrades independently — no WebGL still leaves you with
   a working spotlight, and prefers-reduced-motion collapses the whole
   module to a single static frame.
   ============================================================= */

(function () {
  "use strict";

  var canvas = document.getElementById("bg-canvas");
  var spotlight = document.getElementById("mouse-spotlight");
  var root = document.documentElement;

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var finePointer = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

  /* ---------------------------------------------------------
     Shared pointer state
     --------------------------------------------------------- */

  // Normalised (-1..1) for the camera, raw pixels for the spotlight.
  var pointer = { nx: 0, ny: 0, px: window.innerWidth / 2, py: window.innerHeight / 2 };
  var camTarget = { x: 0, y: 0 };
  var spot = { x: pointer.px, y: pointer.py };
  var lastSpot = { x: -9999, y: -9999 };
  var pointerSeen = false;

  var webglReady = false;
  var frameId = null;
  var running = false;

  /* ---------------------------------------------------------
     Particle wave configuration
     --------------------------------------------------------- */

  var AMOUNT_X = 76;
  var AMOUNT_Y = 52;
  var SEPARATION = 58;
  var TOTAL = AMOUNT_X * AMOUNT_Y;

  var DUST_COUNT = 340;
  var DUST_SPAN_X = AMOUNT_X * SEPARATION;
  var DUST_SPAN_Z = AMOUNT_Y * SEPARATION;
  var DUST_FLOOR = -260;
  var DUST_CEIL = 1500;

  var scene, camera, renderer;
  var geometry, material, particles;
  var dustGeo, dustMat, dustPoints;
  var positions, scales, colors;
  var dustPos, dustVel, dustDrift;

  var time = 0;

  // Reusable scratch array so the per-frame wave math allocates nothing.
  var rowWave = new Float32Array(AMOUNT_Y);

  var COLOR_CYAN, COLOR_INDIGO, COLOR_VIOLET;

  /* ---------------------------------------------------------
     Shaders
     --------------------------------------------------------- */

  var WAVE_VERT = [
    "attribute float aScale;",
    "attribute vec3 aColor;",
    "attribute float aGlow;",
    "varying vec3 vColor;",
    "varying float vFade;",
    "varying float vGlow;",
    "void main() {",
    "  vColor = aColor;",
    "  vGlow = aGlow;",
    "  vec4 mv = modelViewMatrix * vec4(position, 1.0);",
    "  vFade = clamp(1.0 - (-mv.z - 400.0) / 2200.0, 0.05, 1.0);",
    "  gl_PointSize = aScale * (170.0 / max(-mv.z, 1.0));",
    "  gl_Position = projectionMatrix * mv;",
    "}"
  ].join("\n");

  // Crests read hotter than troughs: aGlow lifts both the core colour and
  // the alpha, so the wave reads as light moving through the grid.
  var WAVE_FRAG = [
    "varying vec3 vColor;",
    "varying float vFade;",
    "varying float vGlow;",
    "void main() {",
    "  vec2 uv = gl_PointCoord - vec2(0.5);",
    "  float d = length(uv);",
    "  if (d > 0.5) discard;",
    "  float core = smoothstep(0.5, 0.0, d);",
    "  float alpha = smoothstep(0.5, 0.02, d);",
    "  vec3 lit = vColor * (0.75 + vGlow * 1.15) + vec3(core * vGlow * 0.35);",
    "  gl_FragColor = vec4(lit, alpha * vFade * (0.55 + vGlow * 0.65));",
    "}"
  ].join("\n");

  var DUST_VERT = [
    "attribute float aScale;",
    "attribute float aSeed;",
    "attribute vec3 aColor;",
    "uniform float uTime;",
    "varying vec3 vColor;",
    "varying float vAlpha;",
    "void main() {",
    "  vColor = aColor;",
    "  vec4 mv = modelViewMatrix * vec4(position, 1.0);",
    "  float twinkle = 0.45 + 0.55 * sin(uTime * 1.7 + aSeed * 6.2831);",
    "  float depth = clamp(1.0 - (-mv.z - 300.0) / 2600.0, 0.0, 1.0);",
    "  vAlpha = twinkle * depth;",
    "  gl_PointSize = aScale * (150.0 / max(-mv.z, 1.0));",
    "  gl_Position = projectionMatrix * mv;",
    "}"
  ].join("\n");

  var DUST_FRAG = [
    "varying vec3 vColor;",
    "varying float vAlpha;",
    "void main() {",
    "  vec2 uv = gl_PointCoord - vec2(0.5);",
    "  float d = length(uv);",
    "  if (d > 0.5) discard;",
    "  float soft = smoothstep(0.5, 0.0, d);",
    "  gl_FragColor = vec4(vColor, soft * soft * vAlpha * 0.85);",
    "}"
  ].join("\n");

  /* ---------------------------------------------------------
     WebGL setup
     --------------------------------------------------------- */

  function initWebGL() {
    if (!canvas || typeof window.THREE === "undefined") return false;

    try {
      renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: false,
        alpha: true,
        powerPreference: "high-performance"
      });
    } catch (err) {
      console.warn("[bg3d] WebGL unavailable — particle layer disabled.", err);
      return false;
    }

    COLOR_CYAN = new THREE.Color(0x22d3ee);
    COLOR_INDIGO = new THREE.Color(0x6366f1);
    COLOR_VIOLET = new THREE.Color(0x8b5cf6);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    renderer.setClearColor(0x000000, 0);

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0b0f19, 0.00055);

    camera = new THREE.PerspectiveCamera(62, window.innerWidth / window.innerHeight, 1, 6000);
    camera.position.set(0, 380, 1180);

    buildWave();
    buildDust();

    canvas.classList.add("is-ready");
    return true;
  }

  function buildWave() {
    positions = new Float32Array(TOTAL * 3);
    scales = new Float32Array(TOTAL);
    colors = new Float32Array(TOTAL * 3);
    var glow = new Float32Array(TOTAL);

    var halfX = ((AMOUNT_X - 1) * SEPARATION) / 2;
    var halfY = ((AMOUNT_Y - 1) * SEPARATION) / 2;
    var maxDist = Math.sqrt(halfX * halfX + halfY * halfY);
    var tint = new THREE.Color();
    var i = 0;
    var j = 0;

    for (var ix = 0; ix < AMOUNT_X; ix++) {
      for (var iy = 0; iy < AMOUNT_Y; iy++) {
        var px = ix * SEPARATION - halfX;
        var pz = iy * SEPARATION - halfY;

        positions[i] = px;
        positions[i + 1] = 0;
        positions[i + 2] = pz;

        // Cyan core → indigo mid-field → violet at the rim.
        var dist = Math.sqrt(px * px + pz * pz) / maxDist;
        tint.copy(COLOR_CYAN).lerp(COLOR_INDIGO, Math.min(dist * 1.3, 1));
        tint.lerp(COLOR_VIOLET, Math.max(0, dist - 0.55) * 1.25);

        colors[i] = tint.r;
        colors[i + 1] = tint.g;
        colors[i + 2] = tint.b;

        scales[j] = 6;
        glow[j] = 0.5;

        i += 3;
        j++;
      }
    }

    geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("aScale", new THREE.BufferAttribute(scales, 1));
    geometry.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute("aGlow", new THREE.BufferAttribute(glow, 1));

    material = new THREE.ShaderMaterial({
      vertexShader: WAVE_VERT,
      fragmentShader: WAVE_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    particles = new THREE.Points(geometry, material);
    particles.rotation.x = -0.12;
    scene.add(particles);
  }

  function buildDust() {
    dustPos = new Float32Array(DUST_COUNT * 3);
    dustVel = new Float32Array(DUST_COUNT);
    dustDrift = new Float32Array(DUST_COUNT);

    var dScale = new Float32Array(DUST_COUNT);
    var dSeed = new Float32Array(DUST_COUNT);
    var dColor = new Float32Array(DUST_COUNT * 3);
    var tint = new THREE.Color();

    for (var k = 0; k < DUST_COUNT; k++) {
      var p = k * 3;

      dustPos[p] = (Math.random() - 0.5) * DUST_SPAN_X;
      dustPos[p + 1] = DUST_FLOOR + Math.random() * (DUST_CEIL - DUST_FLOOR);
      dustPos[p + 2] = (Math.random() - 0.5) * DUST_SPAN_Z;

      // Rise rate and lateral sway rate, both per-ember.
      dustVel[k] = 8 + Math.random() * 26;
      dustDrift[k] = 0.25 + Math.random() * 0.7;

      // Mostly cyan/violet with a few near-white embers for sparkle.
      var mix = Math.random();
      tint.copy(COLOR_CYAN).lerp(COLOR_VIOLET, mix);
      if (mix > 0.86) tint.lerp(new THREE.Color(0xffffff), 0.55);

      dColor[p] = tint.r;
      dColor[p + 1] = tint.g;
      dColor[p + 2] = tint.b;

      dScale[k] = 2.5 + Math.random() * 6.5;
      dSeed[k] = Math.random();
    }

    dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
    dustGeo.setAttribute("aScale", new THREE.BufferAttribute(dScale, 1));
    dustGeo.setAttribute("aSeed", new THREE.BufferAttribute(dSeed, 1));
    dustGeo.setAttribute("aColor", new THREE.BufferAttribute(dColor, 3));

    dustMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 } },
      vertexShader: DUST_VERT,
      fragmentShader: DUST_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    dustPoints = new THREE.Points(dustGeo, dustMat);
    scene.add(dustPoints);
  }

  /* ---------------------------------------------------------
     Per-frame simulation
     --------------------------------------------------------- */

  /**
   * Two interfering harmonic pairs.
   *
   * The primary pair is separable — one term depends only on the column,
   * the other only on the row — so the row half is hoisted into `rowWave`
   * and computed AMOUNT_Y times per frame instead of TOTAL times. Only the
   * secondary diagonal pair, which genuinely couples both axes, stays in
   * the inner loop. That keeps this at ~2 trig calls per particle.
   */
  function updateWave() {
    var iy, ix;

    for (iy = 0; iy < AMOUNT_Y; iy++) {
      rowWave[iy] = Math.cos((iy + time) * 0.32);
    }

    var glowAttr = geometry.attributes.aGlow.array;
    var i = 0;
    var j = 0;

    for (ix = 0; ix < AMOUNT_X; ix++) {
      var colWave = Math.sin((ix + time) * 0.28);
      var diagA = ix * 0.11 + time * 0.55;
      var diagB = ix * 0.07 - time * 0.38;

      for (iy = 0; iy < AMOUNT_Y; iy++) {
        var rowW = rowWave[iy];

        // Secondary pair travels diagonally at a different rate, so the
        // two systems beat against each other instead of repeating.
        var s2 = Math.sin(diagA + iy * 0.09);
        var c2 = Math.cos(diagB - iy * 0.13);

        positions[i + 1] = colWave * 26 + rowW * 26 + s2 * 18 + c2 * 13;

        // Normalised crest height (0 trough → 1 peak) drives size and glow.
        var amp = (colWave + rowW + s2 + c2) * 0.25;
        var lift = amp * 0.5 + 0.5;

        scales[j] = 2.2 + lift * 9.5;
        glowAttr[j] = lift * lift;

        i += 3;
        j++;
      }
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.aScale.needsUpdate = true;
    geometry.attributes.aGlow.needsUpdate = true;
  }

  /** Embers rise, sway, and wrap back to the floor when they clear the ceiling. */
  function updateDust(dt) {
    for (var k = 0; k < DUST_COUNT; k++) {
      var p = k * 3;

      dustPos[p + 1] += dustVel[k] * dt;
      dustPos[p] += Math.sin(time * dustDrift[k] + k) * 0.55;
      dustPos[p + 2] += Math.cos(time * dustDrift[k] * 0.7 + k) * 0.35;

      if (dustPos[p + 1] > DUST_CEIL) {
        dustPos[p] = (Math.random() - 0.5) * DUST_SPAN_X;
        dustPos[p + 1] = DUST_FLOOR;
        dustPos[p + 2] = (Math.random() - 0.5) * DUST_SPAN_Z;
      }
    }

    dustGeo.attributes.position.needsUpdate = true;
    dustMat.uniforms.uTime.value = time;
  }

  /** Damped camera parallax on both axes. */
  function updateCamera() {
    camTarget.x += (pointer.nx * 360 - camTarget.x) * 0.045;
    camTarget.y += (-pointer.ny * 210 - camTarget.y) * 0.045;

    camera.position.x = camTarget.x;
    camera.position.y = 380 + camTarget.y;
    camera.lookAt(0, 0, 0);

    particles.rotation.y = Math.sin(time * 0.012) * 0.06;
    if (dustPoints) dustPoints.rotation.y = particles.rotation.y * 0.5;
  }

  /* ---------------------------------------------------------
     Spotlight (DOM layer)
     --------------------------------------------------------- */

  function updateSpotlight() {
    if (!finePointer) return;

    spot.x += (pointer.px - spot.x) * 0.12;
    spot.y += (pointer.py - spot.y) * 0.12;

    // Sub-pixel churn would re-run style resolution for every .glass::after
    // ring on the page for no visible gain, so settle instead.
    if (Math.abs(spot.x - lastSpot.x) < 0.4 && Math.abs(spot.y - lastSpot.y) < 0.4) return;

    lastSpot.x = spot.x;
    lastSpot.y = spot.y;

    if (spotlight) {
      spotlight.style.transform = "translate3d(" + spot.x.toFixed(1) + "px," + spot.y.toFixed(1) + "px,0)";
    }

    root.style.setProperty("--spot-x", spot.x.toFixed(1) + "px");
    root.style.setProperty("--spot-y", spot.y.toFixed(1) + "px");
  }

  /* ---------------------------------------------------------
     Loop + lifecycle
     --------------------------------------------------------- */

  function renderFrame() {
    if (!webglReady) return;
    updateCamera();
    renderer.render(scene, camera);
  }

  var lastTs = 0;

  function animate(ts) {
    frameId = window.requestAnimationFrame(animate);

    // Delta-timed so ember speed is frame-rate independent, clamped so a
    // background tab returning after a long pause doesn't teleport them.
    var dt = lastTs ? Math.min((ts - lastTs) / 1000, 0.05) : 0.016;
    lastTs = ts;
    time += dt * 3.3;

    if (webglReady) {
      updateWave();
      updateDust(dt);
    }
    updateSpotlight();
    renderFrame();
  }

  function start() {
    if (running) return;
    running = true;

    if (reduceMotion) {
      // One static frame, no loop.
      if (webglReady) {
        updateWave();
        updateDust(0);
        renderFrame();
      }
      return;
    }

    lastTs = 0;
    frameId = window.requestAnimationFrame(animate);
  }

  function stop() {
    running = false;
    if (frameId !== null) {
      window.cancelAnimationFrame(frameId);
      frameId = null;
    }
  }

  function onResize() {
    if (!webglReady) return;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    if (!running || reduceMotion) renderFrame();
  }

  function onPointerMove(event) {
    pointer.px = event.clientX;
    pointer.py = event.clientY;
    pointer.nx = (event.clientX / window.innerWidth) * 2 - 1;
    pointer.ny = (event.clientY / window.innerHeight) * 2 - 1;

    if (!pointerSeen && finePointer) {
      pointerSeen = true;
      // Jump the halo to the cursor so it fades in where the pointer already
      // is rather than sliding in from the middle of the screen.
      spot.x = pointer.px;
      spot.y = pointer.py;
      root.classList.add("has-pointer");
    }
  }

  function onVisibility() {
    if (document.hidden) stop();
    else start();
  }

  function onContextLost(event) {
    event.preventDefault();
    webglReady = false;
    stop();
    console.warn("[bg3d] WebGL context lost.");
  }

  function onContextRestored() {
    console.info("[bg3d] WebGL context restored.");
    webglReady = initWebGL();
    start();
  }

  /* ---------------------------------------------------------
     Boot
     --------------------------------------------------------- */

  webglReady = initWebGL();

  window.addEventListener("resize", onResize, { passive: true });
  document.addEventListener("pointermove", onPointerMove, { passive: true });
  document.addEventListener("visibilitychange", onVisibility);

  if (canvas) {
    canvas.addEventListener("webglcontextlost", onContextLost, false);
    canvas.addEventListener("webglcontextrestored", onContextRestored, false);
  }

  // Runs even with no WebGL — the spotlight still needs a loop.
  start();
})();
