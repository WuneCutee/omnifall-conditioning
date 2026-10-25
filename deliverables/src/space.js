/* Starfield (twinkle + slow parallax), occasional shooting stars, and a sparkle burst on each slide change.
   Everything is static when the viewer prefers reduced motion. */
(function () {
  const reduce = (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) || /static/.test(location.search);
  if (/static/.test(location.search)) document.documentElement.classList.add('static');
  const sky = document.getElementById('sky');
  const ctx = sky.getContext('2d');
  let W = 0, H = 0, DPR = 1, stars = [], shooting = [], offset = 0, target = 0, t0 = performance.now();

  // deterministic generator so every viewer sees the same sky
  let seed = 20261025;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  const PALETTE = ['255,255,255', '200,220,255', '170,240,230', '255,230,200', '210,190,255'];

  function build() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = sky.clientWidth; H = sky.clientHeight;
    sky.width = Math.round(W * DPR); sky.height = Math.round(H * DPR);
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    seed = 20261025; stars = [];
    const n = Math.round(W * H / 2600);
    for (let i = 0; i < n; i++) {
      const depth = rnd();
      stars.push({ x: rnd() * W, y: rnd() * H, r: 0.25 + Math.pow(rnd(), 3) * 1.7, d: 0.2 + depth * 0.8,
        c: PALETTE[Math.floor(rnd() * PALETTE.length)], ph: rnd() * Math.PI * 2, sp: 0.6 + rnd() * 2.2, glare: rnd() > 0.985 });
    }
  }

  function drawStar(s, tw, ox) {
    let x = (s.x - ox * s.d) % W; if (x < 0) x += W;
    const a = 0.35 + 0.65 * tw;
    ctx.fillStyle = `rgba(${s.c},${a})`;
    ctx.beginPath(); ctx.arc(x, s.y, s.r, 0, Math.PI * 2); ctx.fill();
    if (s.glare) {  // bright star with a four-point glint
      const L = 6 + 10 * tw;
      const g = ctx.createRadialGradient(x, s.y, 0, x, s.y, L * 1.2);
      g.addColorStop(0, `rgba(${s.c},${0.55 * a})`); g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, s.y, L * 1.2, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = `rgba(${s.c},${0.7 * a})`; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(x - L, s.y); ctx.lineTo(x + L, s.y); ctx.moveTo(x, s.y - L); ctx.lineTo(x, s.y + L); ctx.stroke();
    }
  }

  function spawnShooting() {
    const fromLeft = rnd() > 0.5;
    shooting.push({ x: fromLeft ? rnd() * W * 0.5 : W * (0.5 + rnd() * 0.5), y: rnd() * H * 0.45,
      vx: (fromLeft ? 1 : -1) * (7 + rnd() * 5), vy: 2.6 + rnd() * 2, life: 0, max: 55 + rnd() * 30 });
  }

  function frame(now) {
    const t = (now - t0) / 1000;
    offset += (target - offset) * 0.04;
    ctx.clearRect(0, 0, W, H);
    for (const s of stars) drawStar(s, reduce ? 0.8 : 0.5 + 0.5 * Math.sin(t * s.sp + s.ph), offset);
    if (!reduce) {
      if (rnd() < 0.006 && shooting.length < 2) spawnShooting();
      shooting = shooting.filter(m => m.life < m.max);
      for (const m of shooting) {
        m.life++; m.x += m.vx; m.y += m.vy;
        const k = 1 - m.life / m.max, tail = 18;
        const g = ctx.createLinearGradient(m.x, m.y, m.x - m.vx * tail, m.y - m.vy * tail);
        g.addColorStop(0, `rgba(255,255,255,${0.95 * k})`); g.addColorStop(1, 'rgba(255,255,255,0)');
        ctx.strokeStyle = g; ctx.lineWidth = 1.8; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(m.x, m.y); ctx.lineTo(m.x - m.vx * tail, m.y - m.vy * tail); ctx.stroke();
        ctx.fillStyle = `rgba(255,255,255,${k})`; ctx.beginPath(); ctx.arc(m.x, m.y, 1.8, 0, Math.PI * 2); ctx.fill();
      }
      requestAnimationFrame(frame);
    }
  }

  // sparkle burst drawn on the stage (stage coordinates, 1600x900)
  const sp = document.getElementById('sparks');
  const sctx = sp.getContext('2d');
  sp.width = 1600; sp.height = 900;
  let parts = [], running = false;
  function burst() {
    if (reduce) return;
    const cols = ['255,255,255', '94,234,212', '167,139,250', '251,191,74'];
    for (let i = 0; i < 38; i++) {  // radiate from the eyebrow's signal dot
      const a = -Math.PI * 0.9 + Math.random() * Math.PI * 1.3, v = 1.5 + Math.random() * 5;
      parts.push({ x: 101, y: 83, vx: Math.cos(a) * v, vy: Math.sin(a) * v,
        life: 0, max: 30 + Math.random() * 35, r: 0.6 + Math.random() * 1.6, c: cols[Math.floor(Math.random() * cols.length)] });
    }
    if (!running) { running = true; requestAnimationFrame(tick); }
  }
  function tick() {
    sctx.clearRect(0, 0, 1600, 900);
    parts = parts.filter(p => p.life < p.max);
    for (const p of parts) {
      p.life++; p.x += p.vx; p.y += p.vy; p.vx *= 0.96; p.vy = p.vy * 0.96 + 0.04;
      const k = 1 - p.life / p.max, R = p.r * (1 + 2 * k);
      sctx.fillStyle = `rgba(${p.c},${k})`;
      sctx.beginPath(); sctx.moveTo(p.x, p.y - R * 2); sctx.lineTo(p.x + R * 0.5, p.y - R * 0.5); sctx.lineTo(p.x + R * 2, p.y);
      sctx.lineTo(p.x + R * 0.5, p.y + R * 0.5); sctx.lineTo(p.x, p.y + R * 2); sctx.lineTo(p.x - R * 0.5, p.y + R * 0.5);
      sctx.lineTo(p.x - R * 2, p.y); sctx.lineTo(p.x - R * 0.5, p.y - R * 0.5); sctx.closePath(); sctx.fill();
    }
    if (parts.length) requestAnimationFrame(tick); else { running = false; sctx.clearRect(0, 0, 1600, 900); }
  }

  window.addEventListener('resize', () => { build(); if (reduce) frame(performance.now()); });
  build(); requestAnimationFrame(frame);
  window.spaceFx = { onSlide(i) { target = i * 60; burst(); } };
})();
