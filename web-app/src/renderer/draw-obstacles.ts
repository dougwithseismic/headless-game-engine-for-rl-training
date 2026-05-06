import { toCanvas, scale, type CameraParams } from './math';

function mulberry32(a: number): () => number {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type ObstacleKind = 'tower' | 'wall' | 'bunker' | 'crate';

function classify(w: number, h: number): ObstacleKind {
  const area = w * h;
  if (area < 5000) return 'crate';
  const ratio = h / w;
  if (ratio > 1.8) return 'tower';
  if (ratio < 0.55) return 'wall';
  return 'bunker';
}

function obstacleSeed(obs: { x: number; y: number; width: number; height: number }): number {
  return ((obs.x * 7919 + obs.y * 6271 + obs.width * 3571 + obs.height * 2999) >>> 0);
}

export function drawObstacles(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  obstacles: Array<{ x: number; y: number; width: number; height: number }>,
  weather: boolean,
) {
  const s = scale(cam, canvas, arenaW);
  const now = performance.now() / 1000;

  for (const obs of obstacles) {
    const [ox, oy] = toCanvas(obs.x, obs.y, canvas, arenaW, arenaH, cam);
    const [ox2, oy2] = toCanvas(obs.x + obs.width, obs.y + obs.height, canvas, arenaW, arenaH, cam);
    const ow = ox2 - ox, oh = oy2 - oy;
    if (ox + ow < -10 || ox > w + 10 || oy + oh < -10 || oy > h + 10) continue;

    const seed = obstacleSeed(obs);
    const kind = classify(obs.width, obs.height);

    ctx.fillStyle = '#060608';
    ctx.fillRect(ox + 3, oy + 3, ow, oh);

    const baseGrad = ctx.createLinearGradient(ox, oy, ox, oy + oh);
    baseGrad.addColorStop(0, '#1c1c1f');
    baseGrad.addColorStop(0.45, '#161618');
    baseGrad.addColorStop(1, '#0f0f11');
    ctx.fillStyle = baseGrad;
    ctx.fillRect(ox, oy, ow, oh);

    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, oy, ow, oh);
    ctx.clip();

    drawStructure(ctx, ox, oy, ow, oh, s, kind, mulberry32(seed + 1), cam.camZoom, now);

    if (cam.camZoom > 1.2) {
      drawWindows(ctx, ox, oy, ow, oh, s, kind, mulberry32(seed + 2), now);
    }

    if (cam.camZoom > 1.5 && (kind === 'wall' || kind === 'bunker')) {
      drawDoor(ctx, ox, oy, ow, oh, s, mulberry32(seed + 3));
    }

    if (cam.camZoom > 1.0) {
      drawSurfaceDetail(ctx, ox, oy, ow, oh, s, kind, mulberry32(seed + 4));
    }

    if (weather) {
      drawWeathering(ctx, ox, oy, ow, oh, s, mulberry32(seed + 5));
    }

    ctx.restore();

    drawEdgeLighting(ctx, ox, oy, ow, oh, s, mulberry32(seed + 6), now);

    if (weather) {
      drawDrips(ctx, ox, oy, ow, oh, s, seed, now);
      drawPuddle(ctx, ox, oy, ow, oh, s, seed, now);
    }

    ctx.strokeStyle = '#27272a';
    ctx.lineWidth = 1.2;
    ctx.strokeRect(ox, oy, ow, oh);
    ctx.strokeStyle = '#3f3f46';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + ow, oy); ctx.stroke();
    ctx.strokeStyle = '#0c0c0e';
    ctx.beginPath(); ctx.moveTo(ox, oy + oh); ctx.lineTo(ox + ow, oy + oh); ctx.stroke();
  }
}

function drawStructure(
  ctx: CanvasRenderingContext2D,
  ox: number, oy: number, ow: number, oh: number,
  s: number, kind: ObstacleKind, rand: () => number, zoom: number, now: number,
) {
  ctx.strokeStyle = '#2a2a2e';
  ctx.lineWidth = 0.8;

  switch (kind) {
    case 'tower': {
      const cx = ox + ow / 2;
      ctx.beginPath(); ctx.moveTo(cx, oy); ctx.lineTo(cx, oy + oh); ctx.stroke();
      const floors = Math.max(2, Math.floor(oh / Math.max(12, 20 * s / 0.7)));
      const floorH = oh / floors;
      for (let f = 1; f < floors; f++) {
        const fy = oy + f * floorH;
        ctx.beginPath(); ctx.moveTo(ox, fy); ctx.lineTo(ox + ow, fy); ctx.stroke();
      }
      if (zoom > 2) {
        const ax = ox + ow * (0.3 + rand() * 0.4);
        const antH = Math.min(8, 12 * s / 0.7);
        ctx.strokeStyle = '#3f3f46';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ax, oy); ctx.lineTo(ax, oy - antH); ctx.stroke();
        ctx.fillStyle = '#ef4444';
        ctx.globalAlpha = 0.4 + 0.4 * Math.sin(now * 3.5);
        ctx.fillRect(ax - 1, oy - antH - 1.5, 2.5, 2.5);
        ctx.globalAlpha = 1;
      }
      break;
    }
    case 'wall': {
      const sections = Math.max(2, Math.floor(ow / Math.max(18, 30 * s / 0.7)));
      const secW = ow / sections;
      for (let i = 1; i < sections; i++) {
        const sx = ox + i * secW;
        ctx.beginPath(); ctx.moveTo(sx, oy); ctx.lineTo(sx, oy + oh); ctx.stroke();
      }
      ctx.beginPath(); ctx.moveTo(ox, oy + oh / 2); ctx.lineTo(ox + ow, oy + oh / 2); ctx.stroke();
      break;
    }
    case 'bunker': {
      ctx.beginPath();
      ctx.moveTo(ox + ow / 2, oy); ctx.lineTo(ox + ow / 2, oy + oh);
      ctx.moveTo(ox, oy + oh / 2); ctx.lineTo(ox + ow, oy + oh / 2);
      ctx.stroke();
      if (zoom > 2.5) {
        const bLen = Math.min(6, 10 * s / 0.7);
        ctx.strokeStyle = '#2a2a2e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ox + 2, oy + bLen + 2); ctx.lineTo(ox + 2, oy + 2); ctx.lineTo(ox + bLen + 2, oy + 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(ox + ow - bLen - 2, oy + oh - 2); ctx.lineTo(ox + ow - 2, oy + oh - 2); ctx.lineTo(ox + ow - 2, oy + oh - bLen - 2);
        ctx.stroke();
      }
      break;
    }
    case 'crate': {
      ctx.globalAlpha = 0.25;
      ctx.beginPath();
      ctx.moveTo(ox, oy); ctx.lineTo(ox + ow, oy + oh);
      ctx.moveTo(ox + ow, oy); ctx.lineTo(ox, oy + oh);
      ctx.stroke();
      ctx.globalAlpha = 1;
      break;
    }
  }
}

const ACCENTS = ['#60a5fa', '#f59e0b', '#22c55e', '#ef4444'];

function drawWindows(
  ctx: CanvasRenderingContext2D,
  ox: number, oy: number, ow: number, oh: number,
  s: number, kind: ObstacleKind, rand: () => number, now: number,
) {
  const winSize = Math.max(3, 5 * s / 0.7);
  const gap = Math.max(4, 7 * s / 0.7);
  const margin = Math.max(4, 6 * s / 0.7);

  const cols = Math.max(1, Math.floor((ow - margin * 2) / (winSize + gap)));
  const rows = Math.max(1, Math.floor((oh - margin * 2) / (winSize + gap)));
  const maxWin = kind === 'tower' ? 8 : kind === 'wall' ? 6 : 4;
  const total = Math.min(cols * rows, maxWin);

  const lit: boolean[] = [];
  for (let i = 0; i < total; i++) lit.push(rand() > 0.35);
  const accent = ACCENTS[Math.floor(rand() * ACCENTS.length)];

  const startX = ox + (ow - (cols * (winSize + gap) - gap)) / 2;
  const startY = oy + (oh - (rows * (winSize + gap) - gap)) / 2;

  let idx = 0;
  for (let r = 0; r < rows && idx < total; r++) {
    for (let c = 0; c < cols && idx < total; c++) {
      const wx = startX + c * (winSize + gap);
      const wy = startY + r * (winSize + gap);

      if (lit[idx]) {
        const flicker = 0.55 + 0.45 * Math.sin(now * (1.3 + rand() * 2) + idx * 1.7);
        ctx.globalAlpha = 0.2 * flicker;
        ctx.fillStyle = accent;
        ctx.fillRect(wx - 2, wy - 2, winSize + 4, winSize + 4);
        ctx.globalAlpha = 0.5 * flicker;
        ctx.fillRect(wx, wy, winSize, winSize);
      } else {
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = '#08080a';
        ctx.fillRect(wx, wy, winSize, winSize);
      }

      ctx.globalAlpha = 0.25;
      ctx.strokeStyle = '#2a2a2e';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(wx, wy, winSize, winSize);
      ctx.globalAlpha = 1;
      idx++;
    }
  }
}

function drawDoor(
  ctx: CanvasRenderingContext2D,
  ox: number, oy: number, ow: number, oh: number,
  s: number, rand: () => number,
) {
  const doorW = Math.max(6, 10 * s / 0.7);
  const doorH = Math.min(oh * 0.55, Math.max(10, 16 * s / 0.7));
  const doorX = ox + (ow - doorW) * (0.3 + rand() * 0.4);
  const doorY = oy + oh - doorH;

  ctx.fillStyle = '#08080a';
  ctx.globalAlpha = 0.6;
  ctx.fillRect(doorX, doorY, doorW, doorH);

  ctx.strokeStyle = '#2a2a2e';
  ctx.lineWidth = 0.5;
  ctx.globalAlpha = 0.4;
  ctx.strokeRect(doorX, doorY, doorW, doorH);

  ctx.fillStyle = '#f59e0b';
  ctx.globalAlpha = 0.18;
  ctx.fillRect(doorX + doorW - 3, doorY + doorH * 0.45, 2, 3);

  ctx.globalAlpha = 0.06;
  const tg = ctx.createLinearGradient(doorX, doorY + doorH, doorX, doorY + doorH + 5);
  tg.addColorStop(0, '#f59e0b');
  tg.addColorStop(1, 'transparent');
  ctx.fillStyle = tg;
  ctx.fillRect(doorX, doorY + doorH, doorW, 5);
  ctx.globalAlpha = 1;
}

function drawSurfaceDetail(
  ctx: CanvasRenderingContext2D,
  ox: number, oy: number, ow: number, oh: number,
  s: number, kind: ObstacleKind, rand: () => number,
) {
  if (rand() > 0.35) {
    const ventW = Math.min(ow * 0.3, Math.max(8, 15 * s / 0.7));
    const ventH = Math.max(1, 2 * s / 0.7);
    const ventX = ox + ow * (0.1 + rand() * 0.3);
    const ventY = oy + oh * 0.12;
    const ventCount = 1 + Math.floor(rand() * 2);

    ctx.fillStyle = '#0a0a0c';
    ctx.globalAlpha = 0.5;
    for (let v = 0; v < ventCount; v++) {
      ctx.fillRect(ventX, ventY + v * (ventH + 2), ventW, ventH);
    }
    ctx.globalAlpha = 1;
  }

  if (rand() > 0.6) {
    const stripeH = Math.max(2, 3 * s / 0.7);
    const isTop = rand() > 0.5;
    const stripeY = isTop ? oy + 1 : oy + oh - stripeH - 1;
    const step = Math.max(4, 6 * s / 0.7);

    ctx.globalAlpha = 0.12;
    ctx.fillStyle = '#fbbf24';
    for (let sx = ox; sx < ox + ow; sx += step * 2) {
      ctx.fillRect(sx, stripeY, step, stripeH);
    }
    ctx.globalAlpha = 1;
  }

  if (kind === 'wall' && rand() > 0.3) {
    const pipeY = oy + oh * 0.72;
    ctx.strokeStyle = '#1e1e22';
    ctx.lineWidth = Math.max(1, 1.5 * s / 0.7);
    ctx.globalAlpha = 0.4;
    ctx.beginPath(); ctx.moveTo(ox + 2, pipeY); ctx.lineTo(ox + ow - 2, pipeY); ctx.stroke();

    const joints = 2 + Math.floor(rand() * 3);
    ctx.fillStyle = '#2a2a2e';
    for (let j = 0; j < joints; j++) {
      const jx = ox + ow * (0.15 + (j / joints) * 0.7);
      ctx.fillRect(jx - 1, pipeY - 1.5, 3, 3);
    }
    ctx.globalAlpha = 1;
  }

  const hStep = Math.max(8, 10 * s / 0.7);
  ctx.globalAlpha = 0.02;
  ctx.strokeStyle = '#52525b';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  for (let d = -ow; d < ow + oh; d += hStep) {
    ctx.moveTo(ox + Math.max(0, d), oy + Math.max(0, -d));
    ctx.lineTo(ox + Math.min(ow, d + oh), oy + Math.min(oh, oh - d));
  }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawWeathering(
  ctx: CanvasRenderingContext2D,
  ox: number, oy: number, ow: number, oh: number,
  s: number, rand: () => number,
) {
  const streakCount = Math.max(3, Math.floor(ow / Math.max(8, 12 * s / 0.7)));
  ctx.strokeStyle = '#52525b';
  ctx.lineWidth = 0.5;

  for (let i = 0; i < streakCount; i++) {
    const sx = ox + ow * (0.05 + rand() * 0.9);
    const sy = oy + oh * rand() * 0.25;
    const sLen = oh * (0.25 + rand() * 0.55);
    const wind = 2 * s / 0.7;

    ctx.globalAlpha = 0.05 + rand() * 0.06;
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + wind, sy + sLen); ctx.stroke();
  }

  const stainCount = 1 + Math.floor(rand() * 2);
  for (let i = 0; i < stainCount; i++) {
    const stX = ox + ow * (0.15 + rand() * 0.7);
    const stY = oy + oh * (0.2 + rand() * 0.5);
    const stR = Math.max(3, (5 + rand() * 10) * s / 0.7);

    ctx.globalAlpha = 0.035;
    const sg = ctx.createRadialGradient(stX, stY, 0, stX, stY, stR);
    sg.addColorStop(0, '#000000');
    sg.addColorStop(1, 'transparent');
    ctx.fillStyle = sg;
    ctx.beginPath(); ctx.arc(stX, stY, stR, 0, Math.PI * 2); ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function drawEdgeLighting(
  ctx: CanvasRenderingContext2D,
  ox: number, oy: number, ow: number, oh: number,
  s: number, rand: () => number, now: number,
) {
  const glowW = Math.max(4, 10 * s / 0.7);
  const pulse = 0.65 + 0.35 * Math.sin(now * 1.2 + rand() * 6.28);
  const edges = [rand() > 0.35, rand() > 0.45, rand() > 0.55, rand() > 0.45];
  const alpha = 0.06 * pulse;

  ctx.globalAlpha = alpha;

  if (edges[0]) {
    const g = ctx.createLinearGradient(ox, oy, ox, oy - glowW);
    g.addColorStop(0, '#3f3f46'); g.addColorStop(1, 'transparent');
    ctx.fillStyle = g; ctx.fillRect(ox, oy - glowW, ow, glowW);
  }
  if (edges[1]) {
    const g = ctx.createLinearGradient(ox + ow, oy, ox + ow + glowW, oy);
    g.addColorStop(0, '#3f3f46'); g.addColorStop(1, 'transparent');
    ctx.fillStyle = g; ctx.fillRect(ox + ow, oy, glowW, oh);
  }
  if (edges[2]) {
    const g = ctx.createLinearGradient(ox, oy + oh, ox, oy + oh + glowW);
    g.addColorStop(0, '#3f3f46'); g.addColorStop(1, 'transparent');
    ctx.fillStyle = g; ctx.fillRect(ox, oy + oh, ow, glowW);
  }
  if (edges[3]) {
    const g = ctx.createLinearGradient(ox, oy, ox - glowW, oy);
    g.addColorStop(0, '#3f3f46'); g.addColorStop(1, 'transparent');
    ctx.fillStyle = g; ctx.fillRect(ox - glowW, oy, glowW, oh);
  }
  ctx.globalAlpha = 1;
}

function drawDrips(
  ctx: CanvasRenderingContext2D,
  ox: number, _oy: number, ow: number, oh: number,
  s: number, seed: number, now: number,
) {
  const dripCount = Math.max(2, Math.floor(ow / Math.max(10, 18 * s / 0.7)));
  const fallDist = Math.max(8, 22 * s / 0.7);
  const baseY = _oy + oh;

  ctx.strokeStyle = '#52525b';
  ctx.lineWidth = 1;

  for (let i = 0; i < dripCount; i++) {
    const ph = ((seed * (i + 1) * 127) & 0xffff) / 0xffff;
    const sp = 0.35 + (((seed * (i + 7) * 251) & 0xffff) / 0xffff) * 0.45;
    const xf = 0.08 + (((seed * (i + 3) * 199) & 0xffff) / 0xffff) * 0.84;

    const progress = (now * sp + ph) % 1;
    const dx = ox + xf * ow;
    const dy = baseY + progress * fallDist;
    const dripLen = Math.max(2, 4 * s / 0.7);

    ctx.globalAlpha = 0.25 * (1 - progress);
    ctx.beginPath(); ctx.moveTo(dx, dy - dripLen); ctx.lineTo(dx, dy); ctx.stroke();

    if (progress > 0.85) {
      const splash = (progress - 0.85) / 0.15;
      ctx.globalAlpha = 0.15 * (1 - splash);
      ctx.beginPath();
      ctx.arc(dx, baseY + fallDist, splash * 3 * s / 0.7, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

function drawPuddle(
  ctx: CanvasRenderingContext2D,
  ox: number, oy: number, ow: number, oh: number,
  s: number, seed: number, now: number,
) {
  const puddleH = Math.max(3, 8 * s / 0.7);
  const shimmer = 0.55 + 0.45 * Math.sin(now * 1.8 + seed * 0.001);
  const baseY = oy + oh;

  ctx.globalAlpha = 0.08 * shimmer;
  const g = ctx.createLinearGradient(ox, baseY, ox, baseY + puddleH);
  g.addColorStop(0, '#52525b'); g.addColorStop(1, 'transparent');
  ctx.fillStyle = g;
  ctx.fillRect(ox - 2, baseY, ow + 4, puddleH);

  ctx.globalAlpha = 0.04 * shimmer;
  ctx.fillStyle = '#71717a';
  ctx.fillRect(ox + ow * 0.2, baseY, ow * 0.6, puddleH * 0.5);

  ctx.globalAlpha = 1;
}

const RAIN_LAYERS = [
  { count: 40, speed: 120, len: 4,  lineW: 0.4, alpha: 0.025, wind: 0.8,  color: '#3f3f46' },
  { count: 30, speed: 220, len: 8,  lineW: 0.7, alpha: 0.04,  wind: 1.5,  color: '#52525b' },
  { count: 15, speed: 380, len: 14, lineW: 1.2, alpha: 0.06,  wind: 2.5,  color: '#71717a' },
];

export function drawRain(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
) {
  if (cam.camZoom < 1.2) return;

  const now = performance.now() / 1000;
  const intensity = Math.min(1, (cam.camZoom - 1) / 2);
  const windShift = Math.sin(now * 0.3) * 0.5;

  for (const layer of RAIN_LAYERS) {
    const count = Math.floor(layer.count * intensity);
    ctx.strokeStyle = layer.color;
    ctx.lineWidth = layer.lineW;

    for (let i = 0; i < count; i++) {
      const hash = ((i * 7919 + layer.speed * 31 + 3571) >>> 0);
      const xFrac = (hash & 0xffff) / 0xffff;
      const phase = ((hash * 251) & 0xffff) / 0xffff;
      const lenVar = layer.len * (0.7 + (((hash * 43) & 0xff) / 255) * 0.6);

      const rx = xFrac * w;
      const ry = ((now * layer.speed + phase * h) % (h + lenVar)) - lenVar;
      const wind = (layer.wind + windShift) * cam.camZoom;

      ctx.globalAlpha = layer.alpha + (((hash * 67) & 0xff) / 255) * layer.alpha * 0.5;
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(rx + wind, ry + lenVar);
      ctx.stroke();

      if (layer.lineW > 1 && ry > h * 0.75 && ry < h * 0.78) {
        ctx.globalAlpha = 0.04;
        const splashR = 2 + ((hash * 17) & 3);
        ctx.beginPath();
        ctx.arc(rx + wind, ry + lenVar, splashR, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;
}
