import { toCanvas, scale, type CameraParams } from './math';
import type { EffectsState } from '../types/effects';

export function drawDecals(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  effects: EffectsState,
) {
  const s = scale(cam, canvas, arenaW);
  for (const d of effects.decals) {
    const [cx, cy] = toCanvas(d.x, d.y, canvas, arenaW, arenaH, cam);
    if (cx < -30 || cx > w + 30 || cy < -30 || cy > h + 30) continue;
    const sz = d.size * s / 0.7;
    ctx.globalAlpha = d.alpha;
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, sz);
    g.addColorStop(0, d.color + '40');
    g.addColorStop(1, d.color + '00');
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx, cy, sz, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
    d.alpha *= 0.9995;
  }
}

export function drawAmbient(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  effects: EffectsState,
) {
  const s = scale(cam, canvas, arenaW);
  ctx.globalAlpha = 1;
  for (const p of effects.ambientParticles) {
    p.x += p.vx * 0.016;
    p.y += p.vy * 0.016;
    if (p.x < 0) p.x = arenaW;
    if (p.x > arenaW) p.x = 0;
    if (p.y < 0) p.y = arenaH;
    if (p.y > arenaH) p.y = 0;
    const [cx, cy] = toCanvas(p.x, p.y, canvas, arenaW, arenaH, cam);
    if (cx < -5 || cx > w + 5 || cy < -5 || cy > h + 5) continue;
    ctx.globalAlpha = p.alpha;
    ctx.fillStyle = '#52525b';
    const sz = p.size * s / 0.7;
    ctx.fillRect(cx, cy, sz, sz);
  }
  ctx.globalAlpha = 1;
}

export function drawRipples(
  ctx: CanvasRenderingContext2D,
  _w: number, _h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  effects: EffectsState,
) {
  const s = scale(cam, canvas, arenaW);
  effects.ripples = effects.ripples.filter(r => {
    r.radius += 2.5;
    r.alpha -= 0.015;
    if (r.alpha <= 0) return false;
    const [cx, cy] = toCanvas(r.x, r.y, canvas, arenaW, arenaH, cam);
    const rad = r.radius * s / 0.7;
    ctx.strokeStyle = `rgba(82,82,91,${r.alpha * 0.3})`;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2); ctx.stroke();
    return true;
  });
}

export function drawShots(
  ctx: CanvasRenderingContext2D,
  _w: number, _h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  effects: EffectsState,
  glow: boolean,
) {
  const s = scale(cam, canvas, arenaW);
  effects.shotTraces = effects.shotTraces.filter(t => {
    t.alpha -= 0.035;
    if (t.alpha <= 0) return false;
    const [ox, oy] = toCanvas(t.ox, t.oy, canvas, arenaW, arenaH, cam);
    const [ex, ey] = toCanvas(t.ex, t.ey, canvas, arenaW, arenaH, cam);

    if (glow && t.hit) {
      ctx.save();
      ctx.shadowColor = '#fbbf24';
      ctx.shadowBlur = 10;
      ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ex, ey);
      ctx.strokeStyle = `rgba(251,191,36,${t.alpha * 0.5})`;
      ctx.lineWidth = 2.5;
      ctx.stroke();
      ctx.restore();
    }
    ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ex, ey);
    if (t.hit) {
      ctx.strokeStyle = `rgba(255,220,80,${t.alpha})`;
      ctx.lineWidth = 1.5;
    } else {
      ctx.strokeStyle = `rgba(82,82,91,${t.alpha * 0.4})`;
      ctx.lineWidth = 0.5;
    }
    ctx.stroke();

    if (t.alpha > 0.7) {
      ctx.save();
      ctx.shadowColor = '#fbbf24';
      ctx.shadowBlur = 15 * t.alpha;
      ctx.fillStyle = `rgba(255,240,150,${(t.alpha - 0.7) * 3})`;
      ctx.beginPath(); ctx.arc(ox, oy, 3 * s / 0.7, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }
    return true;
  });
}

export function drawParticles(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  effects: EffectsState,
  glow: boolean,
) {
  const s = scale(cam, canvas, arenaW);
  effects.particles = effects.particles.filter(p => {
    p.life -= p.decay;
    if (p.life <= 0) return false;
    p.x += p.vx * 0.016;
    p.y += p.vy * 0.016;
    p.vx *= 0.95;
    p.vy *= 0.95;
    const [cx, cy] = toCanvas(p.x, p.y, canvas, arenaW, arenaH, cam);
    if (cx < -20 || cx > w + 20 || cy < -20 || cy > h + 20) return true;
    const sz = p.size * p.life * Math.max(1, s / 0.7);
    if (glow && p.type === 'kill') {
      ctx.save();
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.life * 0.8;
      ctx.fillRect(cx - sz / 2, cy - sz / 2, sz, sz);
      ctx.restore();
      ctx.globalAlpha = 1;
    } else {
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.life * 0.5;
      ctx.fillRect(cx - sz / 2, cy - sz / 2, sz, sz);
      ctx.globalAlpha = 1;
    }
    return true;
  });
}

export function drawDmgNumbers(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  effects: EffectsState,
) {
  const s = scale(cam, canvas, arenaW);
  effects.dmgNumbers = effects.dmgNumbers.filter(d => {
    d.life -= 0.015;
    if (d.life <= 0) return false;
    d.y += d.vy * 0.016;
    d.x += d.vx * 0.016;
    d.vy *= 0.97;
    const [cx, cy] = toCanvas(d.x, d.y, canvas, arenaW, arenaH, cam);
    if (cx < -30 || cx > w + 30 || cy < -30 || cy > h + 30) return true;
    const sz = Math.max(8, 11 * s / 0.7);
    ctx.font = `700 ${sz}px 'JetBrains Mono',monospace`;
    ctx.globalAlpha = d.life;
    ctx.fillStyle = d.color;
    ctx.textAlign = 'center';
    ctx.fillText(d.text, cx, cy);
    ctx.textAlign = 'start';
    ctx.globalAlpha = 1;
    return true;
  });
}

export function initAmbient(effects: EffectsState, arenaW: number, arenaH: number) {
  effects.ambientParticles = [];
  for (let i = 0; i < 40; i++) {
    effects.ambientParticles.push({
      x: Math.random() * arenaW,
      y: Math.random() * arenaH,
      vx: (Math.random() - 0.5) * 8,
      vy: (Math.random() - 0.5) * 8,
      size: 0.5 + Math.random() * 1.5,
      alpha: 0.08 + Math.random() * 0.1,
    });
  }
}

export function spawnParticles(
  effects: EffectsState,
  x: number, y: number,
  color: string, count: number,
  type: 'kill' | 'damage' | 'spawn',
) {
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    const spd = 20 + Math.random() * 70;
    effects.particles.push({
      x, y,
      vx: Math.cos(a) * spd,
      vy: Math.sin(a) * spd,
      life: 1,
      decay: 0.02 + Math.random() * 0.04,
      size: type === 'kill' ? 1.5 + Math.random() * 2.5 : 0.8 + Math.random() * 1.5,
      color,
      type,
    });
  }
}

export function spawnDmgNumber(
  effects: EffectsState,
  x: number, y: number,
  amount: number, color: string,
) {
  effects.dmgNumbers.push({
    x, y,
    text: '-' + Math.round(amount),
    color,
    life: 1,
    vy: -40 - Math.random() * 20,
    vx: (Math.random() - 0.5) * 30,
  });
}

export function spawnDecal(effects: EffectsState, x: number, y: number, color: string) {
  effects.decals.push({ x, y, color, alpha: 0.4, size: 8 + Math.random() * 12 });
  if (effects.decals.length > 60) effects.decals.shift();
}

export function spawnRipple(effects: EffectsState, x: number, y: number) {
  effects.ripples.push({ x, y, radius: 0, maxRadius: 40 + Math.random() * 30, alpha: 0.5 });
}
