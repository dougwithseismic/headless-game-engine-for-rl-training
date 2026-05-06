import { toCanvas, scale, type CameraParams } from './math';
import { drawWeapon } from './weapons';
import { TEAM_COLORS, TEAM_GLOW, TEAM_DIM, weaponFor, shortId } from '../constants';
import type { EntityState } from '../types/telemetry';
import type { EffectsState } from '../types/effects';

export function drawEntities(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  entities: EntityState[],
  effects: EffectsState,
  opts: { glow: boolean; trails: boolean; weather: boolean },
  followId: number | null,
) {
  const s = scale(cam, canvas, arenaW);
  effects.hoverEntity = null;

  for (const e of entities) {
    const prev = effects.prevPositions[e.id];
    let drawX = e.position[0], drawY = e.position[1];
    if (prev) {
      drawX = prev.x + (e.position[0] - prev.x) * 0.3;
      drawY = prev.y + (e.position[1] - prev.y) * 0.3;
      prev.x = drawX;
      prev.y = drawY;
    } else {
      effects.prevPositions[e.id] = { x: drawX, y: drawY };
    }

    const [cx, cy] = toCanvas(drawX, drawY, canvas, arenaW, arenaH, cam);
    if (cx < -60 || cx > w + 60 || cy < -60 || cy > h + 60) continue;

    const col = TEAM_COLORS[e.team] || '#fff';
    const dim = TEAM_DIM[e.team] || '#333';
    const glow = TEAM_GLOW[e.team] || '#fff';
    const rad = Math.max(4, 7 * s / 0.7);

    const mdx = effects.mouseCanvasX - cx, mdy = effects.mouseCanvasY - cy;
    if (mdx * mdx + mdy * mdy < (rad + 5) * (rad + 5) && !e.is_dead) {
      effects.hoverEntity = e;
    }

    if (e.is_dead) {
      const ds = Math.max(2, 2.5 * s / 0.7);
      ctx.globalAlpha = 0.12;
      ctx.fillStyle = dim;
      ctx.fillRect(cx - ds, cy - ds, ds * 2, ds * 2);
      ctx.strokeStyle = '#ff444450';
      ctx.lineWidth = Math.max(0.5, s / 0.7 * 0.4);
      ctx.beginPath();
      ctx.moveTo(cx - ds, cy - ds); ctx.lineTo(cx + ds, cy + ds);
      ctx.moveTo(cx + ds, cy - ds); ctx.lineTo(cx - ds, cy + ds);
      ctx.stroke();
      ctx.globalAlpha = 1;
      continue;
    }

    // Drop shadow
    ctx.fillStyle = '#00000030';
    ctx.beginPath();
    ctx.ellipse(cx + 1.5, cy + 2, rad * 0.9, rad * 0.5, 0, 0, Math.PI * 2);
    ctx.fill();

    // Selection ring
    if (e.id === followId) {
      ctx.save();
      ctx.strokeStyle = '#22d3ee';
      ctx.lineWidth = 1.5;
      ctx.shadowColor = '#22d3ee';
      ctx.shadowBlur = 12;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, rad + 5 + Math.sin(performance.now() * 0.003) * 1.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Velocity trail
    if (opts.trails && e.velocity) {
      const spd = Math.sqrt(e.velocity[0] ** 2 + e.velocity[1] ** 2);
      if (spd > 10) {
        const tLen = Math.min(spd * 0.3, 60);
        const vn = [-e.velocity[0] / spd, -e.velocity[1] / spd];
        const [tx, ty] = toCanvas(drawX + vn[0] * tLen, drawY + vn[1] * tLen, canvas, arenaW, arenaH, cam);
        const g = ctx.createLinearGradient(cx, cy, tx, ty);
        g.addColorStop(0, col + '50');
        g.addColorStop(1, col + '00');
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tx, ty);
        ctx.strokeStyle = g;
        ctx.lineWidth = Math.max(2, 3 * s / 0.7);
        ctx.stroke();
      }
    }

    // FOV cone
    const fovA = 0.65, fovR = Math.max(25, 50 * s / 0.7);
    const fg = ctx.createRadialGradient(cx, cy, 0, cx, cy, fovR);
    fg.addColorStop(0, dim + '25');
    fg.addColorStop(1, dim + '00');
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, fovR, e.facing - fovA, e.facing + fovA);
    ctx.closePath();
    ctx.fillStyle = fg;
    ctx.fill();

    // Glow
    if (opts.glow) {
      ctx.save();
      ctx.shadowColor = glow;
      ctx.shadowBlur = Math.max(6, 14 * s / 0.7);
      ctx.beginPath(); ctx.arc(cx, cy, rad, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
      ctx.restore();
    }

    // Hex body
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 3) * i - Math.PI / 6;
      const px = cx + rad * Math.cos(a), py = cy + rad * Math.sin(a);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = col;
    ctx.fill();

    const ig = ctx.createRadialGradient(cx - rad * 0.3, cy - rad * 0.3, 0, cx, cy, rad);
    ig.addColorStop(0, '#ffffff15');
    ig.addColorStop(1, '#00000020');
    ctx.fillStyle = ig;
    ctx.fill();
    ctx.strokeStyle = '#ffffff18';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // Center pip
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(1, 1.5 * s / 0.7), 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff60';
    ctx.fill();

    // Weapon
    const wKey = weaponFor(e.id);
    const wScale = Math.max(0.35, 0.65 * s / 0.7);
    drawWeapon(ctx, cx + Math.cos(e.facing) * rad * 0.55, cy + Math.sin(e.facing) * rad * 0.55, e.facing, wKey, wScale);

    // Health bar
    const bW = Math.max(12, 20 * s / 0.7), bH = Math.max(1.5, 2 * s / 0.7);
    const bX = cx - bW / 2, bY = cy - rad - Math.max(4, 6 * s / 0.7);
    const hp = Math.max(0, e.health / e.max_health);
    const hpC = hp > 0.5 ? '#22c55e' : hp > 0.25 ? '#eab308' : '#ef4444';
    ctx.fillStyle = '#06060c'; ctx.fillRect(bX - 0.5, bY - 0.5, bW + 1, bH + 1);
    ctx.fillStyle = '#161628'; ctx.fillRect(bX, bY, bW, bH);
    if (opts.glow && hp < 0.3) {
      ctx.save(); ctx.shadowColor = hpC; ctx.shadowBlur = 4;
      ctx.fillStyle = hpC; ctx.fillRect(bX, bY, bW * hp, bH);
      ctx.restore();
    } else {
      ctx.fillStyle = hpC; ctx.fillRect(bX, bY, bW * hp, bH);
    }

    // ID label when zoomed
    if (cam.camZoom >= 2) {
      const lY = bY - Math.max(3, 5 * s / 0.7);
      ctx.font = `${Math.max(6, 8 * s / 0.7)}px 'JetBrains Mono',monospace`;
      ctx.fillStyle = '#ffffff40';
      ctx.textAlign = 'center';
      ctx.fillText(shortId(e.id), cx, lY);
      ctx.textAlign = 'start';
    }

    // Wet-ground reflection
    if (opts.weather) {
      const refY = cy + rad + 1;
      const now = performance.now() / 1000;
      const waveOff = Math.sin(now * 2.5 + e.id * 1.3) * 0.6;

      ctx.save();
      ctx.beginPath();
      ctx.rect(cx - rad * 1.5, refY, rad * 3, rad * 1.8);
      ctx.clip();

      ctx.translate(cx + waveOff, refY);
      ctx.scale(1, 0.6);

      const fadeGrad = ctx.createLinearGradient(0, 0, 0, rad * 1.5);
      fadeGrad.addColorStop(0, col + '30');
      fadeGrad.addColorStop(0.6, col + '10');
      fadeGrad.addColorStop(1, col + '00');

      ctx.globalAlpha = 0.18;
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 3) * i - Math.PI / 6;
        const px = rad * Math.cos(a);
        const py = rad * Math.sin(a);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fillStyle = fadeGrad;
      ctx.fill();
      ctx.restore();
    }
  }
}
