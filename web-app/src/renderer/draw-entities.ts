import { toCanvas, scale, type CameraParams } from './math';
import { drawWeapon } from './weapons';
import { TEAM_COLORS, TEAM_GLOW, TEAM_DIM, weaponFor, shortId } from '../constants';
import { getEntityAnim } from '../types/effects';
import type { EntityState } from '../types/telemetry';
import type { EffectsState } from '../types/effects';

const TAU = Math.PI * 2;
const RECOIL_DECAY = 0.88;
const MUZZLE_DECAY = 0.82;
const HIT_DECAY = 0.90;

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
  const now = performance.now();
  effects.hoverEntity = null;

  for (const e of entities) {
    const anim = getEntityAnim(effects, e.id);

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

    const recoilOffset = anim.recoil * Math.max(2, 3.5 * s / 0.7);
    drawX -= anim.shotDirX * recoilOffset / (arenaW / canvas.width * cam.camZoom);
    drawY -= anim.shotDirY * recoilOffset / (arenaH / canvas.height * cam.camZoom);

    const [cx, cy] = toCanvas(drawX, drawY, canvas, arenaW, arenaH, cam);
    if (cx < -80 || cx > w + 80 || cy < -80 || cy > h + 80) continue;

    const col = TEAM_COLORS[e.team] || '#fff';
    const dim = TEAM_DIM[e.team] || '#333';
    const glowCol = TEAM_GLOW[e.team] || '#fff';
    const rad = Math.max(4, 7 * s / 0.7);

    const mdx = effects.mouseCanvasX - cx, mdy = effects.mouseCanvasY - cy;
    if (mdx * mdx + mdy * mdy < (rad + 5) * (rad + 5) && !e.is_dead) {
      effects.hoverEntity = e;
    }

    // ── Dead state ──
    if (e.is_dead) {
      if (anim.deathTime === 0) anim.deathTime = now;
      const elapsed = (now - anim.deathTime) / 1000;
      const fadeAlpha = Math.max(0.06, 0.3 - elapsed * 0.08);

      ctx.globalAlpha = fadeAlpha;
      ctx.strokeStyle = col;
      ctx.lineWidth = Math.max(0.5, 1 * s / 0.7);

      const ds = Math.max(3, 4 * s / 0.7);
      ctx.beginPath();
      ctx.moveTo(cx - ds, cy - ds); ctx.lineTo(cx + ds, cy + ds);
      ctx.moveTo(cx + ds, cy - ds); ctx.lineTo(cx - ds, cy + ds);
      ctx.stroke();

      const ringRad = ds + Math.min(elapsed * 15, 12) * s / 0.7;
      const ringAlpha = Math.max(0, 0.2 - elapsed * 0.06);
      if (ringAlpha > 0) {
        ctx.globalAlpha = ringAlpha;
        ctx.beginPath();
        ctx.arc(cx, cy, ringRad, 0, TAU);
        ctx.stroke();
      }

      ctx.globalAlpha = 1;

      anim.recoil *= RECOIL_DECAY;
      anim.muzzleFlash *= MUZZLE_DECAY;
      anim.hitFlash *= HIT_DECAY;
      continue;
    }
    anim.deathTime = 0;

    // ── Drop shadow ──
    ctx.fillStyle = '#00000025';
    ctx.beginPath();
    ctx.ellipse(cx + 1, cy + 2, rad * 0.85, rad * 0.45, 0, 0, TAU);
    ctx.fill();

    // ── Selection ring ──
    if (e.id === followId) {
      ctx.save();
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(cx, cy, rad + 6, 0, TAU);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    // ── Velocity trail ──
    if (opts.trails && e.velocity) {
      const spd = Math.sqrt(e.velocity[0] ** 2 + e.velocity[1] ** 2);
      if (spd > 10) {
        const tLen = Math.min(spd * 0.3, 60);
        const vn = [-e.velocity[0] / spd, -e.velocity[1] / spd];
        const [tx, ty] = toCanvas(
          e.position[0] + vn[0] * tLen,
          e.position[1] + vn[1] * tLen,
          canvas, arenaW, arenaH, cam,
        );
        const g = ctx.createLinearGradient(cx, cy, tx, ty);
        g.addColorStop(0, col + '40');
        g.addColorStop(1, col + '00');
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tx, ty);
        ctx.strokeStyle = g;
        ctx.lineWidth = Math.max(2, 3 * s / 0.7);
        ctx.stroke();
      }
    }

    // ── FOV cone ──
    const fovA = 0.55, fovR = Math.max(25, 50 * s / 0.7);
    const fg = ctx.createRadialGradient(cx, cy, 0, cx, cy, fovR);
    fg.addColorStop(0, dim + '18');
    fg.addColorStop(1, dim + '00');
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, fovR, e.facing - fovA, e.facing + fovA);
    ctx.closePath();
    ctx.fillStyle = fg;
    ctx.fill();

    // ── Glow (optional) ──
    if (opts.glow) {
      ctx.save();
      ctx.shadowColor = glowCol;
      ctx.shadowBlur = Math.max(6, 14 * s / 0.7);
      ctx.beginPath(); ctx.arc(cx, cy, rad, 0, TAU);
      ctx.fillStyle = col;
      ctx.fill();
      ctx.restore();
    }

    // ── Body: circle with directional wedge ──
    ctx.beginPath();
    ctx.arc(cx, cy, rad, 0, TAU);
    ctx.fillStyle = col;
    ctx.fill();

    // Subtle inner shading
    const ig = ctx.createRadialGradient(cx - rad * 0.25, cy - rad * 0.25, 0, cx, cy, rad);
    ig.addColorStop(0, '#ffffff12');
    ig.addColorStop(0.7, 'transparent');
    ig.addColorStop(1, '#00000018');
    ctx.fillStyle = ig;
    ctx.fill();

    // Rim
    ctx.strokeStyle = '#ffffff15';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // ── Directional chevron ──
    const chevLen = rad * 1.45;
    const chevW = rad * 0.45;
    const tipX = cx + Math.cos(e.facing) * chevLen;
    const tipY = cy + Math.sin(e.facing) * chevLen;
    const lX = cx + Math.cos(e.facing + 0.5) * chevW;
    const lY = cy + Math.sin(e.facing + 0.5) * chevW;
    const rX = cx + Math.cos(e.facing - 0.5) * chevW;
    const rY = cy + Math.sin(e.facing - 0.5) * chevW;

    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(lX, lY);
    ctx.lineTo(rX, rY);
    ctx.closePath();
    ctx.fillStyle = col;
    ctx.fill();
    ctx.strokeStyle = '#ffffff20';
    ctx.lineWidth = 0.5;
    ctx.stroke();

    // ── Hit flash overlay ──
    if (anim.hitFlash > 0.05) {
      ctx.globalAlpha = anim.hitFlash * 0.6;
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(cx, cy, rad + 1, 0, TAU);
      ctx.fill();

      // Hit ring expansion
      const hitRingRad = rad + (1 - anim.hitFlash) * rad * 1.5;
      ctx.globalAlpha = anim.hitFlash * 0.3;
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = Math.max(1, 1.5 * s / 0.7);
      ctx.beginPath();
      ctx.arc(cx, cy, hitRingRad, 0, TAU);
      ctx.stroke();

      ctx.globalAlpha = 1;
    }

    // ── Center pip ──
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(1, 1.5 * s / 0.7), 0, TAU);
    ctx.fillStyle = '#ffffff50';
    ctx.fill();

    // ── Weapon ──
    const wKey = weaponFor(e.id);
    const wScale = Math.max(0.35, 0.65 * s / 0.7);
    drawWeapon(
      ctx,
      cx + Math.cos(e.facing) * rad * 0.55,
      cy + Math.sin(e.facing) * rad * 0.55,
      e.facing, wKey, wScale,
    );

    // ── Muzzle flash ──
    if (anim.muzzleFlash > 0.1) {
      const mfDist = rad + Math.max(6, 10 * s / 0.7);
      const mfX = cx + Math.cos(e.facing) * mfDist;
      const mfY = cy + Math.sin(e.facing) * mfDist;
      const mfRad = Math.max(3, 6 * s / 0.7) * anim.muzzleFlash;

      ctx.save();
      ctx.globalAlpha = anim.muzzleFlash * 0.9;

      // Outer glow
      const mfGrad = ctx.createRadialGradient(mfX, mfY, 0, mfX, mfY, mfRad * 2.5);
      mfGrad.addColorStop(0, '#fef08a60');
      mfGrad.addColorStop(0.4, '#fbbf2430');
      mfGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = mfGrad;
      ctx.beginPath();
      ctx.arc(mfX, mfY, mfRad * 2.5, 0, TAU);
      ctx.fill();

      // Core flash — 4-pointed star
      ctx.fillStyle = '#fef9c3';
      ctx.beginPath();
      for (let i = 0; i < 4; i++) {
        const a = e.facing + (i * Math.PI / 2);
        const outerR = i % 2 === 0 ? mfRad * 1.8 : mfRad * 0.5;
        const px = mfX + Math.cos(a) * outerR;
        const py = mfY + Math.sin(a) * outerR;
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.fill();

      // Hot center
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(mfX, mfY, mfRad * 0.35, 0, TAU);
      ctx.fill();

      ctx.restore();
    }

    // ── Health arc ──
    const hp = Math.max(0, e.health / e.max_health);
    const hpC = hp > 0.5 ? '#22c55e' : hp > 0.25 ? '#eab308' : '#ef4444';
    const arcRad = rad + Math.max(2.5, 3.5 * s / 0.7);
    const arcWidth = Math.max(1.5, 2.2 * s / 0.7);
    const arcStart = -Math.PI / 2;
    const arcSweep = TAU * 0.82;

    // Track background
    ctx.beginPath();
    ctx.arc(cx, cy, arcRad, arcStart, arcStart + arcSweep);
    ctx.strokeStyle = '#1e1e22';
    ctx.lineWidth = arcWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // HP fill
    if (hp > 0.005) {
      ctx.beginPath();
      ctx.arc(cx, cy, arcRad, arcStart, arcStart + arcSweep * hp);
      ctx.strokeStyle = hpC;
      ctx.lineWidth = arcWidth;
      ctx.lineCap = 'round';

      if (opts.glow && hp < 0.3) {
        ctx.save();
        ctx.shadowColor = hpC;
        ctx.shadowBlur = 4;
        ctx.stroke();
        ctx.restore();
      } else {
        ctx.stroke();
      }
    }

    // Reset lineCap
    ctx.lineCap = 'butt';

    // ── ID label when zoomed ──
    if (cam.camZoom >= 2) {
      const lY = cy - rad - Math.max(7, 10 * s / 0.7);
      ctx.font = `500 ${Math.max(7, 9 * s / 0.7)}px 'JetBrains Mono',monospace`;
      ctx.fillStyle = '#ffffff40';
      ctx.textAlign = 'center';
      ctx.fillText(shortId(e.id), cx, lY);
      ctx.textAlign = 'start';
    }

    // ── Weather reflection ──
    if (opts.weather) {
      const refY = cy + rad + 1;
      const waveOff = Math.sin(now / 1000 * 2.5 + e.id * 1.3) * 0.6;

      ctx.save();
      ctx.beginPath();
      ctx.rect(cx - rad * 1.5, refY, rad * 3, rad * 1.8);
      ctx.clip();

      ctx.translate(cx + waveOff, refY);
      ctx.scale(1, 0.6);

      const fadeGrad = ctx.createLinearGradient(0, 0, 0, rad * 1.5);
      fadeGrad.addColorStop(0, col + '25');
      fadeGrad.addColorStop(0.6, col + '08');
      fadeGrad.addColorStop(1, col + '00');

      ctx.globalAlpha = 0.15;
      ctx.beginPath();
      ctx.arc(0, 0, rad, 0, TAU);
      ctx.fillStyle = fadeGrad;
      ctx.fill();
      ctx.restore();
    }

    // ── Decay animation state ──
    anim.recoil *= RECOIL_DECAY;
    anim.muzzleFlash *= MUZZLE_DECAY;
    anim.hitFlash *= HIT_DECAY;
    if (anim.recoil < 0.01) anim.recoil = 0;
    if (anim.muzzleFlash < 0.01) anim.muzzleFlash = 0;
    if (anim.hitFlash < 0.01) anim.hitFlash = 0;
  }
}
