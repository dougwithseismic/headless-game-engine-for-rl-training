import { TEAM_COLORS } from '../constants';
import type { EntityState } from '../types/telemetry';

export function drawMinimap(
  mmCtx: CanvasRenderingContext2D,
  mmW: number, mmH: number,
  arenaW: number, arenaH: number,
  entities: EntityState[],
  obstacles: Array<{ x: number; y: number; width: number; height: number }>,
  camX: number, camY: number, camZoom: number,
  followId: number | null,
) {
  mmCtx.fillStyle = '#0a0a0c';
  mmCtx.fillRect(0, 0, mmW, mmH);

  mmCtx.strokeStyle = '#141416';
  mmCtx.lineWidth = 0.3;
  for (let x = 0; x <= arenaW; x += 200) {
    const mx = (x / arenaW) * mmW;
    mmCtx.beginPath(); mmCtx.moveTo(mx, 0); mmCtx.lineTo(mx, mmH); mmCtx.stroke();
  }
  for (let y = 0; y <= arenaH; y += 200) {
    const my = (y / arenaH) * mmH;
    mmCtx.beginPath(); mmCtx.moveTo(0, my); mmCtx.lineTo(mmW, my); mmCtx.stroke();
  }

  mmCtx.strokeStyle = '#27272a';
  mmCtx.lineWidth = 0.5;
  mmCtx.strokeRect(0, 0, mmW, mmH);

  mmCtx.fillStyle = '#1e1e22';
  for (const obs of obstacles) {
    const ox = (obs.x / arenaW) * mmW;
    const oy = (obs.y / arenaH) * mmH;
    mmCtx.fillRect(ox, oy, (obs.width / arenaW) * mmW, (obs.height / arenaH) * mmH);
  }

  for (const e of entities) {
    const mx = (e.position[0] / arenaW) * mmW;
    const my = (e.position[1] / arenaH) * mmH;
    if (e.is_dead) {
      mmCtx.globalAlpha = 0.15;
      mmCtx.fillStyle = '#555';
      mmCtx.fillRect(mx - 0.5, my - 0.5, 1, 1);
      mmCtx.globalAlpha = 1;
      continue;
    }
    if (e.id === followId) {
      mmCtx.fillStyle = '#3b82f6';
      mmCtx.fillRect(mx - 2, my - 2, 4, 4);
      mmCtx.strokeStyle = '#3b82f6';
      mmCtx.lineWidth = 0.5;
      mmCtx.beginPath(); mmCtx.moveTo(mx, my);
      mmCtx.lineTo(mx + Math.cos(e.facing) * 6, my + Math.sin(e.facing) * 6);
      mmCtx.stroke();
    } else {
      mmCtx.fillStyle = TEAM_COLORS[e.team] || '#fff';
      mmCtx.fillRect(mx - 1, my - 1, 2, 2);
    }
  }

  if (camZoom > 1.05) {
    const vw = arenaW / camZoom, vh = arenaH / camZoom;
    const vx = ((camX - vw / 2) / arenaW) * mmW;
    const vy = ((camY - vh / 2) / arenaH) * mmH;
    mmCtx.strokeStyle = '#3b82f660';
    mmCtx.lineWidth = 1;
    mmCtx.strokeRect(vx, vy, (vw / arenaW) * mmW, (vh / arenaH) * mmH);
  }
}
