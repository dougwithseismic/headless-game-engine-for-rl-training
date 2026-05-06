import { toCanvas, scale, type CameraParams } from './math';
import type { EntityState } from '../types/telemetry';

export function drawFog(
  fogCtx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  entities: EntityState[],
) {
  const s = scale(cam, canvas, arenaW);
  fogCtx.clearRect(0, 0, w, h);
  fogCtx.fillStyle = 'rgba(9,9,11,0.65)';
  fogCtx.fillRect(0, 0, w, h);
  fogCtx.globalCompositeOperation = 'destination-out';

  for (const e of entities) {
    if (e.is_dead) continue;
    const [cx, cy] = toCanvas(e.position[0], e.position[1], canvas, arenaW, arenaH, cam);
    const vr = Math.max(35, 75 * s / 0.7);
    const g = fogCtx.createRadialGradient(cx, cy, 0, cx, cy, vr);
    g.addColorStop(0, 'rgba(0,0,0,1)');
    g.addColorStop(0.5, 'rgba(0,0,0,0.8)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    fogCtx.fillStyle = g;
    fogCtx.beginPath(); fogCtx.arc(cx, cy, vr, 0, Math.PI * 2); fogCtx.fill();

    const fr = Math.max(50, 110 * s / 0.7);
    const fg = fogCtx.createRadialGradient(cx, cy, 8, cx, cy, fr);
    fg.addColorStop(0, 'rgba(0,0,0,0.9)');
    fg.addColorStop(0.6, 'rgba(0,0,0,0.5)');
    fg.addColorStop(1, 'rgba(0,0,0,0)');
    fogCtx.fillStyle = fg;
    fogCtx.beginPath(); fogCtx.moveTo(cx, cy);
    fogCtx.arc(cx, cy, fr, e.facing - 0.6, e.facing + 0.6);
    fogCtx.closePath(); fogCtx.fill();
  }

  fogCtx.globalCompositeOperation = 'source-over';
}
