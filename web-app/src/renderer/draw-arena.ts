import { toCanvas, scale, type CameraParams } from './math';

export function drawBackground(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = '#07070e';
  ctx.fillRect(0, 0, w, h);

  ctx.globalAlpha = 0.01;
  for (let i = 0; i < 30; i++) {
    ctx.fillStyle = Math.random() > 0.5 ? '#ffffff' : '#8b5cf6';
    ctx.fillRect(Math.random() * w, Math.random() * h, 1, 1);
  }
  ctx.globalAlpha = 1;
}

export function drawArenaBounds(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
) {
  const [x0, y0] = toCanvas(0, 0, canvas, arenaW, arenaH, cam);
  const [x1, y1] = toCanvas(arenaW, arenaH, canvas, arenaW, arenaH, cam);
  const bw = x1 - x0, bh = y1 - y0;

  ctx.fillStyle = '#040408';
  if (y0 > 0) ctx.fillRect(0, 0, w, y0);
  if (y1 < h) ctx.fillRect(0, y1, w, h - y1);
  if (x0 > 0) ctx.fillRect(0, y0, x0, bh);
  if (x1 < w) ctx.fillRect(x1, y0, w - x1, bh);

  ctx.strokeStyle = '#8b5cf620';
  ctx.lineWidth = 2;
  ctx.strokeRect(x0, y0, bw, bh);
  ctx.strokeStyle = '#8b5cf610';
  ctx.lineWidth = 6;
  ctx.strokeRect(x0 - 2, y0 - 2, bw + 4, bh + 4);

  const cLen = Math.min(20, bw * 0.05);
  ctx.strokeStyle = '#8b5cf640';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(x0, y0 + cLen); ctx.lineTo(x0, y0); ctx.lineTo(x0 + cLen, y0); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1 - cLen, y0); ctx.lineTo(x1, y0); ctx.lineTo(x1, y0 + cLen); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x0, y1 - cLen); ctx.lineTo(x0, y1); ctx.lineTo(x0 + cLen, y1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1 - cLen, y1); ctx.lineTo(x1, y1); ctx.lineTo(x1, y1 - cLen); ctx.stroke();

  if (cam.camZoom > 1.5) {
    const s = scale(cam, canvas, arenaW);
    const dangerW = 15 * s;
    const gT = ctx.createLinearGradient(x0, y0, x0, y0 + dangerW);
    gT.addColorStop(0, '#ef444410'); gT.addColorStop(1, 'transparent');
    ctx.fillStyle = gT; ctx.fillRect(x0, y0, bw, dangerW);
    const gB = ctx.createLinearGradient(x0, y1, x0, y1 - dangerW);
    gB.addColorStop(0, '#ef444410'); gB.addColorStop(1, 'transparent');
    ctx.fillStyle = gB; ctx.fillRect(x0, y1 - dangerW, bw, dangerW);
    const gL = ctx.createLinearGradient(x0, y0, x0 + dangerW, y0);
    gL.addColorStop(0, '#ef444410'); gL.addColorStop(1, 'transparent');
    ctx.fillStyle = gL; ctx.fillRect(x0, y0, dangerW, bh);
    const gR = ctx.createLinearGradient(x1, y0, x1 - dangerW, y0);
    gR.addColorStop(0, '#ef444410'); gR.addColorStop(1, 'transparent');
    ctx.fillStyle = gR; ctx.fillRect(x1 - dangerW, y0, dangerW, bh);
  }
}

export function drawLineGrid(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
) {
  const step = 100;
  ctx.strokeStyle = '#0c0c18';
  ctx.lineWidth = 0.5;
  for (let x = 0; x <= arenaW; x += step) {
    const [cx] = toCanvas(x, 0, canvas, arenaW, arenaH, cam);
    const [, y0] = toCanvas(0, 0, canvas, arenaW, arenaH, cam);
    const [, y1] = toCanvas(0, arenaH, canvas, arenaW, arenaH, cam);
    if (cx < -1 || cx > w + 1) continue;
    ctx.beginPath(); ctx.moveTo(cx, y0); ctx.lineTo(cx, y1); ctx.stroke();
  }
  for (let y = 0; y <= arenaH; y += step) {
    const [, cy] = toCanvas(0, y, canvas, arenaW, arenaH, cam);
    const [x0] = toCanvas(0, 0, canvas, arenaW, arenaH, cam);
    const [x1] = toCanvas(arenaW, 0, canvas, arenaW, arenaH, cam);
    if (cy < -1 || cy > h + 1) continue;
    ctx.beginPath(); ctx.moveTo(x0, cy); ctx.lineTo(x1, cy); ctx.stroke();
  }
}

export function drawDotGrid(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
) {
  const step = 50;
  ctx.fillStyle = '#181830';
  for (let x = 0; x <= arenaW; x += step) {
    for (let y = 0; y <= arenaH; y += step) {
      const [cx, cy] = toCanvas(x, y, canvas, arenaW, arenaH, cam);
      if (cx < -1 || cx > w + 1 || cy < -1 || cy > h + 1) continue;
      ctx.fillRect(cx, cy, 1, 1);
    }
  }
}

export function drawObstacles(
  ctx: CanvasRenderingContext2D,
  w: number, h: number,
  cam: CameraParams,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  obstacles: Array<{ x: number; y: number; width: number; height: number }>,
) {
  const s = scale(cam, canvas, arenaW);
  for (const obs of obstacles) {
    const [ox, oy] = toCanvas(obs.x, obs.y, canvas, arenaW, arenaH, cam);
    const [ox2, oy2] = toCanvas(obs.x + obs.width, obs.y + obs.height, canvas, arenaW, arenaH, cam);
    const ow = ox2 - ox, oh = oy2 - oy;
    if (ox + ow < -10 || ox > w + 10 || oy + oh < -10 || oy > h + 10) continue;

    ctx.fillStyle = '#06060c'; ctx.fillRect(ox + 3, oy + 3, ow, oh);
    ctx.fillStyle = '#12122a'; ctx.fillRect(ox, oy, ow, oh);

    ctx.save();
    ctx.globalAlpha = 0.04;
    ctx.strokeStyle = '#8b5cf6';
    ctx.lineWidth = 0.5;
    const hStep = Math.max(6, 8 * s / 0.7);
    ctx.beginPath();
    for (let d = -ow; d < ow + oh; d += hStep) {
      ctx.moveTo(ox + Math.max(0, d), oy + Math.max(0, -d));
      ctx.lineTo(ox + Math.min(ow, d + oh), oy + Math.min(oh, oh - d));
    }
    ctx.stroke();
    ctx.restore();

    ctx.strokeStyle = '#222244'; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, ow, oh);
    ctx.strokeStyle = '#2a2a50'; ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + ow, oy); ctx.stroke();
  }
}
