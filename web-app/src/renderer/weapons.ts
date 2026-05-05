import { WEAPONS, type WeaponKey } from '../constants';

export function drawWeapon(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  facing: number,
  key: WeaponKey,
  wScale: number,
) {
  const parts = WEAPONS[key];
  if (!parts) return;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(facing);
  ctx.scale(wScale, wScale);
  for (const p of parts) {
    ctx.fillStyle = p.c;
    ctx.fillRect(p.x, p.y, p.w, p.h);
  }
  ctx.strokeStyle = '#ffffff0a';
  ctx.lineWidth = 0.3 / wScale;
  for (const p of parts) {
    ctx.strokeRect(p.x, p.y, p.w, p.h);
  }
  ctx.restore();
}
