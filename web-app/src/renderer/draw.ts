import { drawBackground, drawArenaBounds, drawLineGrid, drawDotGrid, drawObstacles } from './draw-arena';
import { drawEntities } from './draw-entities';
import { drawDecals, drawAmbient, drawRipples, drawShots, drawParticles, drawDmgNumbers } from './draw-effects';
import { drawFog } from './draw-fog';
import { drawMinimap } from './draw-minimap';
import type { CameraParams } from './math';
import type { EntityState } from '../types/telemetry';
import type { EffectsState } from '../types/effects';

export interface DrawContext {
  ctx: CanvasRenderingContext2D;
  fogCtx: CanvasRenderingContext2D | null;
  mmCtx: CanvasRenderingContext2D | null;
  canvas: HTMLCanvasElement;
  mmW: number;
  mmH: number;
  cam: CameraParams;
  arenaW: number;
  arenaH: number;
  entities: EntityState[];
  effects: EffectsState;
  obstacles: Array<{ x: number; y: number; width: number; height: number }>;
  opts: { fog: boolean; glow: boolean; grid: boolean; trails: boolean };
  followId: number | null;
}

export function drawFrame(dc: DrawContext) {
  const { ctx, canvas, cam, arenaW, arenaH, entities, effects, obstacles, opts, followId } = dc;
  const w = canvas.width, h = canvas.height;

  drawBackground(ctx, w, h);
  drawArenaBounds(ctx, w, h, cam, canvas, arenaW, arenaH);

  if (opts.grid) {
    drawDotGrid(ctx, w, h, cam, canvas, arenaW, arenaH);
  } else {
    drawLineGrid(ctx, w, h, cam, canvas, arenaW, arenaH);
  }

  drawObstacles(ctx, w, h, cam, canvas, arenaW, arenaH, obstacles);
  drawDecals(ctx, w, h, cam, canvas, arenaW, arenaH, effects);
  drawAmbient(ctx, w, h, cam, canvas, arenaW, arenaH, effects);
  drawRipples(ctx, w, h, cam, canvas, arenaW, arenaH, effects);
  drawShots(ctx, w, h, cam, canvas, arenaW, arenaH, effects, opts.glow);
  drawParticles(ctx, w, h, cam, canvas, arenaW, arenaH, effects, opts.glow);
  drawEntities(ctx, w, h, cam, canvas, arenaW, arenaH, entities, effects, opts, followId);
  drawDmgNumbers(ctx, w, h, cam, canvas, arenaW, arenaH, effects);

  if (opts.fog && dc.fogCtx) {
    drawFog(dc.fogCtx, w, h, cam, canvas, arenaW, arenaH, entities);
  }

  if (dc.mmCtx) {
    drawMinimap(dc.mmCtx, dc.mmW, dc.mmH, arenaW, arenaH, entities, obstacles, cam.camX, cam.camY, cam.camZoom, followId);
  }
}
