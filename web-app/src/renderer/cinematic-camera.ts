import type { EntityState } from '../types/telemetry';
import {
  CINEMATIC_ZOOM_MIN,
  CINEMATIC_ZOOM_MAX,
  CINEMATIC_LERP,
  CINEMATIC_PADDING,
} from '../constants';

export function computeCinematicCamera(
  entities: EntityState[],
  currentX: number,
  currentY: number,
  currentZoom: number,
  arenaW: number,
  arenaH: number,
  canvasW: number,
  canvasH: number,
): { x: number; y: number; zoom: number } {
  const alive = entities.filter(e => !e.is_dead);
  if (alive.length === 0) {
    return { x: arenaW / 2, y: arenaH / 2, zoom: CINEMATIC_ZOOM_MIN };
  }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const e of alive) {
    if (e.position[0] < minX) minX = e.position[0];
    if (e.position[1] < minY) minY = e.position[1];
    if (e.position[0] > maxX) maxX = e.position[0];
    if (e.position[1] > maxY) maxY = e.position[1];
  }

  const targetX = (minX + maxX) / 2;
  const targetY = (minY + maxY) / 2;

  const spreadX = maxX - minX + CINEMATIC_PADDING * 2;
  const spreadY = maxY - minY + CINEMATIC_PADDING * 2;

  const ppuX = canvasW / arenaW;
  const ppuY = canvasH / arenaH;

  const zoomX = ppuX > 0 ? canvasW / (spreadX * ppuX) : CINEMATIC_ZOOM_MIN;
  const zoomY = ppuY > 0 ? canvasH / (spreadY * ppuY) : CINEMATIC_ZOOM_MIN;
  const targetZoom = Math.max(
    CINEMATIC_ZOOM_MIN,
    Math.min(CINEMATIC_ZOOM_MAX, Math.min(zoomX, zoomY)),
  );

  const lerp = CINEMATIC_LERP;
  return {
    x: currentX + (targetX - currentX) * lerp,
    y: currentY + (targetY - currentY) * lerp,
    zoom: currentZoom + (targetZoom - currentZoom) * lerp,
  };
}
