export interface CameraParams {
  camX: number;
  camY: number;
  camZoom: number;
  shakeX: number;
  shakeY: number;
}

export function toCanvas(
  ax: number, ay: number,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  cam: CameraParams,
): [number, number] {
  const ppuX = canvas.width / arenaW;
  const ppuY = canvas.height / arenaH;
  return [
    canvas.width / 2 + (ax - cam.camX) * ppuX * cam.camZoom + cam.shakeX,
    canvas.height / 2 + (ay - cam.camY) * ppuY * cam.camZoom + cam.shakeY,
  ];
}

export function fromCanvas(
  sx: number, sy: number,
  canvas: { width: number; height: number },
  arenaW: number, arenaH: number,
  cam: CameraParams,
): [number, number] {
  const ppuX = canvas.width / arenaW;
  const ppuY = canvas.height / arenaH;
  return [
    cam.camX + (sx - cam.shakeX - canvas.width / 2) / (ppuX * cam.camZoom),
    cam.camY + (sy - cam.shakeY - canvas.height / 2) / (ppuY * cam.camZoom),
  ];
}

export function scale(cam: CameraParams, canvas: { width: number }, arenaW: number): number {
  return cam.camZoom * canvas.width / arenaW;
}

export function toMinimap(
  ax: number, ay: number,
  arenaW: number, arenaH: number,
  mmW: number, mmH: number,
): [number, number] {
  return [(ax / arenaW) * mmW, (ay / arenaH) * mmH];
}
