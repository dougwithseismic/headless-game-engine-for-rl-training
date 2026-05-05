import { useEffect } from 'react';
import type { RefObject, MutableRefObject } from 'react';
import { useCameraStore } from '../stores/camera-store';
import { useGameStore } from '../stores/game-store';
import { ZOOM_MIN, ZOOM_MAX } from '../constants';
import { fromCanvas } from '../renderer/math';
import type { EffectsState } from '../types/effects';

export function useCameraControls(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  effectsRef: MutableRefObject<EffectsState>,
  arenaW: number,
  arenaH: number,
) {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;

      const cam = useCameraStore.getState();
      const camParams = { camX: cam.camX, camY: cam.camY, camZoom: cam.camZoom, shakeX: cam.shakeX, shakeY: cam.shakeY };
      const [ax, ay] = fromCanvas(mx, my, canvas, arenaW, arenaH, camParams);
      const f = ev.deltaY < 0 ? 1.06 : 1 / 1.06;
      const newZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, cam.camZoom * f));

      if (newZoom > ZOOM_MIN + 0.01) {
        const ppuX = canvas.width / arenaW;
        const ppuY = canvas.height / arenaH;
        useCameraStore.setState({
          camZoom: newZoom,
          camX: ax - (mx - canvas.width / 2) / (ppuX * newZoom),
          camY: ay - (my - canvas.height / 2) / (ppuY * newZoom),
          cinematic: false,
        });
      } else {
        useCameraStore.setState({ camZoom: ZOOM_MIN, camX: arenaW / 2, camY: arenaH / 2, cinematic: false });
        if (cam.followId !== null) useCameraStore.getState().stopFollowing();
      }
    };

    const onClick = (ev: MouseEvent) => {
      const cam = useCameraStore.getState();
      if (cam.isPanning) return;
      const rect = canvas.getBoundingClientRect();
      const camParams = { camX: cam.camX, camY: cam.camY, camZoom: cam.camZoom, shakeX: cam.shakeX, shakeY: cam.shakeY };
      const [ax, ay] = fromCanvas(ev.clientX - rect.left, ev.clientY - rect.top, canvas, arenaW, arenaH, camParams);
      const entities = useGameStore.getState().entities;

      let best: typeof entities[0] | null = null;
      let bestD = Infinity;
      for (const e of entities) {
        if (e.is_dead) continue;
        const d = Math.hypot(e.position[0] - ax, e.position[1] - ay);
        if (d < 20 / cam.camZoom && d < bestD) { best = e; bestD = d; }
      }
      if (best) {
        useCameraStore.getState().follow(best.id);
      } else {
        useCameraStore.getState().stopFollowing();
      }
    };

    const onMouseDown = (ev: MouseEvent) => {
      if (ev.button === 1 || ev.button === 2) {
        ev.preventDefault();
        useCameraStore.getState().startPan(ev.clientX, ev.clientY);
        canvas.style.cursor = 'grabbing';
      }
    };

    const onMouseMove = (ev: MouseEvent) => {
      const cam = useCameraStore.getState();
      if (cam.isPanning) {
        cam.updatePan(ev.clientX, ev.clientY, canvas.width, canvas.height, arenaW, arenaH);
      }
      const rect = canvas.getBoundingClientRect();
      effectsRef.current.mouseCanvasX = ev.clientX - rect.left;
      effectsRef.current.mouseCanvasY = ev.clientY - rect.top;
    };

    const onMouseUp = (ev: MouseEvent) => {
      if ((ev.button === 1 || ev.button === 2) && useCameraStore.getState().isPanning) {
        useCameraStore.getState().endPan();
        canvas.style.cursor = 'crosshair';
      }
    };

    const onContextMenu = (ev: MouseEvent) => ev.preventDefault();

    const onKeyDown = (ev: KeyboardEvent) => {
      const cam = useCameraStore.getState();
      if (ev.key === 'Escape') {
        if (cam.cinematic) {
          cam.toggleCinematic();
        } else if (cam.followId !== null) {
          cam.stopFollowing();
        } else if (cam.camZoom > 1.05) {
          useCameraStore.setState({ camZoom: 1, camX: arenaW / 2, camY: arenaH / 2 });
        }
      } else if (ev.key === 'c' || ev.key === 'C') {
        useCameraStore.getState().toggleCinematic();
      } else if (ev.key === '=' || ev.key === '+') {
        useCameraStore.getState().setZoom(cam.camZoom * 1.25);
      } else if (ev.key === '-') {
        const newZoom = Math.max(ZOOM_MIN, cam.camZoom / 1.25);
        useCameraStore.setState({ camZoom: newZoom });
        if (newZoom <= ZOOM_MIN + 0.01) {
          useCameraStore.setState({ camX: arenaW / 2, camY: arenaH / 2 });
        }
      } else if (ev.key >= '1' && ev.key <= '9') {
        const alive = useGameStore.getState().entities
          .filter(e => !e.is_dead)
          .sort((a, b) => a.team - b.team || a.id - b.id);
        const i = parseInt(ev.key) - 1;
        if (i < alive.length) useCameraStore.getState().follow(alive[i].id);
      } else if (ev.key === '0') {
        useCameraStore.setState({ camZoom: 1, camX: arenaW / 2, camY: arenaH / 2 });
        useCameraStore.getState().stopFollowing();
      }
    };

    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('click', onClick);
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('contextmenu', onContextMenu);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('keydown', onKeyDown);

    return () => {
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('click', onClick);
      canvas.removeEventListener('mousedown', onMouseDown);
      canvas.removeEventListener('contextmenu', onContextMenu);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [canvasRef, effectsRef, arenaW, arenaH]);
}
