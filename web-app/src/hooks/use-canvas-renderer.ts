import { useEffect, useRef } from 'react';
import type { MutableRefObject, RefObject } from 'react';
import { drawFrame } from '../renderer/draw';
import { computeCinematicCamera } from '../renderer/cinematic-camera';
import { useGameStore } from '../stores/game-store';
import { useCameraStore } from '../stores/camera-store';
import { useRenderStore } from '../stores/render-store';
import { CAM_LERP } from '../constants';
import type { EffectsState } from '../types/effects';

export function useCanvasRenderer(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  fogCanvasRef: RefObject<HTMLCanvasElement | null>,
  mmCanvasRef: RefObject<HTMLCanvasElement | null>,
  effectsRef: MutableRefObject<EffectsState>,
  arenaW: number,
  arenaH: number,
) {
  const fpsRef = useRef({ count: 0, lastTime: performance.now(), value: 0 });
  const fpsElRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    let raf: number;

    const draw = () => {
      const canvas = canvasRef.current;
      if (!canvas) { raf = requestAnimationFrame(draw); return; }

      const ctx = canvas.getContext('2d');
      if (!ctx) { raf = requestAnimationFrame(draw); return; }

      const fogCanvas = fogCanvasRef.current;
      const fogCtx = fogCanvas?.getContext('2d') ?? null;
      const mmCanvas = mmCanvasRef.current;
      const mmCtx = mmCanvas?.getContext('2d') ?? null;

      const { entities, obstacles, tacticalStates } = useGameStore.getState();
      const camera = useCameraStore.getState();
      const opts = useRenderStore.getState();

      // Update camera: cinematic mode or follow mode
      if (camera.cinematic) {
        const { x, y, zoom } = computeCinematicCamera(
          entities, camera.camX, camera.camY, camera.camZoom,
          arenaW, arenaH, canvas.width, canvas.height,
        );
        useCameraStore.setState({ camX: x, camY: y, camZoom: zoom });
      } else if (camera.followId !== null) {
        const target = entities.find(e => e.id === camera.followId);
        if (target && !target.is_dead) {
          const newX = camera.camX + (target.position[0] - camera.camX) * CAM_LERP;
          const newY = camera.camY + (target.position[1] - camera.camY) * CAM_LERP;
          useCameraStore.setState({ camX: newX, camY: newY });
        } else if (target?.is_dead) {
          useCameraStore.getState().stopFollowing();
        }
      }

      // Tick screen shake
      camera.tickShake();
      const cam = useCameraStore.getState();

      drawFrame({
        ctx,
        fogCtx,
        mmCtx,
        canvas,
        mmW: mmCanvas?.width ?? 130,
        mmH: mmCanvas?.height ?? 130,
        cam: { camX: cam.camX, camY: cam.camY, camZoom: cam.camZoom, shakeX: cam.shakeX, shakeY: cam.shakeY },
        arenaW,
        arenaH,
        entities,
        effects: effectsRef.current,
        obstacles,
        tacticalStates,
        opts: { fog: opts.fog, glow: opts.glow, grid: opts.grid, trails: opts.trails, weather: opts.weather, tactical: opts.tactical },
        followId: cam.followId,
      });

      // FPS counter
      fpsRef.current.count++;
      const now = performance.now();
      if (now - fpsRef.current.lastTime > 1000) {
        fpsRef.current.value = Math.round(fpsRef.current.count / ((now - fpsRef.current.lastTime) / 1000));
        fpsRef.current.count = 0;
        fpsRef.current.lastTime = now;
        if (!fpsElRef.current) fpsElRef.current = document.getElementById('fps-value');
        if (fpsElRef.current) fpsElRef.current.textContent = String(fpsRef.current.value);
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [canvasRef, fogCanvasRef, mmCanvasRef, effectsRef, arenaW, arenaH]);
}
