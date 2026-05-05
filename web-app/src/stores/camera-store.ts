import { create } from 'zustand';
import { ZOOM_MIN, ZOOM_MAX } from '../constants';

interface CameraState {
  camX: number;
  camY: number;
  camZoom: number;
  followId: number | null;
  cinematic: boolean;

  isPanning: boolean;
  panStartX: number;
  panStartY: number;
  camStartX: number;
  camStartY: number;

  shakeX: number;
  shakeY: number;
  shakeDecay: number;

  setPosition: (x: number, y: number) => void;
  setZoom: (zoom: number) => void;
  follow: (id: number, zoom?: number) => void;
  stopFollowing: () => void;
  toggleCinematic: () => void;
  startPan: (mouseX: number, mouseY: number) => void;
  updatePan: (mouseX: number, mouseY: number, canvasW: number, canvasH: number, arenaW: number, arenaH: number) => void;
  endPan: () => void;
  addShake: (intensity: number) => void;
  tickShake: () => void;
  reset: (arenaW: number, arenaH: number) => void;
}

export const useCameraStore = create<CameraState>((set, get) => ({
  camX: 500,
  camY: 500,
  camZoom: 1.0,
  followId: null,
  cinematic: false,

  isPanning: false,
  panStartX: 0,
  panStartY: 0,
  camStartX: 0,
  camStartY: 0,

  shakeX: 0,
  shakeY: 0,
  shakeDecay: 0,

  setPosition: (x, y) => set({ camX: x, camY: y }),
  setZoom: (zoom) => set({ camZoom: Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom)) }),

  follow: (id, zoom) => {
    const updates: Partial<CameraState> = { followId: id, cinematic: false };
    if (zoom !== undefined) updates.camZoom = zoom;
    else if (get().camZoom < 2.5) updates.camZoom = 3.5;
    set(updates);
  },

  stopFollowing: () => set({ followId: null }),

  toggleCinematic: () => set(s => ({
    cinematic: !s.cinematic,
    followId: !s.cinematic ? null : s.followId,
  })),

  startPan: (mouseX, mouseY) => set(s => ({
    isPanning: true,
    panStartX: mouseX,
    panStartY: mouseY,
    camStartX: s.camX,
    camStartY: s.camY,
    followId: null,
    cinematic: false,
  })),

  updatePan: (mouseX, mouseY, canvasW, canvasH, arenaW, arenaH) => {
    const s = get();
    if (!s.isPanning) return;
    const ppuX = canvasW / arenaW;
    const ppuY = canvasH / arenaH;
    set({
      camX: s.camStartX - (mouseX - s.panStartX) / (ppuX * s.camZoom),
      camY: s.camStartY - (mouseY - s.panStartY) / (ppuY * s.camZoom),
    });
  },

  endPan: () => set({ isPanning: false }),

  addShake: (intensity) => set(s => ({
    shakeDecay: Math.max(s.shakeDecay, intensity),
  })),

  tickShake: () => {
    const s = get();
    if (s.shakeDecay > 0.01) {
      set({
        shakeX: (Math.random() - 0.5) * s.shakeDecay * 6,
        shakeY: (Math.random() - 0.5) * s.shakeDecay * 6,
        shakeDecay: s.shakeDecay * 0.88,
      });
    } else if (s.shakeX !== 0 || s.shakeY !== 0) {
      set({ shakeX: 0, shakeY: 0, shakeDecay: 0 });
    }
  },

  reset: (arenaW, arenaH) => set({
    camX: arenaW / 2,
    camY: arenaH / 2,
    camZoom: 1.0,
    followId: null,
    cinematic: false,
  }),
}));
