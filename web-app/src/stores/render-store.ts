import { create } from 'zustand';

interface RenderState {
  fog: boolean;
  glow: boolean;
  grid: boolean;
  trails: boolean;
  weather: boolean;
  toggle: (key: 'fog' | 'glow' | 'grid' | 'trails' | 'weather') => void;
}

export const useRenderStore = create<RenderState>((set) => ({
  fog: true,
  glow: true,
  grid: false,
  trails: true,
  weather: true,
  toggle: (key) => set((s) => ({ [key]: !s[key] })),
}));
