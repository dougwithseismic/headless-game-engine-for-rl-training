import { create } from 'zustand';

interface RenderState {
  fog: boolean;
  glow: boolean;
  grid: boolean;
  trails: boolean;
  weather: boolean;
  tactical: boolean;
  toggle: (key: 'fog' | 'glow' | 'grid' | 'trails' | 'weather' | 'tactical') => void;
}

export const useRenderStore = create<RenderState>((set) => ({
  fog: true,
  glow: true,
  grid: false,
  trails: true,
  weather: true,
  tactical: true,
  toggle: (key) => set((s) => ({ [key]: !s[key] })),
}));
