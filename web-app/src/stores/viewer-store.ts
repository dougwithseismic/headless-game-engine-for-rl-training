import { create } from 'zustand';

interface ViewerState {
  connected: boolean;
  tick: number;
  tps: number;
  entityCount: number;
  setConnected: (v: boolean) => void;
  updateTick: (tick: number, entityCount: number) => void;
  updateTps: (tps: number) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  connected: false,
  tick: 0,
  tps: 0,
  entityCount: 0,
  setConnected: (connected) => set({ connected }),
  updateTick: (tick, entityCount) => set({ tick, entityCount }),
  updateTps: (tps) => set({ tps }),
}));
