import { create } from 'zustand';

export interface CsAgent {
  id: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  health: number;
  maxHealth: number;
  team: number;
  isDead: boolean;
  shooting: boolean;
  activeWeapon: number;
  moveDirection: number;
  rayDistances: number[];
  rayHitTypes: number[];
}

export interface CsRound {
  phase: string;
  roundNumber: number;
  tScore: number;
  ctScore: number;
  phaseTimer: number;
  tAlive: number;
  ctAlive: number;
}

export interface KillEntry {
  tick: number;
  killer: number;
  victim: number;
}

export interface ShotEntry {
  tick: number;
  ox: number;
  oz: number;
  dx: number;
  dz: number;
  hit: boolean;
  createdAt: number;
}

export type CameraMode = 'cinematic' | 'free';

interface CsLiteState {
  agents: Map<number, CsAgent>;
  round: CsRound;
  obstacles: Array<{ x: number; y: number; width: number; height: number }>;
  spawnPoints: [number, number][];
  kills: KillEntry[];
  shots: ShotEntry[];
  cameraMode: CameraMode;
  xray: boolean;

  updateAgent: (agent: CsAgent) => void;
  updateRound: (round: CsRound) => void;
  setObstacles: (obs: Array<{ x: number; y: number; width: number; height: number }>, spawns: [number, number][]) => void;
  addKill: (kill: KillEntry) => void;
  addShot: (shot: Omit<ShotEntry, 'createdAt'>) => void;
  toggleCameraMode: () => void;
  toggleXRay: () => void;
  reset: () => void;
}

const initialRound: CsRound = {
  phase: 'buy_freeze',
  roundNumber: 1,
  tScore: 0,
  ctScore: 0,
  phaseTimer: 3,
  tAlive: 5,
  ctAlive: 5,
};

export const useCsLiteStore = create<CsLiteState>((set) => ({
  agents: new Map(),
  round: initialRound,
  obstacles: [],
  spawnPoints: [],
  kills: [],
  shots: [],
  cameraMode: 'cinematic' as CameraMode,
  xray: true,

  updateAgent: (agent) =>
    set((s) => {
      const next = new Map(s.agents);
      next.set(agent.id, agent);
      return { agents: next };
    }),

  updateRound: (round) => set({ round }),

  setObstacles: (obs, spawns) => set({ obstacles: obs, spawnPoints: spawns }),

  addKill: (kill) =>
    set((s) => ({ kills: [...s.kills.slice(-19), kill] })),

  addShot: (shot) =>
    set((s) => ({ shots: [...s.shots.slice(-19), { ...shot, createdAt: Date.now() }] })),

  toggleCameraMode: () =>
    set((s) => ({ cameraMode: s.cameraMode === 'cinematic' ? 'free' : 'cinematic' })),

  toggleXRay: () =>
    set((s) => ({ xray: !s.xray })),

  reset: () =>
    set({
      agents: new Map(),
      round: initialRound,
      kills: [],
      shots: [],
    }),
}));
