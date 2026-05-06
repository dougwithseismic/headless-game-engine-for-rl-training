import { create } from 'zustand';
import type { EntityState } from '../types/telemetry';
import { TEAM_COLORS, shortId } from '../constants';

export interface Obstacle {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface KillEntry {
  tick: number;
  killerId: number;
  victimId: number;
  kCol: string;
  vCol: string;
  wep: string;
}

export interface TacticalState {
  moveTarget: number;
  candidates: [number, number][];
  candidateLos: boolean[];
  path: [number, number][];
  aimAngle: number;
  shooting: boolean;
  rayDistances: number[];
}

interface GameState {
  entities: EntityState[];
  tick: number;
  entityIdMap: Record<number, EntityState>;
  kills: KillEntry[];
  score: number[];
  eventLog: string[];
  tps: number;
  connected: boolean;
  obstacles: Obstacle[];
  spawnPoints: [number, number][];
  tacticalStates: Record<number, TacticalState>;

  processWorldSnapshot: (tick: number, entities: EntityState[]) => void;
  processKill: (tick: number, killerId: number, victimId: number) => void;
  addLogEntry: (entry: string) => void;
  updateTps: (tps: number) => void;
  setConnected: (connected: boolean) => void;
  setObstacles: (obstacles: Obstacle[]) => void;
  setSpawnPoints: (spawnPoints: [number, number][]) => void;
  setTacticalState: (entityId: number, state: TacticalState) => void;
}

export const useGameStore = create<GameState>((set, get) => ({
  entities: [],
  tick: 0,
  entityIdMap: {},
  kills: [],
  score: [0, 0],
  eventLog: [],
  tps: 0,
  connected: false,
  obstacles: [],
  spawnPoints: [],
  tacticalStates: {},

  processWorldSnapshot: (tick, entities) => {
    const entityIdMap: Record<number, EntityState> = {};
    for (const e of entities) entityIdMap[e.id] = e;
    set({ tick, entities, entityIdMap });
  },

  processKill: (tick, killerId, victimId) => {
    const { entityIdMap, kills, score } = get();
    const kTeam = entityIdMap[killerId]?.team ?? 0;
    const vTeam = entityIdMap[victimId]?.team ?? 1;
    const kCol = TEAM_COLORS[kTeam] || '#fff';
    const vCol = TEAM_COLORS[vTeam] || '#fff';
    const WEAPON_KEYS = ['rifle', 'smg', 'shotgun', 'pistol'];
    const wep = WEAPON_KEYS[killerId % WEAPON_KEYS.length];

    const newScore = [...score];
    if (kTeam !== vTeam) newScore[kTeam] = (newScore[kTeam] || 0) + 1;

    const newKills = [{ tick, killerId, victimId, kCol, vCol, wep }, ...kills].slice(0, 12);

    const logEntry = `<span class="timestamp">[${String(tick).padStart(6, '0')}]</span> <span class="event-type event-kill">KILL </span> <span class="hl-id">${shortId(killerId)}</span> eliminated <span class="hl-id">${shortId(victimId)}</span>`;

    set(state => ({
      kills: newKills,
      score: newScore,
      eventLog: [logEntry, ...state.eventLog].slice(0, 200),
    }));
  },

  addLogEntry: (entry) => set(state => ({
    eventLog: [entry, ...state.eventLog].slice(0, 200),
  })),

  updateTps: (tps) => set({ tps }),
  setConnected: (connected) => set({ connected }),
  setObstacles: (obstacles) => set({ obstacles }),
  setSpawnPoints: (spawnPoints) => set({ spawnPoints }),
  setTacticalState: (entityId, state) => set(s => ({
    tacticalStates: { ...s.tacticalStates, [entityId]: state },
  })),
}));
