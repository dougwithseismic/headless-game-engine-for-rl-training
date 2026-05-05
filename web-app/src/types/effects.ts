import type { EntityState } from './telemetry';

export interface Particle {
  x: number; y: number;
  vx: number; vy: number;
  life: number; decay: number;
  size: number; color: string;
  type: 'kill' | 'damage' | 'spawn';
}

export interface ShotTrace {
  ox: number; oy: number;
  ex: number; ey: number;
  hit: boolean; alpha: number;
}

export interface DmgNumber {
  x: number; y: number;
  text: string; color: string;
  life: number;
  vy: number; vx: number;
}

export interface Decal {
  x: number; y: number;
  color: string; alpha: number;
  size: number;
}

export interface Ripple {
  x: number; y: number;
  radius: number; maxRadius: number;
  alpha: number;
}

export interface AmbientParticle {
  x: number; y: number;
  vx: number; vy: number;
  size: number; alpha: number;
}

export interface EffectsState {
  particles: Particle[];
  shotTraces: ShotTrace[];
  dmgNumbers: DmgNumber[];
  decals: Decal[];
  ripples: Ripple[];
  ambientParticles: AmbientParticle[];
  prevPositions: Record<number, { x: number; y: number }>;
  entityIdMap: Record<number, EntityState>;
  hoverEntity: EntityState | null;
  mouseCanvasX: number;
  mouseCanvasY: number;
}

export function createEffectsState(): EffectsState {
  return {
    particles: [],
    shotTraces: [],
    dmgNumbers: [],
    decals: [],
    ripples: [],
    ambientParticles: [],
    prevPositions: {},
    entityIdMap: {},
    hoverEntity: null,
    mouseCanvasX: 0,
    mouseCanvasY: 0,
  };
}
