export const TEAM_COLORS = ['#60a5fa', '#f87171', '#4ade80', '#facc15'];
export const TEAM_GLOW = ['#3b82f6', '#ef4444', '#22c55e', '#eab308'];
export const TEAM_DIM = ['#1e3a5f', '#5f1e1e', '#1e5f2e', '#5f5a1e'];
export const TEAM_NAMES = ['Blue', 'Red', 'Green', 'Yellow'];

export const ZOOM_MIN = 1.0;
export const ZOOM_MAX = 8.0;
export const CAM_LERP = 0.08;

export const CINEMATIC_ZOOM_MIN = 1.2;
export const CINEMATIC_ZOOM_MAX = 5.0;
export const CINEMATIC_LERP = 0.04;
export const CINEMATIC_PADDING = 80;

export const WEAPON_KEYS = ['rifle', 'smg', 'shotgun', 'pistol'] as const;
export type WeaponKey = (typeof WEAPON_KEYS)[number];

export function weaponFor(entityId: number): WeaponKey {
  return WEAPON_KEYS[entityId % WEAPON_KEYS.length];
}

export function shortId(id: number): string {
  return '#' + (id % 10000).toString().padStart(4, '0');
}

interface WeaponPart {
  x: number; y: number; w: number; h: number; c: string;
}

export const WEAPONS: Record<WeaponKey, WeaponPart[]> = {
  rifle: [
    { x: -7, y: -1.2, w: 5, h: 2.4, c: '#3a3530' },
    { x: -2, y: -1.8, w: 7, h: 3.6, c: '#484440' },
    { x: 5, y: -0.7, w: 11, h: 1.4, c: '#5a5550' },
    { x: 0, y: 1.8, w: 2.2, h: 3.5, c: '#3a3530' },
    { x: 0, y: -2.4, w: 5, h: 0.7, c: '#505050' },
    { x: 15, y: -1, w: 1.8, h: 2, c: '#666' },
  ],
  smg: [
    { x: -3, y: -1.5, w: 7, h: 3, c: '#4a4a48' },
    { x: 4, y: -0.6, w: 7, h: 1.2, c: '#5a5a58' },
    { x: 0, y: 1.5, w: 1.8, h: 4.5, c: '#3a3a38' },
    { x: -5, y: -0.8, w: 2.5, h: 1.6, c: '#383838' },
    { x: 10, y: -0.8, w: 1.2, h: 1.6, c: '#666' },
  ],
  shotgun: [
    { x: -9, y: -1.2, w: 6, h: 2.4, c: '#4a3828' },
    { x: -3, y: -1.8, w: 6, h: 3.6, c: '#555048' },
    { x: 3, y: -1, w: 13, h: 2, c: '#5a5550' },
    { x: 7, y: -1.5, w: 3.5, h: 3, c: '#4a3828' },
    { x: 15, y: -1.3, w: 1.3, h: 2.6, c: '#666' },
  ],
  pistol: [
    { x: -0.5, y: -1.2, w: 7, h: 2.4, c: '#4a4a48' },
    { x: 6.5, y: -0.5, w: 2.5, h: 1, c: '#5a5a58' },
    { x: -0.5, y: 1.2, w: 2.5, h: 4, c: '#3a3530' },
    { x: -0.5, y: 4.8, w: 2.5, h: 0.8, c: '#333' },
  ],
};
