export type Vec2 = [number, number];

export interface EntityState {
  id: number;
  position: Vec2;
  velocity: Vec2;
  health: number;
  max_health: number;
  team: number;
  is_dead: boolean;
  facing: number;
}

export type TelemetryEvent =
  | { type: 'WorldSnapshot'; tick: number; entities: EntityState[] }
  | { type: 'Kill'; tick: number; killer: number; victim: number }
  | { type: 'Damage'; tick: number; source: number; target: number; amount: number }
  | { type: 'ShotFired'; tick: number; shooter: number; origin: Vec2; direction: Vec2; hit_target: number | null }
  | { type: 'Spawn'; tick: number; entity: number; position: Vec2; team: number }
  | { type: 'TickComplete'; tick: number; entity_count: number }
  | { type: 'RoundStart'; tick: number; obstacles: Array<{ x: number; y: number; width: number; height: number }>; spawn_points: Vec2[] }
  | { type: 'TacticalState'; tick: number; entity: number; move_target: number; candidates: [number, number][]; candidate_los: boolean[]; path: [number, number][]; aim_angle: number; shooting: boolean; ray_distances: number[] };
