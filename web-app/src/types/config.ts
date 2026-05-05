export interface GameConfig {
  title: string;
  scenario?: string;
  tick_rate: number;
  arena: { width: number; height: number };
  movement: { max_speed: number; acceleration: number; friction: number };
  combat: { default_weapon: { damage: number; fire_rate: number; range: number } };
  spawning: { respawn_delay: number };
  teams: { count: number; players_per_team: number };
  obstacles: Array<{ x: number; y: number; width: number; height: number }>;
  extra?: Record<string, unknown>;
}

export interface MatchResponse {
  title: string;
  tick: number;
  tick_rate: number;
  status: string;
}
