import { describe, it, expect } from 'vitest';
import { resolveScenario } from './registry';
import type { GameConfig } from '../types/config';

const makeConfig = (overrides: Partial<GameConfig> = {}): GameConfig => ({
  title: 'test',
  tick_rate: 64,
  arena: { width: 1000, height: 1000 },
  movement: { max_speed: 200, acceleration: 800, friction: 600 },
  combat: { default_weapon: { damage: 20, fire_rate: 5, range: 500 } },
  spawning: { respawn_delay: 3 },
  teams: { count: 2, players_per_team: 1 },
  obstacles: [],
  ...overrides,
});

describe('resolveScenario', () => {
  it('returns demo scenario when config is undefined', () => {
    const scenario = resolveScenario(undefined);
    expect(scenario.id).toBe('demo');
  });

  it('returns demo scenario when no registered scenario matches', () => {
    const scenario = resolveScenario(makeConfig({ title: 'unknown-thing' }));
    expect(scenario.id).toBe('demo');
  });

  it('demo scenario has required fields', () => {
    const scenario = resolveScenario(undefined);
    expect(scenario.name).toBeTruthy();
    expect(scenario.Canvas).toBeDefined();
    expect(scenario.sidebarPanels).toBeInstanceOf(Array);
    expect(typeof scenario.onTelemetryEvent).toBe('function');
    expect(typeof scenario.match).toBe('function');
  });

  it('demo scenario match returns false', () => {
    const scenario = resolveScenario(undefined);
    expect(scenario.match(makeConfig())).toBe(false);
  });
});
