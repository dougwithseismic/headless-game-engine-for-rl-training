import type { GameConfig } from '../types/config';
import type { ScenarioDefinition } from './types';
import { demoScenario } from './demo';
import { csLiteScenario } from './cs-lite';

const scenarios: ScenarioDefinition[] = [
  csLiteScenario,
];

export function resolveScenario(config: GameConfig | undefined): ScenarioDefinition {
  if (!config) return demoScenario;
  return scenarios.find(s => s.match(config)) ?? demoScenario;
}

export { type ScenarioDefinition };
