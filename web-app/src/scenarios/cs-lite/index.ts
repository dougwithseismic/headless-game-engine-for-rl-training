import type { ScenarioDefinition } from '../types';
import type { TelemetryEvent } from '../../types/telemetry';
import { CsLiteCanvas } from './CsLiteCanvas';
import { CsLiteSidebar } from './CsLiteSidebar';
import { useCsLiteStore } from './store';

export const csLiteScenario: ScenarioDefinition = {
  id: 'cs-lite',
  name: 'CS-Lite 5v5',
  match: (config) => {
    const scenario = config.extra?.scenario as string | undefined;
    if (scenario === 'cs_lite' || scenario === 'cs-lite') return true;
    const title = config.title.toLowerCase();
    return title.includes('cs_lite') || title.includes('cs-lite') || title.includes('counterstrike');
  },
  Canvas: CsLiteCanvas,
  sidebarPanels: [CsLiteSidebar],
  onTelemetryEvent: (event: TelemetryEvent) => {
    const store = useCsLiteStore.getState();

    switch (event.type) {
      case 'Arena3DState':
        store.updateAgent({
          id: event.entity,
          x: event.position[0],
          y: event.position[1],
          z: event.position[2],
          yaw: event.yaw,
          pitch: event.pitch,
          health: event.health,
          maxHealth: event.max_health,
          team: event.team,
          isDead: event.is_dead,
          shooting: event.shooting,
          activeWeapon: event.active_weapon,
          moveDirection: event.move_direction,
          rayDistances: event.ray_distances ?? [],
          rayHitTypes: event.ray_hit_types ?? [],
        });
        break;

      case 'CsLiteRoundState':
        store.updateRound({
          phase: event.phase,
          roundNumber: event.round_number,
          tScore: event.t_score,
          ctScore: event.ct_score,
          phaseTimer: event.phase_timer,
          tAlive: event.t_alive,
          ctAlive: event.ct_alive,
        });
        break;

      case 'RoundStart':
        store.setObstacles(event.obstacles, event.spawn_points);
        break;

      case 'Kill':
        store.addKill({ tick: event.tick, killer: event.killer, victim: event.victim });
        break;

      case 'ShotFired':
        store.addShot({
          tick: event.tick,
          ox: event.origin[0],
          oz: event.origin[1],
          dx: event.direction[0],
          dz: event.direction[1],
          hit: event.hit_target !== null,
        });
        break;
    }
  },
  onConnect: () => {
    useCsLiteStore.getState().reset();
  },
  onDisconnect: () => {
    useCsLiteStore.getState().reset();
  },
};
