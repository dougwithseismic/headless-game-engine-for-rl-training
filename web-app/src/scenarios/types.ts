import type { ComponentType } from 'react';
import type { GameConfig } from '../types/config';
import type { TelemetryEvent } from '../types/telemetry';

export interface ScenarioDefinition {
  id: string;
  name: string;
  match: (config: GameConfig) => boolean;
  Canvas: ComponentType;
  sidebarPanels: ComponentType[];
  onTelemetryEvent: (event: TelemetryEvent) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}
