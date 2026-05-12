import type { ScenarioDefinition } from '../types';
import { DemoCanvas } from './components/DemoCanvas';
import { DemoSidebar } from './components/DemoSidebar';

export const demoScenario: ScenarioDefinition = {
  id: 'demo',
  name: 'Demo',
  match: () => false,
  Canvas: DemoCanvas,
  sidebarPanels: [DemoSidebar],
  onTelemetryEvent: () => {},
};
