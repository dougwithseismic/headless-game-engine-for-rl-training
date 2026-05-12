import { useNavigate, useSearchParams } from 'react-router-dom';
import { ServerHostProvider } from '../contexts/server';
import { useGameConfig } from '../hooks/use-game-config';
import { useViewerWebSocket } from '../hooks/use-websocket';
import { useViewerStore } from '../stores/viewer-store';
import { resolveScenario } from '../scenarios/registry';
import { Sidebar } from '../components/layout/Sidebar';
import { TrainingPanel } from '../components/sidebar/TrainingPanel';

function ViewerHeader({ host }: { host: string }) {
  const navigate = useNavigate();
  const { data: config } = useGameConfig(host);
  const connected = useViewerStore(s => s.connected);
  const tick = useViewerStore(s => s.tick);
  const tps = useViewerStore(s => s.tps);
  const entityCount = useViewerStore(s => s.entityCount);

  return (
    <header>
      <div className="header-left">
        <button onClick={() => navigate('/')} className="header-back" aria-label="Back to dashboard">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M10 12L6 8L10 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        <span className="logo">GhostLobby</span>
        <span className="logo-sep">/</span>
        <span className="match-title">{config?.title ?? 'Connecting...'}</span>
      </div>
      <div className="status-bar">
        <span>
          <span className={`conn-dot${connected ? ' live' : ''}`} />
          <span>{connected ? 'Live' : 'Offline'}</span>
        </span>
        <span>
          <span className="label">Tick</span>
          <span className="value">{tick.toLocaleString()}</span>
        </span>
        <span>
          <span className="label">Ent</span>
          <span className="value">{entityCount}</span>
        </span>
        <span>
          <span className="label">TPS</span>
          <span className="value">{tps ? tps.toLocaleString() : '—'}</span>
        </span>
        <span className="status-host">{host}</span>
      </div>
    </header>
  );
}

function ViewerContent({ host }: { host: string }) {
  const { data: config } = useGameConfig(host);
  const scenario = resolveScenario(config);

  useViewerWebSocket(host, scenario, !!config);

  return (
    <div className="main">
      <scenario.Canvas />
      <Sidebar>
        <TrainingPanel />
        {scenario.sidebarPanels.map((Panel, i) => (
          <Panel key={i} />
        ))}
      </Sidebar>
    </div>
  );
}

export function ViewerPage() {
  const [params] = useSearchParams();
  const host = params.get('host') ?? 'localhost:3000';

  return (
    <ServerHostProvider value={host}>
      <ViewerHeader host={host} />
      <ViewerContent host={host} />
    </ServerHostProvider>
  );
}
