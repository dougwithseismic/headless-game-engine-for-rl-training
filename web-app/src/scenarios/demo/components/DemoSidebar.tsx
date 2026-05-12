import { useGameConfig } from '../../../hooks/use-game-config';
import { useServerHost } from '../../../contexts/server';

export function DemoSidebar() {
  const host = useServerHost();
  const { data: config } = useGameConfig(host);

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Scenario Info</span>
      </div>
      <div style={{ padding: '8px 0', fontSize: 12, color: 'var(--text-secondary)' }}>
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: 'var(--text-dim)' }}>Title: </span>
          {config?.title ?? '—'}
        </div>
        <div style={{ marginBottom: 4 }}>
          <span style={{ color: 'var(--text-dim)' }}>Tick rate: </span>
          {config?.tick_rate ?? '—'}
        </div>
        <div>
          <span style={{ color: 'var(--text-dim)' }}>Arena: </span>
          {config ? `${config.arena.width} x ${config.arena.height}` : '—'}
        </div>
      </div>
    </div>
  );
}
