import { useViewerStore } from '../../../stores/viewer-store';
import { useGameConfig } from '../../../hooks/use-game-config';
import { useServerHost } from '../../../contexts/server';

export function DemoCanvas() {
  const host = useServerHost();
  const { data: config } = useGameConfig(host);
  const connected = useViewerStore(s => s.connected);
  const tick = useViewerStore(s => s.tick);
  const entityCount = useViewerStore(s => s.entityCount);

  return (
    <div className="viewport">
      <div className="demo-canvas-inner">
        <div className="demo-status">
          <span className={`conn-dot${connected ? ' live' : ''}`} />
          {connected ? 'Connected' : 'Connecting...'}
        </div>
        <h2 className="demo-title">{config?.title ?? 'Loading...'}</h2>
        <p className="demo-subtitle">No specialized viewer registered for this scenario.</p>
        {connected && (
          <div className="demo-stats">
            <div className="demo-stat">
              <span className="demo-stat-label">Tick</span>
              <span className="demo-stat-value">{tick.toLocaleString()}</span>
            </div>
            <div className="demo-stat">
              <span className="demo-stat-label">Entities</span>
              <span className="demo-stat-value">{entityCount}</span>
            </div>
            <div className="demo-stat">
              <span className="demo-stat-label">Arena</span>
              <span className="demo-stat-value">
                {config ? `${config.arena.width} x ${config.arena.height}` : '—'}
              </span>
            </div>
          </div>
        )}
        <p className="demo-hint">
          Register a scenario in <code>src/scenarios/</code> to add a viewer.
        </p>
      </div>
    </div>
  );
}
