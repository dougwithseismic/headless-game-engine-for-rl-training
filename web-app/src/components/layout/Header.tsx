import { useGameStore } from '../../stores/game-store';
import { useGameConfig } from '../../hooks/use-game-config';

export function Header() {
  const tick = useGameStore(s => s.tick);
  const entities = useGameStore(s => s.entities);
  const tps = useGameStore(s => s.tps);
  const connected = useGameStore(s => s.connected);
  const { data: config } = useGameConfig();

  return (
    <header>
      <div className="header-left">
        <span className="logo">GhostLobby</span>
        <span className="logo-sep">/</span>
        <span className="match-title">
          {config?.title?.toUpperCase() || 'CONNECTING...'}
        </span>
      </div>
      <div className="status-bar">
        <span>
          <span className={`conn-dot${connected ? ' live' : ''}`} />
          <span>{connected ? 'live' : 'offline'}</span>
        </span>
        <span>
          <span className="label">tick</span>
          <span className="value">{tick.toLocaleString()}</span>
        </span>
        <span>
          <span className="label">ent</span>
          <span className="value">{entities.length}</span>
        </span>
        <span>
          <span className="label">tps</span>
          <span className="value">{tps ? tps.toLocaleString() : '—'}</span>
        </span>
        <span>
          <span className="label">fps</span>
          <span className="value" id="fps-value">{'—'}</span>
        </span>
      </div>
    </header>
  );
}
