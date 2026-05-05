import { useGameStore } from '../../stores/game-store';
import { shortId } from '../../constants';

export function KillFeed() {
  const kills = useGameStore(s => s.kills);

  return (
    <div className="panel" style={{ flexShrink: 0, maxHeight: 160, overflow: 'hidden' }}>
      <div className="panel-header">
        <span className="panel-title">Kill Feed</span>
      </div>
      <ul className="kill-feed">
        {kills.map((k, i) => (
          <li key={`${k.tick}-${i}`} className="kill-entry">
            <span className="kill-tick">{String(k.tick).padStart(6, '0')}</span>
            <span style={{ color: k.kCol, fontWeight: 500 }}>{shortId(k.killerId)}</span>
            <span className="kill-weapon">{k.wep.slice(0, 3)}</span>
            <span style={{ color: k.vCol, fontWeight: 500 }}>{shortId(k.victimId)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
