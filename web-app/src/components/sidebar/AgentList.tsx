import { useGameStore } from '../../stores/game-store';
import { useCameraStore } from '../../stores/camera-store';
import { TEAM_COLORS, weaponFor, shortId } from '../../constants';

export function AgentList() {
  const entities = useGameStore(s => s.entities);
  const followId = useCameraStore(s => s.followId);
  const follow = useCameraStore(s => s.follow);

  const sorted = [...entities].sort((a, b) => a.team - b.team || a.id - b.id);
  const alive = entities.filter(e => !e.is_dead);

  return (
    <div className="panel" style={{ flexShrink: 0 }}>
      <div className="panel-header">
        <span className="panel-title">Agents</span>
        <span className="panel-badge">{alive.length} alive</span>
      </div>
      <ul className="entity-list">
        {sorted.map(e => {
          const col = TEAM_COLORS[e.team] || '#fff';
          const hp = Math.max(0, e.health / e.max_health) * 100;
          const hpC = hp > 50 ? 'var(--hp-good)' : hp > 25 ? 'var(--hp-mid)' : 'var(--hp-low)';
          const wep = weaponFor(e.id);
          const isDead = e.is_dead;
          const isFollowing = e.id === followId;

          return (
            <li
              key={e.id}
              className={`entity-row${isDead ? ' dead' : ''}${isFollowing ? ' following' : ''}`}
              onClick={() => follow(e.id)}
            >
              <div className="entity-team-dot" style={{ background: col }} />
              <span className="entity-id">{shortId(e.id)}</span>
              <span className="entity-weapon">{wep.slice(0, 3)}</span>
              <div className="entity-hp-bar">
                <div className="entity-hp-fill" style={{ width: `${hp}%`, background: hpC }} />
              </div>
              <span className="entity-hp-text">{isDead ? '---' : Math.round(e.health)}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
