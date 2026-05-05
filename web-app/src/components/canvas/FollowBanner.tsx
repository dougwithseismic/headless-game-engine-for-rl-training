import { useCameraStore } from '../../stores/camera-store';
import { useGameStore } from '../../stores/game-store';
import { TEAM_COLORS, weaponFor, shortId } from '../../constants';

export function FollowBanner() {
  const followId = useCameraStore(s => s.followId);
  const cinematic = useCameraStore(s => s.cinematic);
  const stopFollowing = useCameraStore(s => s.stopFollowing);
  const toggleCinematic = useCameraStore(s => s.toggleCinematic);
  const entityIdMap = useGameStore(s => s.entityIdMap);

  if (cinematic) {
    return (
      <div className="follow-banner visible">
        <span className="follow-label">cinematic</span>
        <span className="follow-name" style={{ color: '#facc15' }}>AUTO</span>
        <button className="follow-dismiss" onClick={toggleCinematic}>esc</button>
      </div>
    );
  }

  if (followId === null) return null;

  const entity = entityIdMap[followId];
  const col = entity ? TEAM_COLORS[entity.team] || '#fff' : '#fff';
  const wep = entity ? weaponFor(entity.id).toUpperCase() : '';

  return (
    <div className="follow-banner visible">
      <span className="follow-label">following</span>
      <span className="follow-name">
        <span style={{ color: col }}>{shortId(followId)}</span>{' '}
        <span style={{ color: 'var(--text-muted)', fontSize: '8px' }}>{wep}</span>
      </span>
      <button className="follow-dismiss" onClick={stopFollowing}>esc</button>
    </div>
  );
}
