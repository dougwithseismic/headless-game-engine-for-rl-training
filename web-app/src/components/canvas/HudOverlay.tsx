import { useGameConfig } from '../../hooks/use-game-config';
import { useCameraStore } from '../../stores/camera-store';

export function HudOverlay() {
  const { data: config } = useGameConfig();
  const camZoom = useCameraStore(s => s.camZoom);
  const camX = useCameraStore(s => s.camX);
  const camY = useCameraStore(s => s.camY);

  return (
    <div className="hud-overlay">
      <div><span className="hl">SYS</span> ghostlobby::engine v0.1.0</div>
      <div><span className="hl">CFG</span> {config?.scenario || config?.title || '—'}</div>
      <div><span className="hl">MAP</span> {config ? `${config.arena.width}x${config.arena.height}` : '—'}</div>
      <div><span className="hl">CAM</span> {camZoom.toFixed(1)}x @ {Math.round(camX)},{Math.round(camY)}</div>
      <div className="hud-hint">scroll=zoom · click=follow · right-drag=pan · esc=reset</div>
    </div>
  );
}
