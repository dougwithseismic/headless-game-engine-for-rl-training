import { useRenderStore } from '../../stores/render-store';
import { useCameraStore } from '../../stores/camera-store';

const TOGGLES = ['fog', 'glow', 'grid', 'trails'] as const;

export function CanvasControls() {
  const store = useRenderStore();
  const cinematic = useCameraStore(s => s.cinematic);
  const toggleCinematic = useCameraStore(s => s.toggleCinematic);

  return (
    <div className="controls">
      {TOGGLES.map(key => (
        <button
          key={key}
          className={`ctrl-btn${store[key] ? ' active' : ''}`}
          onClick={() => store.toggle(key)}
        >
          {key}
        </button>
      ))}
      <button
        className={`ctrl-btn${cinematic ? ' active' : ''}`}
        onClick={toggleCinematic}
        title="Cinematic camera (C)"
      >
        cine
      </button>
    </div>
  );
}
