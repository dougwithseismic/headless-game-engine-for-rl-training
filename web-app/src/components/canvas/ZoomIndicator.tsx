import { useCameraStore } from '../../stores/camera-store';

export function ZoomIndicator() {
  const camZoom = useCameraStore(s => s.camZoom);
  return <div className="zoom-indicator">{camZoom.toFixed(1)}x</div>;
}
