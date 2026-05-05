import type { RefObject } from 'react';

export function Minimap({ mmRef }: { mmRef: RefObject<HTMLCanvasElement | null> }) {
  return (
    <div className="minimap-container">
      <span className="minimap-label">minimap</span>
      <canvas ref={mmRef} width={130} height={130} />
    </div>
  );
}
