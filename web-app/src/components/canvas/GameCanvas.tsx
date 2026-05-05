import { useRef } from 'react';
import { useCanvasRenderer } from '../../hooks/use-canvas-renderer';
import { useCanvasResize } from '../../hooks/use-canvas-resize';
import { useCameraControls } from '../../hooks/use-camera-controls';
import { useWebSocket } from '../../hooks/use-websocket';
import { useRenderStore } from '../../stores/render-store';
import { createEffectsState } from '../../types/effects';
import { initAmbient } from '../../renderer/draw-effects';
import { HudOverlay } from './HudOverlay';
import { CanvasControls } from './CanvasControls';
import { Minimap } from './Minimap';
import { FollowBanner } from './FollowBanner';
import { ZoomIndicator } from './ZoomIndicator';
import { EntityTooltip } from './EntityTooltip';
import { useEffect } from 'react';
import { useCameraStore } from '../../stores/camera-store';

interface GameCanvasProps {
  arenaW: number;
  arenaH: number;
  obstacles: Array<{ x: number; y: number; width: number; height: number }>;
}

export function GameCanvas({ arenaW, arenaH, obstacles }: GameCanvasProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fogCanvasRef = useRef<HTMLCanvasElement>(null);
  const mmCanvasRef = useRef<HTMLCanvasElement>(null);
  const effectsRef = useRef(createEffectsState());
  const fog = useRenderStore(s => s.fog);

  useEffect(() => {
    initAmbient(effectsRef.current, arenaW, arenaH);
    useCameraStore.getState().reset(arenaW, arenaH);
  }, [arenaW, arenaH]);

  useCanvasResize(wrapRef, canvasRef, fogCanvasRef, arenaW, arenaH);
  useCameraControls(canvasRef, effectsRef, arenaW, arenaH);
  useWebSocket(effectsRef);
  useCanvasRenderer(canvasRef, fogCanvasRef, mmCanvasRef, effectsRef, arenaW, arenaH, obstacles);

  return (
    <div className="canvas-wrap" ref={wrapRef}>
      <canvas id="arena" ref={canvasRef} style={{ cursor: 'crosshair' }} />
      <canvas
        id="fog"
        ref={fogCanvasRef}
        style={{
          position: 'absolute',
          pointerEvents: 'none',
          zIndex: 1,
          borderRadius: '2px',
          display: fog ? '' : 'none',
        }}
      />
      <div className="scanlines" />
      <div className="vignette" />
      <HudOverlay />
      <CanvasControls />
      <Minimap mmRef={mmCanvasRef} />
      <FollowBanner />
      <ZoomIndicator />
      <EntityTooltip effectsRef={effectsRef} canvasRef={canvasRef} />
    </div>
  );
}
