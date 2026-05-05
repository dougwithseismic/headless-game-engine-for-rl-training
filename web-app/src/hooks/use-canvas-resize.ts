import { useEffect } from 'react';
import type { RefObject } from 'react';

export function useCanvasResize(
  wrapRef: RefObject<HTMLDivElement | null>,
  canvasRef: RefObject<HTMLCanvasElement | null>,
  fogCanvasRef: RefObject<HTMLCanvasElement | null>,
  arenaW: number,
  arenaH: number,
) {
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;

    const resize = () => {
      const canvas = canvasRef.current;
      const fog = fogCanvasRef.current;
      if (!canvas) return;

      const mW = wrap.clientWidth - 32;
      const mH = wrap.clientHeight - 32;
      const aspect = arenaW / arenaH;
      let w: number, h: number;
      if (mW / mH > aspect) {
        h = mH; w = h * aspect;
      } else {
        w = mW; h = w / aspect;
      }
      canvas.width = Math.floor(w);
      canvas.height = Math.floor(h);
      if (fog) {
        fog.width = canvas.width;
        fog.height = canvas.height;
        fog.style.width = canvas.width + 'px';
        fog.style.height = canvas.height + 'px';
      }
    };

    const observer = new ResizeObserver(resize);
    observer.observe(wrap);
    resize();

    return () => observer.disconnect();
  }, [wrapRef, canvasRef, fogCanvasRef, arenaW, arenaH]);
}
