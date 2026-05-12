import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface ObjectTrailProps {
  /** Callback returning current position of the object to trail. */
  getPosition: () => [number, number, number] | undefined;
  /** Number of trail segments. */
  length?: number;
  /** Trail color (CSS color string). */
  color?: string;
  /** Maximum opacity at the head of the trail. */
  opacity?: number;
}

/**
 * Fading trail rendered behind any moving 3D object.
 * Uses a line with per-vertex RGBA colors for the fade effect.
 *
 * Position data flows in via `getPosition` callback -- no store coupling.
 */
export function ObjectTrail({
  getPosition,
  length = 80,
  color = '#00ffcc',
  opacity = 0.4,
}: ObjectTrailProps) {
  const geomRef = useRef<THREE.BufferGeometry>(null);
  const positionsRef = useRef(new Float32Array(length * 3));
  const colorsRef = useRef(new Float32Array(length * 4));
  const countRef = useRef(0);
  const baseColor = useRef(new THREE.Color(color));

  useFrame(() => {
    const pos = getPosition();
    if (!pos || !geomRef.current) return;

    const positions = positionsRef.current;
    const colors = colorsRef.current;

    // Shift all positions back by 1 slot (newest at index 0)
    positions.copyWithin(3, 0, (length - 1) * 3);
    positions[0] = pos[0];
    positions[1] = pos[1];
    positions[2] = pos[2];

    if (countRef.current < length) countRef.current++;

    // Update vertex colors with fading alpha
    const c = baseColor.current;
    for (let i = 0; i < countRef.current; i++) {
      const alpha = (1 - i / length) * opacity;
      colors[i * 4] = c.r;
      colors[i * 4 + 1] = c.g;
      colors[i * 4 + 2] = c.b;
      colors[i * 4 + 3] = alpha;
    }

    geomRef.current.setAttribute(
      'position',
      new THREE.BufferAttribute(positions.slice(0, countRef.current * 3), 3),
    );
    geomRef.current.setAttribute(
      'color',
      new THREE.BufferAttribute(colors.slice(0, countRef.current * 4), 4),
    );
    geomRef.current.computeBoundingSphere();
  });

  return (
    <line>
      <bufferGeometry ref={geomRef} />
      <lineBasicMaterial vertexColors transparent depthWrite={false} />
    </line>
  );
}
