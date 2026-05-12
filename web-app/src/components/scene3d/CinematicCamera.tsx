import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface CinematicCameraProps {
  getTarget: () => [number, number, number] | undefined;
  radius?: number;
  height?: number;
  speed?: number;
  lerp?: number;
}

/**
 * Orbiting cinematic camera with smooth interpolation.
 * Must be placed inside a React Three Fiber `<Canvas>`.
 *
 * Position data flows in via `getTarget` callback -- no store coupling.
 */
export function CinematicCamera({
  getTarget,
  radius = 150,
  height = 120,
  speed = 0.15,
  lerp: lerpFactor = 0.03,
}: CinematicCameraProps) {
  const angleRef = useRef(0);
  const posRef = useRef(new THREE.Vector3());
  const targetRef = useRef(new THREE.Vector3());

  useFrame((state, delta) => {
    angleRef.current += delta * speed;
    const target = getTarget() ?? [0, 0, 0];
    const tx = target[0], ty = target[1], tz = target[2];

    const desiredPos = new THREE.Vector3(
      tx + Math.cos(angleRef.current) * radius,
      ty + height * 0.5,
      tz + Math.sin(angleRef.current) * radius,
    );
    const desiredTarget = new THREE.Vector3(tx, ty, tz);

    posRef.current.lerp(desiredPos, lerpFactor);
    targetRef.current.lerp(desiredTarget, lerpFactor);

    state.camera.position.copy(posRef.current);
    state.camera.lookAt(targetRef.current);
  });

  return null;
}
