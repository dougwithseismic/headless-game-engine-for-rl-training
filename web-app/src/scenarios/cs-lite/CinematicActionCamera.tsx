import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useCsLiteStore } from './store';

const MIN_RADIUS = 15;
const MAX_RADIUS = 60;
const MIN_HEIGHT = 12;
const MAX_HEIGHT = 50;
const PADDING = 8;
const ORBIT_SPEED = 0.12;
const POSITION_LERP = 0.025;
const ZOOM_LERP = 0.008;
const TARGET_LERP = 0.02;

export function CinematicActionCamera({
  arenaWidth,
  arenaDepth,
}: {
  arenaWidth: number;
  arenaDepth: number;
}) {
  const angleRef = useRef(0);
  const posRef = useRef(new THREE.Vector3(arenaWidth / 2, 40, arenaDepth / 2 + 30));
  const targetRef = useRef(new THREE.Vector3(arenaWidth / 2, 0, arenaDepth / 2));
  const radiusRef = useRef(40);
  const heightRef = useRef(30);

  useFrame((state, delta) => {
    const agents = useCsLiteStore.getState().agents;
    const alive: { x: number; z: number }[] = [];
    agents.forEach((a) => {
      if (!a.isDead) alive.push({ x: a.x, z: a.z });
    });

    let cx: number, cz: number, spread: number;

    if (alive.length === 0) {
      cx = arenaWidth / 2;
      cz = arenaDepth / 2;
      spread = Math.max(arenaWidth, arenaDepth) * 0.4;
    } else if (alive.length === 1) {
      cx = alive[0].x;
      cz = alive[0].z;
      spread = 10;
    } else {
      let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
      for (const a of alive) {
        if (a.x < minX) minX = a.x;
        if (a.x > maxX) maxX = a.x;
        if (a.z < minZ) minZ = a.z;
        if (a.z > maxZ) maxZ = a.z;
      }
      cx = (minX + maxX) / 2;
      cz = (minZ + maxZ) / 2;
      const dx = maxX - minX + PADDING * 2;
      const dz = maxZ - minZ + PADDING * 2;
      spread = Math.sqrt(dx * dx + dz * dz) / 2;
    }

    const desiredRadius = Math.min(MAX_RADIUS, Math.max(MIN_RADIUS, spread * 1.2));
    const desiredHeight = Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, spread * 0.7));
    radiusRef.current += (desiredRadius - radiusRef.current) * ZOOM_LERP;
    heightRef.current += (desiredHeight - heightRef.current) * ZOOM_LERP;

    angleRef.current += delta * ORBIT_SPEED;

    const desiredPos = new THREE.Vector3(
      cx + Math.cos(angleRef.current) * radiusRef.current,
      heightRef.current,
      cz + Math.sin(angleRef.current) * radiusRef.current,
    );
    const desiredTarget = new THREE.Vector3(cx, 1, cz);

    posRef.current.lerp(desiredPos, POSITION_LERP);
    targetRef.current.lerp(desiredTarget, TARGET_LERP);

    state.camera.position.copy(posRef.current);
    state.camera.lookAt(targetRef.current);
  });

  return null;
}
