import { useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';

const MOVE_SPEED = 40;
const SHIFT_MULTIPLIER = 2.5;

export function FreeCam({
  arenaWidth,
  arenaDepth,
}: {
  arenaWidth: number;
  arenaDepth: number;
}) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const keys = useRef(new Set<string>());
  const { camera } = useThree();

  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      keys.current.add(e.key.toLowerCase());
    };
    const onUp = (e: KeyboardEvent) => {
      keys.current.delete(e.key.toLowerCase());
    };
    const onBlur = () => keys.current.clear();

    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup', onUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  useFrame((_, delta) => {
    const controls = controlsRef.current;
    if (!controls) return;

    const pressed = keys.current;
    if (pressed.size === 0) return;

    const speed = MOVE_SPEED * delta * (pressed.has('shift') ? SHIFT_MULTIPLIER : 1);

    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    forward.y = 0;
    forward.normalize();

    const right = new THREE.Vector3();
    right.crossVectors(forward, camera.up).normalize();

    const move = new THREE.Vector3();

    if (pressed.has('w')) move.add(forward);
    if (pressed.has('s')) move.sub(forward);
    if (pressed.has('d')) move.add(right);
    if (pressed.has('a')) move.sub(right);
    if (pressed.has('q') || pressed.has(' ')) move.y += 1;
    if (pressed.has('e')) move.y -= 1;

    if (move.lengthSq() === 0) return;
    move.normalize().multiplyScalar(speed);

    camera.position.add(move);
    controls.target.add(move);
  });

  return (
    <OrbitControls
      ref={controlsRef}
      target={[arenaWidth / 2, 0, arenaDepth / 2]}
      maxPolarAngle={Math.PI / 2.1}
      minDistance={5}
      maxDistance={150}
      enableDamping
      dampingFactor={0.08}
    />
  );
}
