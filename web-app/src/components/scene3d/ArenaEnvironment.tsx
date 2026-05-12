import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Grid, Line } from '@react-three/drei';
import * as THREE from 'three';

interface ArenaEnvironmentProps {
  size: [number, number, number];
  getAgentPosition?: () => [number, number, number] | undefined;
  warnDistance?: number;
  showCeiling?: boolean;
  gridCellSize?: number;
  gridSectionSize?: number;
}

interface WallDef {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: [number, number, number];
  distance: (pos: [number, number, number]) => number;
}

function LandingPad({ position, size }: { position: [number, number, number]; size: number }) {
  const segments = 48;
  const r = size;

  const outerRing = useMemo(() => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      pts.push([Math.cos(a) * r, 0, Math.sin(a) * r]);
    }
    return pts;
  }, [r]);

  const innerRing = useMemo(() => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i <= segments; i++) {
      const a = (i / segments) * Math.PI * 2;
      pts.push([Math.cos(a) * r * 0.4, 0, Math.sin(a) * r * 0.4]);
    }
    return pts;
  }, [r]);

  const crossSize = r * 0.25;

  return (
    <group position={position}>
      <Line points={outerRing} color="#5a6070" lineWidth={1.5} />
      <Line points={innerRing} color="#4a5060" lineWidth={1} />
      {/* Crosshairs */}
      <Line points={[[-crossSize, 0, 0], [crossSize, 0, 0]]} color="#5a6070" lineWidth={1} />
      <Line points={[[0, 0, -crossSize], [0, 0, crossSize]]} color="#5a6070" lineWidth={1} />
      {/* Corner tick marks on outer ring */}
      {[0, Math.PI / 2, Math.PI, Math.PI * 1.5].map((a, i) => {
        const x = Math.cos(a) * r;
        const z = Math.sin(a) * r;
        const dx = Math.cos(a) * r * 0.15;
        const dz = Math.sin(a) * r * 0.15;
        return (
          <Line
            key={i}
            points={[[x - dx, 0, z - dz], [x + dx, 0, z + dz]]}
            color="#6a7080"
            lineWidth={1}
          />
        );
      })}
    </group>
  );
}

function SafetyStripes({ width, depth }: { width: number; depth: number }) {
  const stripeWidth = 1.5;
  const stripeColor = '#8a7030';

  return (
    <group position={[0, 0.02, 0]}>
      {/* Floor edge stripes — warning markings along arena boundary */}
      <mesh position={[width / 2, 0, stripeWidth / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width, stripeWidth]} />
        <meshStandardMaterial color={stripeColor} roughness={0.9} transparent opacity={0.3} />
      </mesh>
      <mesh position={[width / 2, 0, depth - stripeWidth / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width, stripeWidth]} />
        <meshStandardMaterial color={stripeColor} roughness={0.9} transparent opacity={0.3} />
      </mesh>
      <mesh position={[stripeWidth / 2, 0, depth / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[stripeWidth, depth]} />
        <meshStandardMaterial color={stripeColor} roughness={0.9} transparent opacity={0.3} />
      </mesh>
      <mesh position={[width - stripeWidth / 2, 0, depth / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[stripeWidth, depth]} />
        <meshStandardMaterial color={stripeColor} roughness={0.9} transparent opacity={0.3} />
      </mesh>
    </group>
  );
}

function AltitudeRings({ center, maxHeight, count = 4 }: {
  center: [number, number];
  maxHeight: number;
  count?: number;
}) {
  const segments = 64;
  const radius = 6;

  const rings = useMemo(() => {
    const result: { height: number; points: [number, number, number][] }[] = [];
    for (let i = 1; i <= count; i++) {
      const h = (i / count) * maxHeight * 0.8;
      const pts: [number, number, number][] = [];
      for (let j = 0; j <= segments; j++) {
        const a = (j / segments) * Math.PI * 2;
        pts.push([Math.cos(a) * radius, 0, Math.sin(a) * radius]);
      }
      result.push({ height: h, points: pts });
    }
    return result;
  }, [maxHeight, count]);

  return (
    <group position={[center[0], 0, center[1]]}>
      {rings.map((ring, i) => {
        const t = (i + 1) / count;
        const opacity = 0.15 + t * 0.15;
        return (
          <group key={i} position={[0, ring.height, 0]}>
            <Line
              points={ring.points}
              color="#4060a0"
              lineWidth={0.8}
              transparent
              opacity={opacity}
            />
            {/* Small tick marks at cardinal directions */}
            {[0, Math.PI / 2, Math.PI, Math.PI * 1.5].map((a, j) => (
              <Line
                key={j}
                points={[
                  [Math.cos(a) * (radius - 0.8), 0, Math.sin(a) * (radius - 0.8)],
                  [Math.cos(a) * (radius + 0.8), 0, Math.sin(a) * (radius + 0.8)],
                ]}
                color="#4060a0"
                lineWidth={0.6}
                transparent
                opacity={opacity}
              />
            ))}
          </group>
        );
      })}

      {/* Vertical reference line through center */}
      <Line
        points={[[0, 0, 0], [0, maxHeight * 0.85, 0]]}
        color="#303850"
        lineWidth={0.4}
        transparent
        opacity={0.2}
        dashed
        dashSize={2}
        gapSize={2}
      />
    </group>
  );
}

export function ArenaEnvironment({
  size,
  getAgentPosition,
  warnDistance = 40,
  showCeiling = false,
  gridCellSize = 10,
  gridSectionSize = 50,
}: ArenaEnvironmentProps) {
  const [w, h, d] = size;

  const wallMatRefs = useRef<(THREE.MeshStandardMaterial | null)[]>([]);

  const walls: WallDef[] = useMemo(() => {
    const defs: WallDef[] = [
      { position: [w / 2, h / 2, 0], rotation: [0, 0, 0], scale: [w, h, 1], distance: (pos) => pos[2] },
      { position: [w / 2, h / 2, d], rotation: [0, 0, 0], scale: [w, h, 1], distance: (pos) => d - pos[2] },
      { position: [0, h / 2, d / 2], rotation: [0, Math.PI / 2, 0], scale: [d, h, 1], distance: (pos) => pos[0] },
      { position: [w, h / 2, d / 2], rotation: [0, Math.PI / 2, 0], scale: [d, h, 1], distance: (pos) => w - pos[0] },
    ];
    if (showCeiling) {
      defs.push({ position: [w / 2, h, d / 2], rotation: [Math.PI / 2, 0, 0], scale: [w, d, 1], distance: (pos) => h - pos[1] });
    }
    return defs;
  }, [w, h, d, showCeiling]);

  useFrame(() => {
    if (!getAgentPosition) return;
    const pos = getAgentPosition();
    if (!pos) return;

    for (let i = 0; i < walls.length; i++) {
      const mat = wallMatRefs.current[i];
      if (!mat) continue;

      const dist = walls[i].distance(pos);
      const t = Math.max(0, 1 - dist / warnDistance);
      const tSq = t * t;

      mat.opacity = 0.02 + tSq * 0.25;
      mat.emissiveIntensity = tSq * 1.5;
      mat.color.setRGB(0.35 + tSq * 0.55, 0.35 * (1 - tSq * 0.5), 0.2 * (1 - tSq));
      mat.emissive.setRGB(tSq * 0.7, tSq * 0.15, 0);
    }
  });

  return (
    <>
      {/* Concrete floor */}
      <mesh position={[w / 2, -0.01, d / 2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[w, d]} />
        <meshStandardMaterial color="#2e3038" roughness={0.92} metalness={0.02} />
      </mesh>

      {/* Grid — warehouse floor markings */}
      <Grid
        args={[w, d]}
        cellSize={gridCellSize}
        cellThickness={0.5}
        cellColor="#3a3e48"
        sectionSize={gridSectionSize}
        sectionThickness={1.0}
        sectionColor="#4a5575"
        fadeDistance={400}
        fadeStrength={1}
        position={[w / 2, 0.01, d / 2]}
        infiniteGrid
      />

      {/* Safety edge stripes */}
      <SafetyStripes width={w} depth={d} />

      {/* Center landing pad */}
      <LandingPad position={[w / 2, 0.02, d / 2]} size={Math.min(w, d) * 0.12} />

      {/* Altitude reference rings */}
      <AltitudeRings center={[w / 2, d / 2]} maxHeight={h} count={4} />

      {/* Corner overhead lights */}
      {[
        [w * 0.15, h * 0.9, d * 0.15],
        [w * 0.85, h * 0.9, d * 0.15],
        [w * 0.15, h * 0.9, d * 0.85],
        [w * 0.85, h * 0.9, d * 0.85],
      ].map((pos, i) => (
        <pointLight key={i} position={pos as [number, number, number]} color="#ffe8cc" intensity={0.3} distance={h * 1.2} decay={2} />
      ))}

      {/* Boundary walls */}
      {walls.map((wall, i) => (
        <mesh key={i} position={wall.position} rotation={wall.rotation}>
          <planeGeometry args={[wall.scale[0], wall.scale[1]]} />
          <meshStandardMaterial
            ref={(el) => { wallMatRefs.current[i] = el; }}
            color="#555555"
            emissive="#000000"
            emissiveIntensity={0}
            transparent
            opacity={0.02}
            depthWrite={false}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </>
  );
}
