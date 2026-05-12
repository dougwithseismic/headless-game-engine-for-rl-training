import { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useCsLiteStore, type CsAgent, type ShotEntry } from './store';
import { CinematicActionCamera } from './CinematicActionCamera';
import { FreeCam } from './FreeCam';
import { useGameConfig } from '../../hooks/use-game-config';
import { useServerHost } from '../../contexts/server';

const T_COLOR = '#f06449';
const CT_COLOR = '#4da6ff';
const T_DEAD = '#4a2020';
const CT_DEAD = '#1a2e4a';
const FLOOR_COLOR = '#1c1c30';
const WALL_COLOR = '#4a4a6a';
const WALL_TOP_COLOR = '#6a6a8a';
const SITE_A_COLOR = '#ff6b4a';
const SITE_B_COLOR = '#4da6ff';
const GRID_COLOR = '#2a2a48';

function Floor({ width, depth }: { width: number; depth: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[width / 2, 0, depth / 2]} receiveShadow>
      <planeGeometry args={[width, depth]} />
      <meshStandardMaterial color={FLOOR_COLOR} roughness={0.85} metalness={0.1} />
    </mesh>
  );
}

function GridLines({ width, depth }: { width: number; depth: number }) {
  const { minor, major } = useMemo(() => {
    const minorLines: [number, number, number][][] = [];
    const majorLines: [number, number, number][][] = [];
    const minorStep = 5;
    const majorStep = 20;
    for (let x = 0; x <= width; x += minorStep) {
      const isMajor = x % majorStep === 0;
      (isMajor ? majorLines : minorLines).push([[x, 0.01, 0], [x, 0.01, depth]]);
    }
    for (let z = 0; z <= depth; z += minorStep) {
      const isMajor = z % majorStep === 0;
      (isMajor ? majorLines : minorLines).push([[0, 0.01, z], [width, 0.01, z]]);
    }
    return { minor: minorLines, major: majorLines };
  }, [width, depth]);

  return (
    <>
      {minor.map((pts, i) => (
        <Line key={`m${i}`} points={pts} color={GRID_COLOR} lineWidth={0.5} transparent opacity={0.4} />
      ))}
      {major.map((pts, i) => (
        <Line key={`M${i}`} points={pts} color={GRID_COLOR} lineWidth={1} transparent opacity={0.7} />
      ))}
    </>
  );
}

function Obstacle({ x, y, width, height }: { x: number; y: number; width: number; height: number }) {
  const wallHeight = 3;
  return (
    <group>
      <mesh position={[x + width / 2, wallHeight / 2, y + height / 2]} castShadow receiveShadow>
        <boxGeometry args={[width, wallHeight, height]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.7} metalness={0.15} />
      </mesh>
      {/* Top edge highlight */}
      <mesh position={[x + width / 2, wallHeight + 0.03, y + height / 2]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[width, height]} />
        <meshBasicMaterial color={WALL_TOP_COLOR} transparent opacity={0.5} />
      </mesh>
    </group>
  );
}

const CONE_Y = 0.06;
const VIS_RAY_COUNT = 90;

function raycastObstacles(
  ox: number,
  oz: number,
  dx: number,
  dz: number,
  maxDist: number,
  obstacles: Array<{ x: number; y: number; width: number; height: number }>,
  arenaW?: number,
  arenaD?: number,
): number {
  let closest = maxDist;

  if (arenaW !== undefined && arenaD !== undefined) {
    if (dx > 1e-8) closest = Math.min(closest, (arenaW - ox) / dx);
    else if (dx < -1e-8) closest = Math.min(closest, -ox / dx);
    if (dz > 1e-8) closest = Math.min(closest, (arenaD - oz) / dz);
    else if (dz < -1e-8) closest = Math.min(closest, -oz / dz);
    if (closest < 0) closest = 0;
  }
  for (const obs of obstacles) {
    const minX = obs.x, maxX = obs.x + obs.width;
    const minZ = obs.y, maxZ = obs.y + obs.height;

    let tmin = 0, tmax = closest;

    if (Math.abs(dx) < 1e-8) {
      if (ox < minX || ox > maxX) continue;
    } else {
      let t1 = (minX - ox) / dx, t2 = (maxX - ox) / dx;
      if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) continue;
    }

    if (Math.abs(dz) < 1e-8) {
      if (oz < minZ || oz > maxZ) continue;
    } else {
      let t1 = (minZ - oz) / dz, t2 = (maxZ - oz) / dz;
      if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
      if (tmin > tmax) continue;
    }

    if (tmin > 0 && tmin < closest) closest = tmin;
  }
  return closest;
}

function FogOverlay({ width, depth }: { width: number; depth: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[width / 2, 0.03, depth / 2]}>
      <planeGeometry args={[width, depth]} />
      <meshBasicMaterial color="#0a0a18" transparent opacity={0.15} depthWrite={false} />
    </mesh>
  );
}

function VisionCone({
  agent,
  fov,
  maxRange,
  obstacles,
  arenaWidth,
  arenaDepth,
}: {
  agent: CsAgent;
  fov: number;
  maxRange: number;
  obstacles: Array<{ x: number; y: number; width: number; height: number }>;
  arenaWidth: number;
  arenaDepth: number;
}) {
  const isT = agent.team === 0;
  const teamColor = isT ? T_COLOR : CT_COLOR;
  const color = useMemo(() => new THREE.Color(teamColor), [teamColor]);

  const fanGeo = useMemo(() => {
    const edgeCount = VIS_RAY_COUNT;
    const ringCount = 8;
    const vertCount = 1 + edgeCount * ringCount;
    const positions = new Float32Array(vertCount * 3);
    const alphas = new Float32Array(vertCount);
    const colors = new Float32Array(vertCount * 3);

    // Center vertex: bright, full alpha
    positions[0] = agent.x;
    positions[1] = CONE_Y;
    positions[2] = agent.z;
    colors[0] = color.r;
    colors[1] = color.g;
    colors[2] = color.b;
    alphas[0] = 1.0;

    const rayDists: number[] = [];
    for (let i = 0; i < edgeCount; i++) {
      const t = i / (edgeCount - 1);
      const rayYaw = agent.yaw + fov * (t - 0.5);
      const dx = Math.cos(rayYaw);
      const dz = Math.sin(rayYaw);
      rayDists.push(raycastObstacles(agent.x, agent.z, dx, dz, maxRange, obstacles, arenaWidth, arenaDepth));
    }

    for (let r = 0; r < ringCount; r++) {
      const ringFrac = (r + 1) / ringCount;
      for (let i = 0; i < edgeCount; i++) {
        const t = i / (edgeCount - 1);
        const rayYaw = agent.yaw + fov * (t - 0.5);
        const dx = Math.cos(rayYaw);
        const dz = Math.sin(rayYaw);
        const dist = rayDists[i] * ringFrac;
        const vi = 1 + r * edgeCount + i;

        positions[vi * 3] = agent.x + dx * dist;
        positions[vi * 3 + 1] = CONE_Y;
        positions[vi * 3 + 2] = agent.z + dz * dist;

        const distFrac = dist / maxRange;
        const falloff = 1.0 - distFrac * distFrac;
        const angularFade = 1.0 - 0.3 * Math.pow(Math.abs(t - 0.5) * 2, 2);
        const a = falloff * angularFade;
        alphas[vi] = a;
        colors[vi * 3] = color.r;
        colors[vi * 3 + 1] = color.g;
        colors[vi * 3 + 2] = color.b;
      }
    }

    const indices: number[] = [];
    // Center to first ring
    for (let i = 0; i < edgeCount - 1; i++) {
      indices.push(0, 1 + i, 1 + i + 1);
    }
    // Ring to ring
    for (let r = 0; r < ringCount - 1; r++) {
      const base = 1 + r * edgeCount;
      const next = 1 + (r + 1) * edgeCount;
      for (let i = 0; i < edgeCount - 1; i++) {
        indices.push(base + i, next + i, next + i + 1);
        indices.push(base + i, next + i + 1, base + i + 1);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    return geo;
  }, [agent.x, agent.z, agent.yaw, fov, maxRange, obstacles, color, arenaWidth, arenaDepth]);

  const shaderMat = useMemo(() => new THREE.ShaderMaterial({
    vertexShader: `
      attribute float alpha;
      varying float vAlpha;
      varying vec3 vColor;
      void main() {
        vAlpha = alpha;
        vColor = color;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying float vAlpha;
      varying vec3 vColor;
      void main() {
        gl_FragColor = vec4(vColor, vAlpha * 0.25);
      }
    `,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    vertexColors: true,
  }), []);

  return (
    <mesh geometry={fanGeo} material={shaderMat} />
  );
}

function BombSite({ center, radius, label, color }: { center: [number, number, number]; radius: number; label: string; color: string }) {
  const rings = useMemo(() => {
    const count = 5;
    return Array.from({ length: count }, (_, i) => {
      const frac = (i + 1) / count;
      const r = radius * frac;
      const opacity = 0.12 * (1 - frac * 0.7);
      return { innerRadius: r - radius / count, outerRadius: r, opacity };
    });
  }, [radius]);

  return (
    <group>
      {/* Gradient fill rings - inner bright, outer faded */}
      {rings.map((ring, i) => (
        <mesh key={i} rotation={[-Math.PI / 2, 0, 0]} position={[center[0], 0.02, center[2]]}>
          <ringGeometry args={[ring.innerRadius, ring.outerRadius, 48]} />
          <meshBasicMaterial color={color} transparent opacity={ring.opacity} depthWrite={false} />
        </mesh>
      ))}
      {/* Outer ring border */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[center[0], 0.025, center[2]]}>
        <ringGeometry args={[radius - 0.15, radius, 48]} />
        <meshBasicMaterial color={color} transparent opacity={0.35} depthWrite={false} />
      </mesh>
      {/* Inner glow */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[center[0], 0.015, center[2]]}>
        <circleGeometry args={[radius * 0.25, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.08} depthWrite={false} />
      </mesh>
      {/* Dashed ring at ~60% radius */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[center[0], 0.028, center[2]]}>
        <ringGeometry args={[radius * 0.58, radius * 0.62, 48]} />
        <meshBasicMaterial color={color} transparent opacity={0.15} depthWrite={false} />
      </mesh>
      <Text
        position={[center[0], 0.05, center[2]]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={2.5}
        color={color}
        anchorX="center"
        anchorY="middle"
        font={undefined}
        fillOpacity={0.6}
      >
        {label}
      </Text>
    </group>
  );
}

function AgentModel({ agent }: { agent: CsAgent }) {
  const meshRef = useRef<THREE.Group>(null);
  const muzzleRef = useRef<THREE.Mesh>(null);
  const isT = agent.team === 0;
  const baseColor = agent.isDead ? (isT ? T_DEAD : CT_DEAD) : (isT ? T_COLOR : CT_COLOR);
  const bodyHeight = 1.8;
  const bodyRadius = 0.4;

  useFrame(() => {
    if (muzzleRef.current) {
      muzzleRef.current.visible = agent.shooting && !agent.isDead;
    }
  });

  if (agent.isDead) {
    return (
      <group position={[agent.x, 0.1, agent.z]}>
        <mesh rotation={[0, 0, Math.PI / 2]}>
          <capsuleGeometry args={[bodyRadius * 0.6, bodyHeight * 0.5, 4, 8]} />
          <meshStandardMaterial color={baseColor} transparent opacity={0.4} />
        </mesh>
      </group>
    );
  }

  const hpFrac = agent.health / agent.maxHealth;
  const hpColor = hpFrac > 0.5 ? '#2ecc71' : hpFrac > 0.25 ? '#f39c12' : '#e74c3c';

  return (
    <group ref={meshRef} position={[agent.x, 0, agent.z]}>
      {/* Body capsule */}
      <mesh position={[0, bodyHeight / 2 + bodyRadius, 0]} castShadow>
        <capsuleGeometry args={[bodyRadius, bodyHeight - bodyRadius * 2, 4, 8]} />
        <meshStandardMaterial color={baseColor} emissive={baseColor} emissiveIntensity={0.15} roughness={0.6} />
      </mesh>

      {/* Head */}
      <mesh position={[0, bodyHeight + bodyRadius * 0.5, 0]} castShadow>
        <sphereGeometry args={[bodyRadius * 0.45, 8, 8]} />
        <meshStandardMaterial color={baseColor} emissive={baseColor} emissiveIntensity={0.15} roughness={0.6} />
      </mesh>

      {/* HP ring on ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
        <ringGeometry args={[0.8, 1.0, 32, 1, 0, Math.PI * 2 * hpFrac]} />
        <meshBasicMaterial color={hpColor} side={THREE.DoubleSide} />
      </mesh>

      {/* Facing direction indicator */}
      <mesh
        position={[
          Math.cos(agent.yaw) * 1.5,
          bodyHeight * 0.6,
          Math.sin(agent.yaw) * 1.5,
        ]}
      >
        <sphereGeometry args={[0.12, 6, 6]} />
        <meshBasicMaterial color={baseColor} />
      </mesh>

      {/* Gun barrel */}
      <mesh
        position={[
          Math.cos(agent.yaw) * 1.0,
          bodyHeight * 0.65,
          Math.sin(agent.yaw) * 1.0,
        ]}
        rotation={[0, -agent.yaw + Math.PI / 2, 0]}
      >
        <cylinderGeometry args={[0.04, 0.04, 0.8, 4]} />
        <meshStandardMaterial color="#666" />
      </mesh>

      {/* Muzzle flash */}
      <mesh
        ref={muzzleRef}
        position={[
          Math.cos(agent.yaw) * 1.8,
          bodyHeight * 0.65,
          Math.sin(agent.yaw) * 1.8,
        ]}
        visible={false}
      >
        <sphereGeometry args={[0.25, 6, 6]} />
        <meshBasicMaterial color="#ffcc00" />
      </mesh>

      {/* Team indicator ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.04, 0]}>
        <ringGeometry args={[1.0, 1.15, 32]} />
        <meshBasicMaterial color={baseColor} transparent opacity={0.5} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

function XRayOutline({ agent }: { agent: CsAgent }) {
  const isT = agent.team === 0;
  const teamColor = isT ? T_COLOR : CT_COLOR;
  const color = useMemo(() => new THREE.Color(teamColor), [teamColor]);
  const bodyHeight = 1.8;
  const bodyRadius = 0.4;
  const s = 1.03;

  const material = useMemo(() => new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: color },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vViewDir;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vViewDir = normalize(-mvPosition.xyz);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      void main() {
        float rim = 1.0 - abs(dot(vNormal, vViewDir));
        float edge = pow(rim, 2.0);
        float fill = 0.18;
        float alpha = fill + edge * 0.65;
        gl_FragColor = vec4(uColor, alpha);
      }
    `,
    transparent: true,
    depthTest: true,
    depthFunc: THREE.GreaterDepth,
    depthWrite: false,
    side: THREE.FrontSide,
    blending: THREE.AdditiveBlending,
  }), [color]);

  return (
    <group position={[agent.x, 0, agent.z]} renderOrder={999}>
      <mesh position={[0, bodyHeight / 2 + bodyRadius, 0]} material={material} renderOrder={999}>
        <capsuleGeometry args={[bodyRadius * s, (bodyHeight - bodyRadius * 2) * s, 8, 16]} />
      </mesh>
      <mesh position={[0, bodyHeight + bodyRadius * 0.5, 0]} material={material} renderOrder={999}>
        <sphereGeometry args={[bodyRadius * 0.45 * s, 12, 12]} />
      </mesh>
    </group>
  );
}

const SHOT_LIFETIME_MS = 300;

function ShotTracer({ shot }: { shot: ShotEntry }) {
  const [opacity, setOpacity] = useState(1);
  const len = shot.hit ? 8 : 4;
  const eyeHeight = 1.6;

  useFrame(() => {
    const age = Date.now() - shot.createdAt;
    const t = Math.max(0, 1 - age / SHOT_LIFETIME_MS);
    setOpacity(t);
  });

  if (opacity <= 0) return null;

  const baseOpacity = shot.hit ? 0.8 : 0.3;
  return (
    <Line
      points={[
        [shot.ox, eyeHeight, shot.oz],
        [shot.ox + shot.dx * len, eyeHeight, shot.oz + shot.dz * len],
      ]}
      color={shot.hit ? '#ff4444' : '#ffcc33'}
      lineWidth={shot.hit ? 2 : 1}
      transparent
      opacity={baseOpacity * opacity}
    />
  );
}

function Scene() {
  const host = useServerHost();
  const { data: config } = useGameConfig(host);
  const agents = useCsLiteStore((s) => s.agents);
  const obstacles = useCsLiteStore((s) => s.obstacles);
  const shots = useCsLiteStore((s) => s.shots);
  const cameraMode = useCsLiteStore((s) => s.cameraMode);
  const xray = useCsLiteStore((s) => s.xray);

  const arenaW = (config?.extra?.arena_width as number) ?? 80;
  const arenaD = (config?.extra?.arena_depth as number) ?? 60;
  const fov = (config?.extra?.ray_h_fov as number) ?? 2.094;
  const rayMaxRange = (config?.extra?.ray_max_range as number) ?? 60;

  const siteACenter: [number, number, number] = [arenaW * 0.25, 0, arenaD * 0.75];
  const siteBCenter: [number, number, number] = [arenaW * 0.75, 0, arenaD * 0.75];

  return (
    <>
      <fog attach="fog" args={['#0c0c1a', 60, 180]} />
      <ambientLight intensity={0.5} />
      <hemisphereLight args={['#334', '#121218', 0.4]} />
      <directionalLight position={[arenaW / 2, 40, arenaD / 2]} intensity={0.7} castShadow />
      <pointLight position={[arenaW * 0.25, 6, arenaD * 0.75]} intensity={0.5} color={SITE_A_COLOR} distance={20} />
      <pointLight position={[arenaW * 0.75, 6, arenaD * 0.75]} intensity={0.5} color={SITE_B_COLOR} distance={20} />

      <Floor width={arenaW} depth={arenaD} />
      <GridLines width={arenaW} depth={arenaD} />
      <FogOverlay width={arenaW} depth={arenaD} />

      {/* Arena boundary walls */}
      {[
        { pos: [arenaW / 2, 1.5, -0.15] as const, size: [arenaW, 3, 0.3] as const },
        { pos: [arenaW / 2, 1.5, arenaD + 0.15] as const, size: [arenaW, 3, 0.3] as const },
        { pos: [-0.15, 1.5, arenaD / 2] as const, size: [0.3, 3, arenaD] as const },
        { pos: [arenaW + 0.15, 1.5, arenaD / 2] as const, size: [0.3, 3, arenaD] as const },
      ].map((w, i) => (
        <mesh key={`wall${i}`} position={[w.pos[0], w.pos[1], w.pos[2]]}>
          <boxGeometry args={[w.size[0], w.size[1], w.size[2]]} />
          <meshStandardMaterial color="#2a2a40" transparent opacity={0.4} />
        </mesh>
      ))}

      {obstacles.map((obs, i) => (
        <Obstacle key={i} x={obs.x} y={obs.y} width={obs.width} height={obs.height} />
      ))}

      <BombSite center={siteACenter} radius={6} label="A" color={SITE_A_COLOR} />
      <BombSite center={siteBCenter} radius={6} label="B" color={SITE_B_COLOR} />

      {[...agents.values()].map((agent) => (
        <AgentModel key={agent.id} agent={agent} />
      ))}

      {xray && [...agents.values()]
        .filter((a) => !a.isDead)
        .map((agent) => (
          <XRayOutline key={`xray-${agent.id}`} agent={agent} />
        ))}

      {[...agents.values()]
        .filter((a) => !a.isDead)
        .map((agent) => (
          <VisionCone key={`fov-${agent.id}`} agent={agent} fov={fov} maxRange={rayMaxRange} obstacles={obstacles} arenaWidth={arenaW} arenaDepth={arenaD} />
        ))}

      {shots
        .filter((s) => Date.now() - s.createdAt < SHOT_LIFETIME_MS)
        .map((shot, i) => (
          <ShotTracer key={`${shot.createdAt}-${i}`} shot={shot} />
        ))}

      {cameraMode === 'cinematic' ? (
        <CinematicActionCamera arenaWidth={arenaW} arenaDepth={arenaD} />
      ) : (
        <FreeCam arenaWidth={arenaW} arenaDepth={arenaD} />
      )}
    </>
  );
}

export function CsLite3DCanvas() {
  const host = useServerHost();
  const { data: config } = useGameConfig(host);
  const arenaW = (config?.extra?.arena_width as number) ?? 80;
  const arenaD = (config?.extra?.arena_depth as number) ?? 60;

  return (
    <Canvas
      shadows
      gl={{ antialias: true }}
      camera={{
        position: [arenaW / 2, 80, arenaD / 2 + 5],
        fov: 50,
        near: 0.1,
        far: 500,
      }}
      style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
    >
      <color attach="background" args={['#0c0c1a']} />
      <Scene />
    </Canvas>
  );
}
