import { useEffect } from 'react';
import { useGameConfig } from './hooks/use-game-config';
import { useGameStore } from './stores/game-store';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { GameCanvas } from './components/canvas/GameCanvas';
import { Scoreboard } from './components/sidebar/Scoreboard';
import { AgentList } from './components/sidebar/AgentList';
import { KillFeed } from './components/sidebar/KillFeed';
import { Terminal } from './components/sidebar/Terminal';

export default function App() {
  const { data: config } = useGameConfig();

  const arenaW = config?.arena.width ?? 1000;
  const arenaH = config?.arena.height ?? 1000;

  // Seed store with config obstacles until a RoundStart event arrives
  useEffect(() => {
    if (config?.obstacles && useGameStore.getState().obstacles.length === 0) {
      useGameStore.getState().setObstacles(config.obstacles);
    }
  }, [config]);

  return (
    <>
      <Header />
      <div className="main">
        <GameCanvas arenaW={arenaW} arenaH={arenaH} />
        <Sidebar>
          <Scoreboard />
          <AgentList />
          <KillFeed />
          <Terminal />
        </Sidebar>
      </div>
    </>
  );
}
