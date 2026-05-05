use std::time::Instant;

use ghostlobby_engine::config::GameConfig;
use ghostlobby_engine::tick::TickRunner;

fn main() {
    let config = GameConfig::from_file("configs/arena_deathmatch.json")
        .expect("failed to load config");

    println!("GhostLobby Headless Benchmark");
    println!("Config: {}", config.title);
    println!("Teams: {} x {}", config.teams.count, config.teams.players_per_team);
    println!();

    let mut runner = TickRunner::new(config);
    let total_ticks: u64 = 1_000_000;

    let start = Instant::now();
    for _ in 0..total_ticks {
        runner.tick();
        let _ = runner.drain_telemetry();
    }
    let elapsed = start.elapsed();

    let tps = total_ticks as f64 / elapsed.as_secs_f64();
    let us_per_tick = elapsed.as_micros() as f64 / total_ticks as f64;

    println!("Ticks:       {}", total_ticks);
    println!("Elapsed:     {:.2}s", elapsed.as_secs_f64());
    println!("Ticks/sec:   {:.0}", tps);
    println!("us/tick:     {:.2}", us_per_tick);
}
