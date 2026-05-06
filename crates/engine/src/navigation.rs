use bevy_ecs::prelude::*;
use glam::Vec2;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Static navigation grid computed from obstacle geometry.
/// Inserted as a Bevy Resource at world setup time.
#[derive(Resource, Debug, Clone)]
pub struct NavGrid {
    cells: Vec<bool>,
    pub width: usize,
    pub height: usize,
    pub cell_size: f32,
    pub origin: Vec2,
}

/// A* open-set entry, ordered so the *smallest* f-score is popped first.
#[derive(Debug, Clone)]
struct AstarNode {
    pos: (usize, usize),
    f_score: f32,
}

impl PartialEq for AstarNode {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
    }
}

impl Eq for AstarNode {}

impl Ord for AstarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) acts as a min-heap.
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AstarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Cost of moving one cell in a cardinal direction.
const CARDINAL_COST: f32 = 1.0;
/// Cost of moving one cell diagonally.
const DIAGONAL_COST: f32 = std::f32::consts::SQRT_2;

/// 8-connectivity neighbor offsets: (dx, dy) as isize.
const NEIGHBORS: [(isize, isize); 8] = [
    (0, 1),   // N
    (1, 1),   // NE
    (1, 0),   // E
    (1, -1),  // SE
    (0, -1),  // S
    (-1, -1), // SW
    (-1, 0),  // W
    (-1, 1),  // NW
];

impl NavGrid {
    /// Build a nav grid from arena bounds and obstacle rectangles.
    ///
    /// `obstacles` is a slice of `(center, half_extents)` pairs.
    /// `agent_radius` inflates each obstacle so agents don't clip through walls.
    /// Cells overlapping inflated obstacle AABBs are marked as blocked.
    /// Boundary cells within `agent_radius` of the arena edge are also blocked.
    pub fn from_obstacles(
        arena_width: f32,
        arena_height: f32,
        obstacles: &[(Vec2, Vec2)],
        cell_size: f32,
        agent_radius: f32,
    ) -> Self {
        let width = (arena_width / cell_size).ceil() as usize;
        let height = (arena_height / cell_size).ceil() as usize;
        let origin = Vec2::new(cell_size / 2.0, cell_size / 2.0);
        let half_cell = cell_size / 2.0;

        let mut cells = vec![true; width * height];

        for gy in 0..height {
            for gx in 0..width {
                let cx = origin.x + gx as f32 * cell_size;
                let cy = origin.y + gy as f32 * cell_size;

                // -- Boundary check: block if the cell AABB is within
                //    agent_radius of any arena edge.
                let cell_min_x = cx - half_cell;
                let cell_min_y = cy - half_cell;
                let cell_max_x = cx + half_cell;
                let cell_max_y = cy + half_cell;

                if cell_min_x < agent_radius
                    || cell_min_y < agent_radius
                    || cell_max_x > arena_width - agent_radius
                    || cell_max_y > arena_height - agent_radius
                {
                    cells[gy * width + gx] = false;
                    continue;
                }

                // -- Obstacle overlap: inflate each obstacle by agent_radius and
                //    test AABB intersection with the cell AABB.
                for &(center, half_ext) in obstacles {
                    let obs_min_x = center.x - half_ext.x - agent_radius;
                    let obs_min_y = center.y - half_ext.y - agent_radius;
                    let obs_max_x = center.x + half_ext.x + agent_radius;
                    let obs_max_y = center.y + half_ext.y + agent_radius;

                    // AABB-AABB overlap test.
                    if cell_min_x < obs_max_x
                        && cell_max_x > obs_min_x
                        && cell_min_y < obs_max_y
                        && cell_max_y > obs_min_y
                    {
                        cells[gy * width + gx] = false;
                        break;
                    }
                }
            }
        }

        Self {
            cells,
            width,
            height,
            cell_size,
            origin,
        }
    }

    // -- internal helpers ------------------------------------------------

    /// Flat index into `cells` for grid coordinates.
    #[inline]
    fn idx(&self, gx: usize, gy: usize) -> usize {
        gy * self.width + gx
    }

    /// Whether grid cell `(gx, gy)` is walkable (bounds-checked).
    #[inline]
    fn cell_walkable(&self, gx: usize, gy: usize) -> bool {
        gx < self.width && gy < self.height && self.cells[self.idx(gx, gy)]
    }

    /// Octile heuristic: max(|dx|,|dy|) + (sqrt(2)-1) * min(|dx|,|dy|).
    #[inline]
    fn heuristic(a: (usize, usize), b: (usize, usize)) -> f32 {
        let dx = (a.0 as f32 - b.0 as f32).abs();
        let dy = (a.1 as f32 - b.1 as f32).abs();
        dx.max(dy) + (DIAGONAL_COST - CARDINAL_COST) * dx.min(dy)
    }

    // -- public API -------------------------------------------------------

    /// Check if a world position falls on a walkable cell.
    pub fn is_walkable(&self, pos: Vec2) -> bool {
        match self.world_to_grid(pos) {
            Some((gx, gy)) => self.cell_walkable(gx, gy),
            None => false,
        }
    }

    /// Convert a world position to grid coordinates.
    /// Returns `None` if outside grid bounds.
    pub fn world_to_grid(&self, pos: Vec2) -> Option<(usize, usize)> {
        if pos.x < 0.0 || pos.y < 0.0 {
            return None;
        }
        let gx = (pos.x / self.cell_size) as usize;
        let gy = (pos.y / self.cell_size) as usize;
        if gx >= self.width || gy >= self.height {
            return None;
        }
        Some((gx, gy))
    }

    /// Convert grid coordinates to the world-space center of that cell.
    pub fn grid_to_world(&self, gx: usize, gy: usize) -> Vec2 {
        Vec2::new(
            self.origin.x + gx as f32 * self.cell_size,
            self.origin.y + gy as f32 * self.cell_size,
        )
    }

    /// Find a path from `from` to `to` using A* with 8-connectivity.
    ///
    /// Returns a list of world-space waypoints (cell centers), or `None` if no
    /// path exists. The first waypoint is the cell nearest to `from`, the last
    /// is nearest to `to`.
    pub fn astar(&self, from: Vec2, to: Vec2) -> Option<Vec<Vec2>> {
        let start = self.world_to_grid(from)?;
        let goal = self.world_to_grid(to)?;

        if !self.cell_walkable(start.0, start.1) || !self.cell_walkable(goal.0, goal.1) {
            return None;
        }

        // Same cell: trivial path.
        if start == goal {
            return Some(vec![self.grid_to_world(start.0, start.1)]);
        }

        let mut open = BinaryHeap::new();
        let mut g_score: HashMap<(usize, usize), f32> = HashMap::new();
        let mut came_from: HashMap<(usize, usize), (usize, usize)> = HashMap::new();

        g_score.insert(start, 0.0);
        open.push(AstarNode {
            pos: start,
            f_score: Self::heuristic(start, goal) * self.cell_size,
        });

        while let Some(current) = open.pop() {
            let (cx, cy) = current.pos;

            if current.pos == goal {
                // Reconstruct path.
                let mut path = vec![goal];
                let mut node = goal;
                while let Some(&parent) = came_from.get(&node) {
                    path.push(parent);
                    node = parent;
                }
                path.reverse();
                return Some(
                    path.into_iter()
                        .map(|(gx, gy)| self.grid_to_world(gx, gy))
                        .collect(),
                );
            }

            let current_g = g_score[&current.pos];

            for &(dx, dy) in &NEIGHBORS {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx < 0 || ny < 0 {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                if !self.cell_walkable(nx, ny) {
                    continue;
                }

                let step_cost = if dx == 0 || dy == 0 {
                    CARDINAL_COST
                } else {
                    DIAGONAL_COST
                };
                let tentative_g = current_g + step_cost * self.cell_size;

                let existing_g = g_score.get(&(nx, ny)).copied().unwrap_or(f32::INFINITY);
                if tentative_g < existing_g {
                    came_from.insert((nx, ny), current.pos);
                    g_score.insert((nx, ny), tentative_g);
                    let h = Self::heuristic((nx, ny), goal) * self.cell_size;
                    open.push(AstarNode {
                        pos: (nx, ny),
                        f_score: tentative_g + h,
                    });
                }
            }
        }

        // Exhausted open set without reaching goal.
        None
    }

    pub fn path_distance(&self, from: Vec2, to: Vec2) -> Option<f32> {
        let path = self.astar(from, to)?;
        if path.len() <= 1 {
            return Some(0.0);
        }
        let mut total = 0.0f32;
        for pair in path.windows(2) {
            total += pair[0].distance(pair[1]);
        }
        Some(total)
    }

    /// If `pos` is on a blocked cell, search along `direction` for the nearest
    /// walkable cell. Returns the original position if already walkable.
    pub fn snap_to_walkable(&self, pos: Vec2, direction: Vec2) -> Vec2 {
        if self.is_walkable(pos) {
            return pos;
        }

        // Normalize direction; if it's zero-length, fall through to spiral.
        let dir_len = direction.length();
        if dir_len > f32::EPSILON {
            let dir = direction / dir_len;
            // Step along direction in cell_size increments.
            for step in 1..=20 {
                let candidate = pos + dir * self.cell_size * step as f32;
                if self.is_walkable(candidate) {
                    return candidate;
                }
            }
        }

        // Fallback: spiral search outward from the grid cell nearest to `pos`.
        if let Some((gx, gy)) = self.world_to_grid(pos) {
            let max_radius = self.width.max(self.height);
            for r in 1..=max_radius {
                let r = r as isize;
                // Walk the perimeter of the square at radius r.
                for d in -r..=r {
                    let candidates = [
                        (gx as isize + d, gy as isize + r),  // top edge
                        (gx as isize + d, gy as isize - r),  // bottom edge
                        (gx as isize + r, gy as isize + d),  // right edge
                        (gx as isize - r, gy as isize + d),  // left edge
                    ];
                    for (cx, cy) in candidates {
                        if cx >= 0
                            && cy >= 0
                            && (cx as usize) < self.width
                            && (cy as usize) < self.height
                            && self.cell_walkable(cx as usize, cy as usize)
                        {
                            return self.grid_to_world(cx as usize, cy as usize);
                        }
                    }
                }
            }
        }

        // Absolute last resort: return the original position unchanged.
        pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELL: f32 = 10.0;
    const RADIUS: f32 = 5.0;

    /// Helper: build a NavGrid for a 200x200 arena with no obstacles.
    fn open_grid() -> NavGrid {
        NavGrid::from_obstacles(200.0, 200.0, &[], CELL, RADIUS)
    }

    /// Helper: build a 200x200 grid with a single 40x40 obstacle at center (100, 100).
    fn center_obstacle_grid() -> NavGrid {
        let obstacles = vec![(Vec2::new(100.0, 100.0), Vec2::new(20.0, 20.0))];
        NavGrid::from_obstacles(200.0, 200.0, &obstacles, CELL, RADIUS)
    }

    // ================================================================
    // 1. Grid construction
    // ================================================================

    #[test]
    fn grid_dimensions_match_arena_and_cell_size() {
        let grid = open_grid();
        // 200 / 10 = 20
        assert_eq!(grid.width, 20);
        assert_eq!(grid.height, 20);
        assert_eq!(grid.cells.len(), 20 * 20);
    }

    #[test]
    fn grid_dimensions_round_up_for_non_integer_division() {
        // 150 / 10 = 15 exactly, 155 / 10 = 15.5 -> ceil -> 16
        let grid = NavGrid::from_obstacles(155.0, 150.0, &[], CELL, 0.0);
        assert_eq!(grid.width, 16);
        assert_eq!(grid.height, 15);
    }

    #[test]
    fn open_arena_interior_cells_are_walkable() {
        let grid = open_grid();
        // Interior cells (away from boundary) should all be walkable
        // With agent_radius=5 and cell_size=10, boundary is ~1 cell thick
        for gy in 1..grid.height - 1 {
            for gx in 1..grid.width - 1 {
                let pos = grid.grid_to_world(gx, gy);
                assert!(
                    grid.is_walkable(pos),
                    "interior cell ({gx}, {gy}) at {pos:?} should be walkable"
                );
            }
        }
    }

    #[test]
    fn open_arena_boundary_cells_are_blocked() {
        let grid = open_grid();
        // Bottom row (gy=0) should be blocked due to agent_radius
        for gx in 0..grid.width {
            let pos = grid.grid_to_world(gx, 0);
            assert!(
                !grid.is_walkable(pos),
                "boundary cell ({gx}, 0) at {pos:?} should be blocked"
            );
        }
        // Top row (gy = height-1) should be blocked
        let last_y = grid.height - 1;
        for gx in 0..grid.width {
            let pos = grid.grid_to_world(gx, last_y);
            assert!(
                !grid.is_walkable(pos),
                "boundary cell ({gx}, {last_y}) at {pos:?} should be blocked"
            );
        }
        // Left column (gx=0)
        for gy in 0..grid.height {
            let pos = grid.grid_to_world(0, gy);
            assert!(
                !grid.is_walkable(pos),
                "boundary cell (0, {gy}) at {pos:?} should be blocked"
            );
        }
        // Right column (gx = width-1)
        let last_x = grid.width - 1;
        for gy in 0..grid.height {
            let pos = grid.grid_to_world(last_x, gy);
            assert!(
                !grid.is_walkable(pos),
                "boundary cell ({last_x}, {gy}) at {pos:?} should be blocked"
            );
        }
    }

    #[test]
    fn obstacle_blocks_cells_beneath_it() {
        let grid = center_obstacle_grid();
        // Obstacle center at (100, 100) with half_extents (20, 20) + agent_radius 5
        // So blocked region: center +/- 25 in each axis -> [75..125]
        // Cell at world (100, 100) should definitely be blocked
        assert!(!grid.is_walkable(Vec2::new(100.0, 100.0)));
    }

    #[test]
    fn obstacle_inflation_blocks_nearby_cells() {
        let grid = center_obstacle_grid();
        // With agent_radius=5 and half_extent=20, blocked region extends to 25 units
        // from center in each direction. A cell at (80, 100) has center inside [75..125],
        // so it should be blocked.
        assert!(!grid.is_walkable(Vec2::new(80.0, 100.0)));
    }

    #[test]
    fn cells_outside_obstacle_inflation_are_walkable() {
        let grid = center_obstacle_grid();
        // Cell at (50, 50) is well outside the obstacle inflation zone
        // and also not on a boundary, so it should be walkable
        assert!(grid.is_walkable(Vec2::new(50.0, 50.0)));
    }

    #[test]
    fn origin_is_half_cell_offset() {
        let grid = open_grid();
        assert!((grid.origin.x - CELL / 2.0).abs() < f32::EPSILON);
        assert!((grid.origin.y - CELL / 2.0).abs() < f32::EPSILON);
    }

    // ================================================================
    // 2. Coordinate conversion
    // ================================================================

    #[test]
    fn world_to_grid_round_trip() {
        let grid = open_grid();
        // Pick an interior cell
        let gx = 5_usize;
        let gy = 7_usize;
        let world = grid.grid_to_world(gx, gy);
        let (rx, ry) = grid.world_to_grid(world).expect("should be in bounds");
        assert_eq!(rx, gx);
        assert_eq!(ry, gy);
    }

    #[test]
    fn world_to_grid_returns_none_for_negative_coords() {
        let grid = open_grid();
        assert!(grid.world_to_grid(Vec2::new(-10.0, 50.0)).is_none());
        assert!(grid.world_to_grid(Vec2::new(50.0, -10.0)).is_none());
    }

    #[test]
    fn world_to_grid_returns_none_for_out_of_bounds() {
        let grid = open_grid();
        // Arena is 200x200, grid is 20x20 cells of size 10
        assert!(grid.world_to_grid(Vec2::new(250.0, 100.0)).is_none());
        assert!(grid.world_to_grid(Vec2::new(100.0, 250.0)).is_none());
    }

    #[test]
    fn grid_to_world_produces_cell_centers() {
        let grid = open_grid();
        // Cell (0,0) center should be at origin (5, 5)
        let pos = grid.grid_to_world(0, 0);
        assert!((pos.x - 5.0).abs() < f32::EPSILON);
        assert!((pos.y - 5.0).abs() < f32::EPSILON);

        // Cell (1,0) center should be at (15, 5)
        let pos = grid.grid_to_world(1, 0);
        assert!((pos.x - 15.0).abs() < f32::EPSILON);
        assert!((pos.y - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn world_to_grid_handles_positions_within_cell() {
        let grid = open_grid();
        // Position (17.0, 8.0) should map to cell (1, 0) since cell_size=10
        // cell (1,0) spans x=[10..20), y=[0..10)
        let (gx, gy) = grid.world_to_grid(Vec2::new(17.0, 8.0)).unwrap();
        assert_eq!(gx, 1);
        assert_eq!(gy, 0);
    }

    // ================================================================
    // 3. is_walkable
    // ================================================================

    #[test]
    fn is_walkable_interior_empty() {
        let grid = open_grid();
        // Interior cell
        assert!(grid.is_walkable(Vec2::new(100.0, 100.0)));
    }

    #[test]
    fn is_walkable_under_obstacle() {
        let grid = center_obstacle_grid();
        assert!(!grid.is_walkable(Vec2::new(100.0, 100.0)));
    }

    #[test]
    fn is_walkable_boundary() {
        let grid = open_grid();
        // Position near the edge (cell 0,0 center is at 5,5, which is within agent_radius of 0)
        assert!(!grid.is_walkable(Vec2::new(5.0, 5.0)));
    }

    #[test]
    fn is_walkable_out_of_bounds_returns_false() {
        let grid = open_grid();
        assert!(!grid.is_walkable(Vec2::new(-10.0, 100.0)));
        assert!(!grid.is_walkable(Vec2::new(300.0, 100.0)));
    }

    // ================================================================
    // 4. A* pathfinding
    // ================================================================

    #[test]
    fn astar_straight_line_in_open_space() {
        let grid = open_grid();
        let from = Vec2::new(50.0, 100.0);
        let to = Vec2::new(150.0, 100.0);
        let path = grid.astar(from, to).expect("path should exist in open arena");
        assert!(path.len() >= 2, "path should have at least start and end");
        // First waypoint should be near `from`, last near `to`
        assert!((path.first().unwrap().y - 100.0).abs() < CELL);
        assert!((path.last().unwrap().y - 100.0).abs() < CELL);
    }

    #[test]
    fn astar_path_around_obstacle() {
        let grid = center_obstacle_grid();
        // Path from left of obstacle to right of obstacle
        let from = Vec2::new(50.0, 100.0);
        let to = Vec2::new(150.0, 100.0);
        let path = grid.astar(from, to).expect("path should exist around obstacle");

        // All waypoints must be on walkable cells
        for wp in &path {
            assert!(
                grid.is_walkable(*wp),
                "waypoint {wp:?} should be on a walkable cell"
            );
        }

        // Path should go around the obstacle, meaning at least some waypoints
        // deviate from y=100
        let deviates = path.iter().any(|wp| (wp.y - 100.0).abs() > CELL);
        assert!(
            deviates,
            "path should deviate from straight line to go around obstacle"
        );
    }

    #[test]
    fn astar_no_path_returns_none() {
        // Create a completely walled-off scenario: obstacle fills the entire arena
        let obstacles = vec![(Vec2::new(100.0, 100.0), Vec2::new(100.0, 100.0))];
        let grid = NavGrid::from_obstacles(200.0, 200.0, &obstacles, CELL, RADIUS);
        // Everything should be blocked, so no path
        let result = grid.astar(Vec2::new(10.0, 10.0), Vec2::new(190.0, 190.0));
        assert!(result.is_none());
    }

    #[test]
    fn astar_same_cell_returns_single_waypoint() {
        let grid = open_grid();
        let pos = Vec2::new(100.0, 100.0);
        let path = grid.astar(pos, pos).expect("same-cell path should exist");
        assert_eq!(path.len(), 1, "same-cell path should have exactly one waypoint");
    }

    #[test]
    fn astar_diagonal_movement_works() {
        let grid = open_grid();
        let from = Vec2::new(30.0, 30.0);
        let to = Vec2::new(170.0, 170.0);
        let path = grid.astar(from, to).expect("diagonal path should exist");
        assert!(path.len() >= 2);

        // In an open grid, a diagonal path should use diagonal steps
        // So path length should be less than manhattan distance worth of waypoints
        // Manhattan from (2,2) to (16,16) = 28 steps
        // Diagonal from (2,2) to (16,16) = 14 steps
        assert!(
            path.len() <= 20,
            "diagonal path should be efficient, got {} waypoints",
            path.len()
        );
    }

    #[test]
    fn astar_returns_none_for_blocked_start() {
        let grid = center_obstacle_grid();
        // Start inside the obstacle
        let from = Vec2::new(100.0, 100.0);
        let to = Vec2::new(50.0, 50.0);
        let result = grid.astar(from, to);
        assert!(result.is_none(), "should return None when start is blocked");
    }

    #[test]
    fn astar_returns_none_for_blocked_goal() {
        let grid = center_obstacle_grid();
        let from = Vec2::new(50.0, 50.0);
        let to = Vec2::new(100.0, 100.0);
        let result = grid.astar(from, to);
        assert!(result.is_none(), "should return None when goal is blocked");
    }

    // ================================================================
    // 5. path_distance
    // ================================================================

    #[test]
    fn path_distance_straight_line() {
        let grid = open_grid();
        let from = Vec2::new(55.0, 105.0); // cell (5, 10)
        let to = Vec2::new(105.0, 105.0); // cell (10, 10)
        let dist = grid.path_distance(from, to).expect("distance should exist");
        // 5 cardinal steps * cell_size = 50.0
        assert!(
            (dist - 50.0).abs() < 1.0,
            "expected ~50.0, got {dist}"
        );
    }

    #[test]
    fn path_distance_around_obstacle_longer_than_euclidean() {
        let grid = center_obstacle_grid();
        let from = Vec2::new(50.0, 100.0);
        let to = Vec2::new(150.0, 100.0);
        let dist = grid
            .path_distance(from, to)
            .expect("distance should exist around obstacle");
        let euclidean = (to - from).length();
        assert!(
            dist > euclidean,
            "path distance ({dist}) should be greater than euclidean ({euclidean})"
        );
    }

    #[test]
    fn path_distance_returns_none_when_no_path() {
        let obstacles = vec![(Vec2::new(100.0, 100.0), Vec2::new(100.0, 100.0))];
        let grid = NavGrid::from_obstacles(200.0, 200.0, &obstacles, CELL, RADIUS);
        assert!(grid.path_distance(Vec2::new(10.0, 10.0), Vec2::new(190.0, 190.0)).is_none());
    }

    #[test]
    fn path_distance_same_cell_is_zero() {
        let grid = open_grid();
        let pos = Vec2::new(100.0, 100.0);
        let dist = grid.path_distance(pos, pos).expect("same-cell distance should exist");
        assert!(
            dist.abs() < f32::EPSILON,
            "same-cell distance should be 0, got {dist}"
        );
    }

    // ================================================================
    // 6. snap_to_walkable
    // ================================================================

    #[test]
    fn snap_to_walkable_already_walkable() {
        let grid = open_grid();
        let pos = Vec2::new(100.0, 100.0);
        let snapped = grid.snap_to_walkable(pos, Vec2::new(1.0, 0.0));
        // Should return a position on the same cell
        let (gx1, gy1) = grid.world_to_grid(pos).unwrap();
        let (gx2, gy2) = grid.world_to_grid(snapped).unwrap();
        assert_eq!(gx1, gx2);
        assert_eq!(gy1, gy2);
    }

    #[test]
    fn snap_to_walkable_from_blocked_cell() {
        let grid = center_obstacle_grid();
        let pos = Vec2::new(100.0, 100.0); // inside obstacle
        assert!(!grid.is_walkable(pos));

        let snapped = grid.snap_to_walkable(pos, Vec2::new(1.0, 0.0));
        assert!(
            grid.is_walkable(snapped),
            "snapped position {snapped:?} should be walkable"
        );
    }

    #[test]
    fn snap_to_walkable_respects_direction() {
        let grid = center_obstacle_grid();
        let pos = Vec2::new(100.0, 100.0); // inside obstacle

        let snapped_right = grid.snap_to_walkable(pos, Vec2::new(1.0, 0.0));
        let snapped_left = grid.snap_to_walkable(pos, Vec2::new(-1.0, 0.0));

        // Snapping right should produce a position to the right of center
        assert!(
            snapped_right.x > 100.0,
            "snap right should go right, got x={}",
            snapped_right.x
        );
        // Snapping left should produce a position to the left of center
        assert!(
            snapped_left.x < 100.0,
            "snap left should go left, got x={}",
            snapped_left.x
        );
    }

    #[test]
    fn snap_to_walkable_fallback_when_direction_leads_nowhere() {
        // Create a grid where the direction leads into more blocked cells,
        // but there ARE walkable cells elsewhere (via spiral fallback)
        let grid = center_obstacle_grid();
        let pos = Vec2::new(100.0, 100.0);

        // Direction toward more obstacle... but snap should still find a walkable cell
        // since the grid is not fully blocked
        let snapped = grid.snap_to_walkable(pos, Vec2::new(0.0, 0.0));
        assert!(
            grid.is_walkable(snapped),
            "fallback should still find a walkable cell, got {snapped:?}"
        );
    }

    // ================================================================
    // 7. Multiple obstacles
    // ================================================================

    #[test]
    fn multiple_obstacles_all_blocked() {
        let obstacles = vec![
            (Vec2::new(50.0, 50.0), Vec2::new(10.0, 10.0)),
            (Vec2::new(150.0, 150.0), Vec2::new(10.0, 10.0)),
        ];
        let grid = NavGrid::from_obstacles(200.0, 200.0, &obstacles, CELL, RADIUS);

        // Both obstacle centers should be blocked
        assert!(!grid.is_walkable(Vec2::new(50.0, 50.0)));
        assert!(!grid.is_walkable(Vec2::new(150.0, 150.0)));

        // Space between obstacles should be walkable
        assert!(grid.is_walkable(Vec2::new(100.0, 100.0)));
    }

    #[test]
    fn path_navigates_between_multiple_obstacles() {
        let obstacles = vec![
            (Vec2::new(100.0, 80.0), Vec2::new(30.0, 10.0)),
            (Vec2::new(100.0, 120.0), Vec2::new(30.0, 10.0)),
        ];
        let grid = NavGrid::from_obstacles(200.0, 200.0, &obstacles, CELL, RADIUS);

        // Path from left to right should find a way through the gap
        let path = grid.astar(Vec2::new(30.0, 100.0), Vec2::new(170.0, 100.0));
        assert!(path.is_some(), "should find path through gap between obstacles");

        let path = path.unwrap();
        for wp in &path {
            assert!(
                grid.is_walkable(*wp),
                "waypoint {wp:?} should be walkable"
            );
        }
    }

    // ================================================================
    // 8. Edge cases
    // ================================================================

    #[test]
    fn zero_agent_radius_no_boundary_inflation() {
        let grid = NavGrid::from_obstacles(100.0, 100.0, &[], 10.0, 0.0);
        // With zero agent_radius, even boundary cells should be walkable
        // since there's no inflation. Cell (0,0) center is at (5,5).
        assert!(grid.is_walkable(Vec2::new(5.0, 5.0)));
    }

    #[test]
    fn large_cell_size_produces_small_grid() {
        let grid = NavGrid::from_obstacles(100.0, 100.0, &[], 50.0, 0.0);
        assert_eq!(grid.width, 2);
        assert_eq!(grid.height, 2);
    }

    #[test]
    fn astar_adjacent_cells() {
        let grid = open_grid();
        // Two adjacent cell centers
        let from = Vec2::new(55.0, 105.0); // cell (5, 10)
        let to = Vec2::new(65.0, 105.0);   // cell (6, 10)
        let path = grid.astar(from, to).expect("adjacent path should exist");
        assert_eq!(path.len(), 2, "adjacent cells should have 2-waypoint path");
    }
}
