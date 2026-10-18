//! Human-like drag trajectories for slider captchas.
//!
//! The generated points are **deltas relative to the handle start position**
//! so a client can replay them with `mouse.move(handle_x + p.x, handle_y + p.y)`.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::f64::consts::TAU;

/// One sampled trajectory point: offset in pixels and time in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    /// Horizontal delta from the handle start.
    pub x: f64,

    /// Vertical delta from the handle start.
    pub y: f64,

    /// Milliseconds since the drag started.
    pub t: f64,
}

/// Default drag duration in milliseconds: `clamp(350 + 1.1 * distance, 300, 1500)`.
pub fn default_duration_ms(distance: f64) -> f64 {
    (350.0 + 1.1 * distance.max(0.0)).clamp(300.0, 1500.0)
}

/// Generate a human-like drag trajectory sampled at ~60 Hz.
///
/// The path eases in/out (smootherstep), adds a small overshoot near the end,
/// decaying tremor and per-point jitter, and always ends exactly at
/// `(distance, 0)`. The same `seed` produces the exact same points.
pub fn human_trajectory(distance: f64, duration_ms: f64, seed: u64) -> Vec<TrajectoryPoint> {
    let distance = if distance.is_finite() {
        distance.clamp(0.0, 1e6)
    } else {
        0.0
    };

    let duration = if duration_ms.is_finite() {
        duration_ms.clamp(80.0, 5000.0)
    } else {
        default_duration_ms(distance)
    };

    let n = ((duration / 16.7).ceil() as usize).max(2);
    let mut rng = SplitMix64::new(seed);

    let tremor_phase = rng.next_f64() * TAU;
    let tremor_freq = 2.0 + rng.next_f64() * 1.5;
    let tremor_amp = (0.8 + rng.next_f64() * 0.8).min(distance * 0.5);

    let overshoot_px = (2.0 + 0.02 * distance).clamp(2.0, 8.0).min(distance * 0.5);
    let overshoot_ratio = if distance > 1.0 {
        (overshoot_px / distance).min(0.08)
    } else {
        0.0
    };

    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        let u = i as f64 / (n - 1) as f64;
        let last = i + 1 == n;

        let base = distance * smootherstep(u);
        let bump = distance * overshoot_ratio * bump_curve(u);
        let tremor =
            tremor_amp * (u * tremor_freq * TAU + tremor_phase).sin() * 4.0 * u * (1.0 - u);

        let jitter_scale = distance.min(1.0);
        let jitter_x = (rng.next_f64() - 0.5) * 0.8 * jitter_scale;
        let jitter_y = (rng.next_f64() - 0.5) * 0.6 * jitter_scale;

        let x = if last {
            distance
        } else {
            (base + bump + tremor + jitter_x).max(0.0)
        };

        let y = if last { 0.0 } else { tremor * 0.35 + jitter_y };

        points.push(TrajectoryPoint {
            x,
            y,
            t: duration * u,
        });
    }

    points
}

/// `6u^5 - 15u^4 + 10u^3` — zero velocity at both ends.
fn smootherstep(u: f64) -> f64 {
    u * u * u * (u * (u * 6.0 - 15.0) + 10.0)
}

/// Sine bump on `[0.6, 1.0]`, peaking at `u = 0.8` and returning to 0 at `u = 1`.
fn bump_curve(u: f64) -> f64 {
    if !(0.6..1.0).contains(&u) {
        return 0.0;
    }

    ((u - 0.6) / 0.4 * PI).sin()
}

/// Small deterministic PRNG (SplitMix64), avoiding a `rand` dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);

        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;

        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_exactly_at_distance() {
        let points = human_trajectory(180.0, 620.0, 42);
        let last = points.last().unwrap();

        assert_eq!(last.x, 180.0);
        assert_eq!(last.y, 0.0);
        assert!((last.t - 620.0).abs() < 1e-9);
    }

    #[test]
    fn samples_at_60hz() {
        let points = human_trajectory(120.0, 600.0, 7);
        assert_eq!(points.len(), (600.0f64 / 16.7).ceil() as usize);

        for pair in points.windows(2) {
            let dt = pair[1].t - pair[0].t;
            assert!((dt - 600.0 / (points.len() - 1) as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn stays_within_bounds() {
        let distance = 200.0;
        let points = human_trajectory(distance, 700.0, 1234);

        for point in &points {
            assert!(
                (-3.0..=distance + 10.0).contains(&point.x),
                "x out of bounds: {}",
                point.x
            );
            assert!(point.y.abs() <= 3.0, "y out of bounds: {}", point.y);
        }

        // The path mostly moves forward; small corrections around the
        // overshoot are human-like.
        let forward = points.windows(2).filter(|w| w[1].x >= w[0].x).count();
        assert!(
            forward * 4 >= points.len() * 3,
            "too many backward steps: {forward}/{}",
            points.len()
        );

        let max_back = points
            .windows(2)
            .map(|w| (w[0].x - w[1].x).max(0.0))
            .fold(0.0, f64::max);
        assert!(max_back <= 3.0, "backward jump too large: {max_back}");
    }

    #[test]
    fn is_deterministic_per_seed() {
        let a = human_trajectory(150.0, 500.0, 99);
        let b = human_trajectory(150.0, 500.0, 99);
        let c = human_trajectory(150.0, 500.0, 100);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn handles_zero_distance() {
        let points = human_trajectory(0.0, 300.0, 1);
        assert!(points.len() >= 2);

        for point in &points {
            assert_eq!(point.x, 0.0);
        }
    }

    #[test]
    fn default_duration_is_clamped() {
        assert_eq!(default_duration_ms(0.0), 350.0);
        assert_eq!(default_duration_ms(100.0), 460.0);
        assert_eq!(default_duration_ms(10_000.0), 1500.0);
    }
}
