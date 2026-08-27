# config.py
# central config for aeromind - every hyperparameter, aircraft param
# and stage threshold lives here, nothing hardcoded elsewhere
# imported by jsbsim_env, reward, curriculum, train, evaluate,
# emergency, vision, stats, knowledge, and api.server

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AircraftConfig:
    # xtr-33 outdoor variant, flat profile 3d aerobatic park flyer
    # printables.com/model/1268237-xtr-33
    # geometry is from the published model, mass and thrust are
    # estimates - update once the real plane is built and bench
    # tested, nothing else in the codebase needs editing when they do
    name: str = "xtr-33-outdoor"
    wingspan_m: float = 0.95
    mean_chord_m: float = 0.22
    wing_area_m2: float = 0.209
    aspect_ratio: float = 4.32

    mass_kg: float = 0.180
    stall_speed_estimate_mps: float = 3.7

    battery_cells: int = 3
    motor_kv: int = 1800
    prop_diameter_in: float = 8.0
    prop_pitch_in: float = 4.3
    max_static_thrust_n: float = 3.92

    aileron_max_deflection_deg: float = 35.0
    elevator_max_deflection_deg: float = 35.0
    rudder_max_deflection_deg: float = 35.0

    @property
    def thrust_to_weight(self) -> float:
        weight_n = self.mass_kg * 9.81
        return self.max_static_thrust_n / weight_n


@dataclass(frozen=True)
class ActionConfig:
    # normalized rl action bounds, mapped to aircraft config degrees
    # and throttle fraction inside jsbsim_env
    aileron_range: tuple = (-1.0, 1.0)
    elevator_range: tuple = (-1.0, 1.0)
    rudder_range: tuple = (-1.0, 1.0)
    throttle_range: tuple = (0.0, 1.0)


@dataclass(frozen=True)
class SimConfig:
    jsbsim_root_dir: str = ""
    aircraft_model: str = "generic_trainer"
    sim_dt: float = 1.0 / 120.0
    control_dt: float = 1.0 / 60.0
    max_episode_seconds: float = 60.0
    initial_altitude_m: float = 100.0
    initial_airspeed_mps: float = 10.0
    initial_heading_deg: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "SAC"
    policy: str = "MlpPolicy"
    total_timesteps_per_stage: int = 1_000_000
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    seed: int = 42
    checkpoint_every_steps: int = 50_000


@dataclass(frozen=True)
class Stage1Stabilization:
    target_altitude_m: float = 100.0
    target_heading_deg: float = 0.0
    max_roll_deg: float = 5.0
    max_pitch_deg: float = 5.0
    airspeed_tolerance_mps: float = 1.0
    hold_duration_s: float = 15.0


@dataclass(frozen=True)
class Stage2Navigation:
    num_waypoints: int = 4
    waypoint_spacing_m: float = 150.0
    capture_radius_m: float = 10.0
    altitude_tolerance_m: float = 10.0


@dataclass(frozen=True)
class Stage3Takeoff:
    target_climb_rate_mps: float = 2.0
    min_airspeed_margin_mps: float = 2.0
    max_roll_during_climb_deg: float = 10.0
    climbout_altitude_m: float = 30.0


@dataclass(frozen=True)
class Stage4Landing:
    approach_altitude_m: float = 20.0
    max_touchdown_sink_rate_mps: float = 0.5
    centerline_tolerance_m: float = 3.0
    target_touchdown_airspeed_mps: float = 7.0
    touchdown_zone_length_m: float = 30.0


@dataclass(frozen=True)
class Stage5Disturbance:
    wind_speed_range_mps: tuple = (0.0, 8.0)
    gust_magnitude_range_mps: tuple = (0.0, 4.0)
    turbulence_intensity_range: tuple = (0.0, 0.3)
    crosswind_component_range_mps: tuple = (0.0, 5.0)


@dataclass(frozen=True)
class Stage6Emergency:
    engine_failure_probability: float = 0.1
    low_battery_threshold_pct: float = 20.0
    sensor_dropout_duration_range_s: tuple = (0.5, 3.0)
    sensor_dropout_probability: float = 0.1
    glide_target_sink_rate_mps: float = 1.5


@dataclass(frozen=True)
class VisionConfig:
    image_width: int = 64
    image_height: int = 64
    channels: int = 1
    enabled_from_stage: int = 5


@dataclass(frozen=True)
class KnowledgeConfig:
    persist_dir: Path = Path("data/aviation_kb")
    collection_name: str = "aviation_kb"
    chunk_size_chars: int = 800
    chunk_overlap_chars: int = 100
    top_k_results: int = 3


@dataclass(frozen=True)
class PathsConfig:
    models_dir: Path = Path("models")
    replays_dir: Path = Path("replays")
    live_stats_path: Path = Path("api/live_stats.json")
    log_dir: Path = Path("logs")


@dataclass(frozen=True)
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    poll_interval_seconds: float = 5.0


@dataclass(frozen=True)
class Config:
    aircraft: AircraftConfig = field(default_factory=AircraftConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stage1: Stage1Stabilization = field(default_factory=Stage1Stabilization)
    stage2: Stage2Navigation = field(default_factory=Stage2Navigation)
    stage3: Stage3Takeoff = field(default_factory=Stage3Takeoff)
    stage4: Stage4Landing = field(default_factory=Stage4Landing)
    stage5: Stage5Disturbance = field(default_factory=Stage5Disturbance)
    stage6: Stage6Emergency = field(default_factory=Stage6Emergency)
    vision: VisionConfig = field(default_factory=VisionConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


config = Config()


if __name__ == "__main__":
    print(f"aircraft: {config.aircraft.name}")
    print(f"wingspan: {config.aircraft.wingspan_m} m")
    print(f"wing area: {config.aircraft.wing_area_m2} m2")
    print(f"mass: {config.aircraft.mass_kg} kg")
    print(f"max thrust: {config.aircraft.max_static_thrust_n} n")
    print(f"thrust to weight: {config.aircraft.thrust_to_weight:.2f}")
    print(f"stall speed estimate: {config.aircraft.stall_speed_estimate_mps} m/s")