# jsbsim_env.py
# custom gymnasium environment wrapping jsbsim
# reads every tunable value from config, hardcodes nothing
# used by train.py, evaluate.py, reward.py, curriculum.py, emergency.py

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import jsbsim

from config import config


EARTH_RADIUS_M = 6371000.0
FT_TO_M = 0.3048
KTS_TO_MPS = 0.514444
DEG_TO_RAD = math.pi / 180.0


class AeroMindEnv(gym.Env):
    # action: [aileron, elevator, rudder, throttle], all normalized
    # observation (16-d): roll, pitch, yaw, p, q, r, airspeed, altitude,
    # vspeed, alpha, dist-to-waypoint, bearing-to-waypoint, wind speed,
    # wind direction, battery pct, engine health
    # reward is intentionally always 0 here - reward.py owns scoring,
    # this file only owns physics plumbing
    metadata = {"render_modes": []}

    def __init__(self, aircraft_model: Optional[str] = None, jsbsim_root_dir: Optional[str] = None):
        super().__init__()
        self.aircraft_model = aircraft_model or config.sim.aircraft_model
        self.root_dir = jsbsim_root_dir if jsbsim_root_dir is not None else (config.sim.jsbsim_root_dir or None)

        self.fdm = jsbsim.FGFDMExec(self.root_dir)
        self.fdm.set_debug_level(0)
        self.fdm.load_model(self.aircraft_model)
        self.fdm.set_dt(config.sim.sim_dt)

        self._substeps = max(1, round(config.sim.control_dt / config.sim.sim_dt))
        self._initialized = False

        low = np.array([
            config.action.aileron_range[0],
            config.action.elevator_range[0],
            config.action.rudder_range[0],
            config.action.throttle_range[0],
        ], dtype=np.float32)
        high = np.array([
            config.action.aileron_range[1],
            config.action.elevator_range[1],
            config.action.rudder_range[1],
            config.action.throttle_range[1],
        ], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

        self.waypoint_lat_deg = 0.0
        self.waypoint_lon_deg = 0.0
        self.battery_pct = 100.0
        self.engine_health = 1.0
        self.elapsed_time_s = 0.0

    def _apply_initial_conditions(self):
        self.fdm.set_property_value("ic/h-sl-ft", config.sim.initial_altitude_m / FT_TO_M)
        self.fdm.set_property_value("ic/vc-kts", config.sim.initial_airspeed_mps / KTS_TO_MPS)
        self.fdm.set_property_value("ic/psi-true-deg", config.sim.initial_heading_deg)
        self.fdm.set_property_value("ic/lat-gc-deg", 0.0)
        self.fdm.set_property_value("ic/long-gc-deg", 0.0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._apply_initial_conditions()
        if not self._initialized:
            self.fdm.run_ic()
            self._initialized = True
        else:
            self.fdm.reset_to_initial_conditions(0)

        self.fdm.set_property_value("propulsion/engine[0]/set-running", 1)

        self.battery_pct = 100.0
        self.engine_health = 1.0
        self.elapsed_time_s = 0.0
        self.set_waypoint()

        return self._get_observation(), {}

    def set_waypoint(self, distance_m: Optional[float] = None, bearing_rad: Optional[float] = None):
        # places a target waypoint at distance_m/bearing_rad from current
        # position - random bearing if not given, used by stage 2 nav and
        # by curriculum.py to advance through a chain of legs
        distance_m = config.stage2.waypoint_spacing_m if distance_m is None else distance_m
        bearing_rad = self.np_random.uniform(-math.pi, math.pi) if bearing_rad is None else bearing_rad

        lat_rad = self.fdm.get_property_value("position/lat-gc-deg") * DEG_TO_RAD
        lon_rad = self.fdm.get_property_value("position/long-gc-deg") * DEG_TO_RAD
        d_over_r = distance_m / EARTH_RADIUS_M

        new_lat_rad = math.asin(
            math.sin(lat_rad) * math.cos(d_over_r)
            + math.cos(lat_rad) * math.sin(d_over_r) * math.cos(bearing_rad)
        )
        new_lon_rad = lon_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(d_over_r) * math.cos(lat_rad),
            math.cos(d_over_r) - math.sin(lat_rad) * math.sin(new_lat_rad),
        )
        self.waypoint_lat_deg = new_lat_rad / DEG_TO_RAD
        self.waypoint_lon_deg = new_lon_rad / DEG_TO_RAD

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        aileron, elevator, rudder, throttle = (float(a) for a in action)

        self.fdm.set_property_value("fcs/aileron-cmd-norm", aileron)
        self.fdm.set_property_value("fcs/elevator-cmd-norm", elevator)
        self.fdm.set_property_value("fcs/rudder-cmd-norm", rudder)
        self.fdm.set_property_value("fcs/throttle-cmd-norm", throttle * self.engine_health)

        for _ in range(self._substeps):
            self.fdm.run()

        self.elapsed_time_s += config.sim.control_dt
        self._update_battery(throttle)

        obs = self._get_observation()
        terminated = self._check_terminated()
        truncated = self.elapsed_time_s >= config.sim.max_episode_seconds
        reward = 0.0
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def _update_battery(self, throttle: float):
        # simple linear depletion, retuned later against real flight logs
        drain_per_s = 100.0 / (config.sim.max_episode_seconds * 4.0)
        self.battery_pct = max(0.0, self.battery_pct - drain_per_s * throttle * config.sim.control_dt)

    def _check_terminated(self) -> bool:
        agl_m = self.fdm.get_property_value("position/h-agl-ft") * FT_TO_M
        return agl_m <= 0.0

    def _distance_bearing_to_waypoint(self):
        lat_rad = self.fdm.get_property_value("position/lat-gc-deg") * DEG_TO_RAD
        lon_rad = self.fdm.get_property_value("position/long-gc-deg") * DEG_TO_RAD
        wp_lat_rad = self.waypoint_lat_deg * DEG_TO_RAD
        wp_lon_rad = self.waypoint_lon_deg * DEG_TO_RAD

        d_lat = wp_lat_rad - lat_rad
        d_lon = wp_lon_rad - lon_rad
        a = (math.sin(d_lat / 2) ** 2
             + math.cos(lat_rad) * math.cos(wp_lat_rad) * math.sin(d_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance_m = EARTH_RADIUS_M * c

        bearing_rad = math.atan2(
            math.sin(d_lon) * math.cos(wp_lat_rad),
            math.cos(lat_rad) * math.sin(wp_lat_rad)
            - math.sin(lat_rad) * math.cos(wp_lat_rad) * math.cos(d_lon),
        )
        return distance_m, bearing_rad

    def _get_observation(self) -> np.ndarray:
        fdm = self.fdm
        roll = fdm.get_property_value("attitude/phi-rad")
        pitch = fdm.get_property_value("attitude/theta-rad")
        yaw = fdm.get_property_value("attitude/psi-rad")
        p = fdm.get_property_value("velocities/p-rad_sec")
        q = fdm.get_property_value("velocities/q-rad_sec")
        r = fdm.get_property_value("velocities/r-rad_sec")
        airspeed = fdm.get_property_value("velocities/vc-kts") * KTS_TO_MPS
        altitude = fdm.get_property_value("position/h-sl-ft") * FT_TO_M
        vspeed = fdm.get_property_value("velocities/h-dot-fps") * FT_TO_M
        alpha = fdm.get_property_value("aero/alpha-deg") * DEG_TO_RAD
        wind_north = fdm.get_property_value("atmosphere/wind-north-fps") * FT_TO_M
        wind_east = fdm.get_property_value("atmosphere/wind-east-fps") * FT_TO_M
        wind_speed = math.hypot(wind_north, wind_east)
        wind_dir = math.atan2(wind_east, wind_north)

        distance_wp, bearing_wp = self._distance_bearing_to_waypoint()

        return np.array([
            roll, pitch, yaw,
            p, q, r,
            airspeed, altitude, vspeed,
            alpha,
            distance_wp, bearing_wp,
            wind_speed, wind_dir,
            self.battery_pct / 100.0,
            self.engine_health,
        ], dtype=np.float32)

    def _build_info(self) -> dict:
        return {
            "sim_time_s": self.fdm.get_sim_time(),
            "battery_pct": self.battery_pct,
            "engine_health": self.engine_health,
        }

    def close(self):
        pass


if __name__ == "__main__":
    # smoke test against a jsbsim bundled aircraft - aircraft/generic_trainer.xml
    # doesn't exist yet, swap aircraft_model back to config default once it does
    env = AeroMindEnv(aircraft_model="c182")
    obs, info = env.reset(seed=0)
    print("observation shape:", obs.shape)
    print("action space:", env.action_space)
    print("first observation:", obs)

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"episode ended at step {i}, terminated={terminated}, truncated={truncated}")
            break
    print("ran without crashing, final info:", info)