"""Domain-constrained SYNTHETIC mine-operations simulator (equipment level, shift resolution).

Why synthetic: no real mine-level, shift-level equipment / delay / production records are publicly
available for any Indian manganese mine (see docs/DATA_SOURCE_AUDIT.md). Weather INPUTS are real
(IMD / ERA5); every operational RESPONSE produced here is SYNTHETIC and labelled as such.

Mechanism (per shift, per unit):
  hazard = base_failures_per_100h/100 x (1 + age/pm_interval) x wet_multiplier x scenario_stress
  failure ~ Bernoulli(1 - exp(-hazard x available_hours)); repair ~ LogNormal(median, sigma)
  preventive maintenance when age >= pm_interval (consumes hours, resets age -> lowers later hazard)
  shared shocks: power outages (Poisson) and heavy-rain site stoppages (logistic in real rainfall)
  drills -> drilled metres -> daily blast (unless explosive-supply episode) -> blasted-ore inventory
  loading capacity  = sum(excavator operating h x ore rate x capacity trend)
  haulage capacity  = trucks available x payload x 60 / cycle(rain, soil moisture)
  production        = min(loading, haulage, blasted inventory) x utilisation x residual
Physical invariants are asserted: 0 <= availability, utilisation <= 1, operating <= available <=
scheduled, all hours >= 0, production >= 0.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EQUIP_PREFIX = {"excavator": "SYN_EXC", "dumper": "SYN_TRUCK", "drill": "SYN_DRILL"}
SCENARIOS = ("NORMAL", "HEAVY_RAIN", "EQUIPMENT_DEGRADATION", "BLAST_DELAY", "DRILL_DELAY",
             "HAULAGE_DISRUPTION", "COMBINED_DISRUPTION")
ACTIONS = ("EQUIPMENT_RECOVERY", "SCHEDULE_ADJUSTMENT", "BLAST_DRILL_DELAY_REDUCTION")


@dataclass
class Unit:
    equipment_id: str
    kind: str
    age_h: float = 0.0             # operating hours since last preventive maintenance
    broken_h: float = 0.0          # remaining repair hours
    in_service: bool = True        # trucks outside the current deployed fleet are not in service


@dataclass
class MineState:
    units: list
    inventory_t: float
    explosive_episode_days: int = 0
    drilled_pending_m: float = 0.0
    fleet_trucks: int = 18
    days_to_fleet_change: int = 100
    day_index: int = 0
    extra: dict = field(default_factory=dict)


class OpsSimulator:
    def __init__(self, cfg: dict, start_date: str):
        self.cfg = cfg
        self.start = pd.Timestamp(start_date)

    # ------------------------------------------------------------------ setup
    def initial_state(self, rng) -> MineState:
        f = self.cfg["fleet"]
        units = []
        for kind in ("excavator", "dumper", "drill"):
            n = f[kind]["n"] + (4 if kind == "dumper" else 0)   # 4 pool trucks for redeployment
            for i in range(n):
                units.append(Unit(f"{EQUIP_PREFIX[kind]}_{i + 1:03d}", kind,
                                  age_h=float(rng.uniform(0, f[kind]["pm_interval_h"]))))
        st = MineState(units=units, inventory_t=0.0, fleet_trucks=f["dumper"]["n"],
                       days_to_fleet_change=int(rng.exponential(self.cfg["fleet_changes"]["mean_days_between"])) + 1)
        self._apply_fleet(st)
        st.inventory_t = self.cfg["blasting"]["initial_inventory_days"] * self._nominal_daily_t()
        return st

    def _nominal_daily_t(self):
        e = self.cfg["fleet"]["excavator"]
        return e["n"] * e["ore_rate_tph"] * self.cfg["mine"]["scheduled_hours_per_day"] * 0.8

    def _apply_fleet(self, st: MineState):
        trucks = [u for u in st.units if u.kind == "dumper"]
        for i, u in enumerate(trucks):
            u.in_service = i < st.fleet_trucks

    # ------------------------------------------------------------------ one day
    def step_day(self, st: MineState, date, weather: dict, rng, scenario="NORMAL", actions=(), record_units=False,
                 evolve_regimes=True):
        c, f = self.cfg, self.cfg["fleet"]
        w, bl = c["weather"], c["blasting"]
        rain, sm, tmax = weather["rainfall_mm"], weather["soil_moisture_m3m3"], weather["temperature_max_c"]
        actions = set(actions)
        sc = {"COMBINED_DISRUPTION": {"HEAVY_RAIN", "EQUIPMENT_DEGRADATION", "BLAST_DELAY"}}.get(scenario, {scenario})

        # ---- scenario / action modifiers (used by the counterfactual engine; NORMAL + no action in history)
        if "HEAVY_RAIN" in sc:
            rain, sm = max(rain, float(weather.get("scenario_rain_mm", 25.0))), max(sm, 0.46)
        hazard_mult = {"excavator": 1.0, "dumper": 1.0, "drill": 1.0}
        repair_mult = {"excavator": 1.0, "dumper": 1.0, "drill": 1.0}
        if "EQUIPMENT_DEGRADATION" in sc:
            hazard_mult["excavator"] = hazard_mult["dumper"] = 3.0
        if "DRILL_DELAY" in sc:
            hazard_mult["drill"] = 4.0
        if "EQUIPMENT_RECOVERY" in actions:            # standby crew / critical spares
            repair_mult["excavator"] = repair_mult["dumper"] = 0.4
        if "BLAST_DRILL_DELAY_REDUCTION" in actions:   # drill spares, pre-positioned explosives
            repair_mult["drill"] = 0.5
        cycle_mult = 1.4 if "HAULAGE_DISRUPTION" in sc else 1.0
        truck_delta = -4 if "HAULAGE_DISRUPTION" in sc else 0
        if "SCHEDULE_ADJUSTMENT" in actions:            # redeploy trucks + re-sequence benches
            truck_delta += 3
            cycle_mult *= 0.9
        explosive_active = st.explosive_episode_days > 0 or "BLAST_DELAY" in sc
        blast_prob = bl["explosive_supply_blast_probability"] if explosive_active else 1.0
        if "BLAST_DRILL_DELAY_REDUCTION" in actions and explosive_active:
            blast_prob = max(blast_prob, 0.9)
        wet_hole = w["wet_hole_drill_loss"] * (0.5 if "BLAST_DRILL_DELAY_REDUCTION" in actions else 1.0)

        # ---- truck fleet regime (redeployments between operations)
        if evolve_regimes:   # history only; counterfactuals hold regimes fixed for every portfolio
            st.days_to_fleet_change -= 1
            if st.days_to_fleet_change <= 0:
                fc = c["fleet_changes"]
                st.fleet_trucks = int(np.clip(st.fleet_trucks + rng.choice(fc["step_choices"]), fc["min_trucks"], fc["max_trucks"]))
                st.days_to_fleet_change = int(rng.exponential(fc["mean_days_between"])) + 20
            if st.explosive_episode_days > 0:
                st.explosive_episode_days -= 1
            elif rng.random() < bl["explosive_supply_episode_rate_per_day"]:
                st.explosive_episode_days = int(rng.integers(*bl["explosive_supply_episode_days"]))
        deployed = int(np.clip(st.fleet_trucks + truck_delta, 0, f["dumper"]["n"] + 4))
        for i, u in enumerate([u for u in st.units if u.kind == "dumper"]):
            u.in_service = i < deployed

        # ---- site-wide shared shocks (same for every unit)
        wet = rain > 20 or sm > w["wet_soil_threshold"]
        p_stop = 1 / (1 + math.exp(-(rain - w["stoppage_logistic_mid_mm"]) / w["stoppage_logistic_scale_mm"]))
        stoppage = float(rng.uniform(*w["stoppage_hours"])) if rng.random() < p_stop else 0.0
        outage = float(rng.uniform(*c["shared_shocks"]["power_outage_hours"])) \
            if rng.random() < c["shared_shocks"]["power_outage_rate_per_day"] else 0.0
        site_lost = min(c["mine"]["scheduled_hours_per_day"], stoppage + outage)

        years = (pd.Timestamp(date) - self.start).days / 365.25
        trend = (1 + c["mine"]["capacity_trend_per_year"]) ** years
        cycle = f["dumper"]["base_cycle_min"] * cycle_mult * (1 + w["haul_cycle_per_mm"] * rain
                                                              + w["haul_cycle_per_sm_above"] * max(0.0, sm - w["wet_soil_threshold"]))
        sunday = pd.Timestamp(date).dayofweek == 6
        heat_loss = w["heat_utilisation_loss_per_c"] * max(0.0, tmax - w["heat_threshold_c"])

        day = {k: 0.0 for k in ("sched_ex", "avail_ex", "maint_ex", "break_ex", "oper_ex", "blast_wait", "haul_wait",
                                "drill_lost", "trucks_avail", "production", "drilled_m", "failures", "pm_events")}
        records = []
        shift_hours = c["mine"]["shifts"]
        for shift, sched in shift_hours.items():
            site_shift_lost = min(sched, site_lost * sched / c["mine"]["scheduled_hours_per_day"])
            avail = {}
            for u in st.units:
                if u.kind == "dumper" and not u.in_service:
                    continue
                spec = f[u.kind]
                maint = breakdown = 0.0
                failure = pm = False
                if u.broken_h > 0:
                    breakdown = min(sched, u.broken_h)
                    u.broken_h -= breakdown
                elif u.age_h >= spec["pm_interval_h"] and shift == "A":
                    maint = min(sched, float(rng.uniform(*spec["pm_hours"])))
                    u.age_h = 0.0
                    pm = True
                a_h = max(0.0, sched - maint - breakdown)
                if a_h > 0:
                    lam = spec["base_failures_per_100h"] / 100.0 * (1 + c["ageing"]["hazard_per_pm_interval"] * u.age_h / spec["pm_interval_h"])
                    if wet and u.kind in ("dumper", "drill"):
                        lam *= w["wet_failure_multiplier"]
                    lam *= hazard_mult[u.kind]
                    if rng.random() < 1 - math.exp(-lam * a_h):
                        failure = True
                        t_fail = float(rng.uniform(0, a_h))
                        repair = float(rng.lognormal(math.log(spec["repair_median_h"] * repair_mult[u.kind]), spec["repair_sigma"]))
                        lost_now = min(a_h - t_fail, repair)
                        breakdown += lost_now
                        u.broken_h = repair - lost_now
                        a_h = t_fail
                avail[u.equipment_id] = (u, a_h, maint, breakdown, failure, pm)

            eff = max(0.0, sched - site_shift_lost)
            ex = [v for v in avail.values() if v[0].kind == "excavator"]
            tr = [v for v in avail.values() if v[0].kind == "dumper"]
            dr = [v for v in avail.values() if v[0].kind == "drill"]
            frac = eff / sched if sched else 0.0
            ex_hours = sum(v[1] * frac for v in ex)
            load_cap = ex_hours * f["excavator"]["ore_rate_tph"] * trend
            trucks_equiv = sum(v[1] for v in tr) / sched if sched else 0.0
            haul_rate = trucks_equiv * f["dumper"]["payload_t"] * 60.0 / cycle      # t/h across the fleet
            haul_cap = haul_rate * eff
            util = float(np.clip(c["utilisation"]["base"] * (c["mine"]["sunday_crew_factor"] if sunday else 1.0)
                                 * (1 - heat_loss) + rng.normal(0, c["utilisation"]["noise_sd"]), 0.3, 1.0))
            possible = min(load_cap, haul_cap) * util
            produced = max(0.0, min(possible, st.inventory_t) * float(rng.lognormal(0, c["production"]["residual_lognormal_sd"])))
            produced = min(produced, st.inventory_t)
            st.inventory_t -= produced
            rate_ex = f["excavator"]["ore_rate_tph"] * trend
            n_ex = max(1, len(ex))
            haul_wait = max(0.0, (load_cap - haul_cap) * util / rate_ex) / n_ex if load_cap > 0 else 0.0
            blast_wait = max(0.0, (possible - produced) / rate_ex) / n_ex if possible > produced else 0.0
            dr_hours = sum(v[1] * frac for v in dr) * (1 - (wet_hole if rain > 20 else 0.0))
            drilled = dr_hours * f["drill"]["metres_per_h"]
            st.drilled_pending_m += drilled
            drill_lost = sum(sched - v[1] * frac * (1 - (wet_hole if rain > 20 else 0.0)) for v in dr) / max(1, len(dr))

            day["sched_ex"] += sched * len(ex)
            day["avail_ex"] += sum(v[1] for v in ex)
            day["maint_ex"] += sum(v[2] for v in ex)
            day["break_ex"] += sum(v[3] for v in ex)
            day["oper_ex"] += min(ex_hours * util, sum(v[1] for v in ex))
            day["blast_wait"] += blast_wait
            day["haul_wait"] += haul_wait
            day["drill_lost"] += drill_lost
            day["trucks_avail"] += trucks_equiv * sched
            day["production"] += produced
            day["drilled_m"] += drilled
            day["failures"] += sum(v[4] for v in avail.values())
            day["pm_events"] += sum(v[5] for v in avail.values())

            # operating hours accrue age (drives the failure hazard)
            for u, a_h, maint, brk, failure, pm in avail.values():
                operating = a_h * frac * (util if u.kind != "drill" else 1.0)
                u.age_h += operating
                if record_units:
                    cap = {"excavator": f["excavator"]["ore_rate_tph"] * trend, "dumper": f["dumper"]["payload_t"] * 60.0 / cycle,
                           "drill": f["drill"]["metres_per_h"]}[u.kind]
                    share = (produced * a_h / max(1e-9, sum(v[1] for v in ex))) if u.kind == "excavator" else 0.0
                    records.append({
                        "shift": shift, "equipment_id": u.equipment_id, "equipment_type": u.kind,
                        "capacity_per_h": round(cap, 2), "capacity_unit": {"excavator": "t ore/h", "dumper": "t/h", "drill": "m/h"}[u.kind],
                        "scheduled_hours": sched, "available_hours": round(a_h, 3), "operating_hours": round(operating, 3),
                        "idle_hours": round(a_h - operating, 3), "maintenance_hours": round(maint, 3),
                        "breakdown_hours": round(brk, 3), "availability": round(a_h / sched, 4),
                        "utilization": round(operating / a_h, 4) if a_h > 0 else 0.0,
                        "failure_event": failure, "maintenance_event": pm,
                        "weather_state": "HEAVY_RAIN" if rain >= w["heavy_rain_mm"] else ("WET" if wet else "DRY"),
                        "haulage_state": "CONSTRAINED" if haul_cap < load_cap else "NORMAL",
                        "production_contribution_t": round(share, 2),
                    })
        # daily blast: turns drilled metres into blasted ore (unless explosive supply prevents it)
        if st.drilled_pending_m > 0 and rng.random() < blast_prob:
            room = max(0.0, bl["max_inventory_days"] * self._nominal_daily_t() - st.inventory_t)
            blasted_m = min(st.drilled_pending_m, room / bl["tonnes_per_drill_metre"])
            st.inventory_t += blasted_m * bl["tonnes_per_drill_metre"]
            st.drilled_pending_m -= blasted_m
        st.day_index += 1

        sched_day = c["mine"]["scheduled_hours_per_day"]
        n_ex = f["excavator"]["n"]
        availability = day["avail_ex"] / day["sched_ex"]
        out = {
            "equipment_availability": availability,
            "equipment_downtime_h": day["break_ex"] / n_ex,
            "maintenance_hours": day["maint_ex"] / n_ex,
            "drilling_delay_h": day["drill_lost"] / 3.0,           # lost hours per drill, averaged per shift
            "blast_delay_h": day["blast_wait"],
            "truck_count": day["trucks_avail"] / sched_day,
            "haulage_delay_h": day["haul_wait"] + site_lost * 0.5,  # waiting for trucks + half of site stoppage
            "actual_tonnes": day["production"],
            "utilization": (day["oper_ex"] / day["avail_ex"]) if day["avail_ex"] > 0 else 0.0,
            "site_stoppage_h": stoppage, "power_outage_h": outage, "explosive_episode": explosive_active,
            "blasted_inventory_t": st.inventory_t, "failures": day["failures"], "pm_events": day["pm_events"],
            "deployed_trucks": deployed,
        }
        check_day(out, sched_day)
        return out, records


def check_day(d, sched):
    """Programmatic physical-consistency assertions for one simulated day."""
    assert 0.0 <= d["equipment_availability"] <= 1.0 + 1e-9, d
    assert 0.0 <= d["utilization"] <= 1.0 + 1e-9, d
    for k in ("equipment_downtime_h", "maintenance_hours", "drilling_delay_h", "blast_delay_h", "haulage_delay_h",
              "actual_tonnes", "truck_count", "blasted_inventory_t"):
        assert d[k] >= -1e-9, (k, d[k])
    assert d["equipment_downtime_h"] + d["maintenance_hours"] <= sched + 1e-6, d


def simulate(cfg, weather: pd.DataFrame, seed: int, record_units=False, snapshot_every=None):
    """Run the simulator over `weather` (daily rows). Returns (daily df, unit-shift df, snapshots)."""
    rng = np.random.default_rng(seed)
    sim = OpsSimulator(cfg, str(weather["date"].iloc[0]))
    st = sim.initial_state(rng)
    days, units, snaps = [], [], {}
    for i, row in enumerate(weather.itertuples(index=False)):
        w = row._asdict()
        if snapshot_every and i % snapshot_every == 0:
            snaps[str(w["date"])[:10]] = (copy.deepcopy(st), int(rng.integers(0, 2**31 - 1)))
        out, recs = sim.step_day(st, w["date"], w, rng, record_units=record_units)
        out["date"] = str(w["date"])[:10]
        days.append(out)
        for r in recs:
            r["date"] = out["date"]
        units.extend(recs)
    return pd.DataFrame(days), pd.DataFrame(units), snaps, sim


def counterfactual_week(sim: OpsSimulator, state: MineState, weather_week: pd.DataFrame, seed: int,
                        scenario="NORMAL", actions=()):
    """Simulated 7-day production under a scenario and action portfolio (common random numbers)."""
    st = copy.deepcopy(state)
    rng = np.random.default_rng(seed)
    total, rows = 0.0, []
    for row in weather_week.itertuples(index=False):
        w = row._asdict()
        out, _ = sim.step_day(st, w["date"], w, rng, scenario=scenario, actions=actions, evolve_regimes=False)
        total += out["actual_tonnes"]
        rows.append(out)
    return total, pd.DataFrame(rows)
