import csv
import random

from maspy import Agent, Environment, Goal, Belief, action, Any, Percept, pl, gain, broadcast, tell, Channel, Admin
from maspy.learning import EnvModel, qlearning, listed
from rl_common import add_derived_state_features, phase_allows, served_counts, transition_step
from config import (
    HORIZON,
    QUEUE_CAP,
    SERVICE_RATE,
    ARR_MIN,
    ARR_MAX,
    SWITCH_PENALTY,
    QUEUE_PENALTY_W,
    PRIORITY_BONUS,
    PRIORITY_PENALTY,
    MIN_GREEN,
)

DIRECTIONS = ("N", "S", "E", "W")
PHASES = ("N", "S", "E", "W", "NS", "EW")


class Intersection(Environment):
    def __init__(self, env_name: str):
        super().__init__(env_name)
        # Parameters for the simple traffic dynamics
        self.horizon: int = HORIZON
        self.service_rate: int = SERVICE_RATE
        # Random arrivals per step (min..max)
        self.arr_min: int = ARR_MIN
        self.arr_max: int = ARR_MAX
        self.queue_cap: int = QUEUE_CAP
        self.switch_penalty: float = SWITCH_PENALTY
        self.queue_penalty_w: float = QUEUE_PENALTY_W
        self.priority_bonus: float = PRIORITY_BONUS
        self.priority_penalty: float = PRIORITY_PENALTY
        self.min_green: int = MIN_GREEN
        self.last_switch_time: int = 0
        self.yellow_duration: int = 2
        self.arr_lambda: float = 1.0
        self.arr_rng = random.Random()
        self.metrics = {
            "steps": [],
            "queue_sum_total": 0,
            "step_count": 0,
            "throughput": {"N": 0, "S": 0, "E": 0, "W": 0},
        }
        
        self.create(Percept("phase", PHASES, listed))
        self.create(Percept("queue_n", list(range(self.queue_cap + 1)), listed))
        self.create(Percept("queue_s", list(range(self.queue_cap + 1)), listed))
        self.create(Percept("queue_e", list(range(self.queue_cap + 1)), listed))
        self.create(Percept("queue_w", list(range(self.queue_cap + 1)), listed))
        self.create(Percept("time", list(range(self.horizon + 1)), listed))
        self.create(Percept("yellow", list(range(self.yellow_duration + 1)), listed))
        self.create(Percept("elapsed_green_bucket", [0, 1, 2], listed))
        self.create(Percept("pressure_bucket", [-2, -1, 0, 1, 2], listed))

        self.possible_starts = {
            "phase": ["NS"],
            "queue_n": [0, 1, 2],
            "queue_s": [0, 1, 2],
            "queue_e": [0, 1, 2],
            "queue_w": [0, 1, 2],
            "time": [0],
            "yellow": [0],
            "elapsed_green_bucket": [0],
            "pressure_bucket": [0],
        }


    # Transition function used for learning (model-based)
    def control_transition(self, state: dict, action: str):
        phase = state["phase"]
        qn = int(state["queue_n"])  # type: ignore[arg-type]
        qs = int(state["queue_s"])  # type: ignore[arg-type]
        qe = int(state["queue_e"])  # type: ignore[arg-type]
        qw = int(state["queue_w"])  # type: ignore[arg-type]
        t = int(state["time"])       # type: ignore[arg-type]
        yellow = int(state["yellow"])  # type: ignore[arg-type]

        curr_state = {
            "phase": phase,
            "queue_n": qn,
            "queue_s": qs,
            "queue_e": qe,
            "queue_w": qw,
            "time": t,
            "yellow": yellow,
            "last_switch_time": int(state.get("last_switch_time", self.last_switch_time)),
        }
        lam_map = {d: self.arr_lambda for d in DIRECTIONS}
        new_state, reward, _, _ = transition_step(
            curr_state,
            action,
            self.arr_rng,
            lam_map,
            min_green=self.min_green,
            service_rate=self.service_rate,
            queue_cap=self.queue_cap,
            yellow_duration=self.yellow_duration,
            switch_penalty=self.switch_penalty,
        )

        terminated = int(new_state["time"]) >= self.horizon
        return new_state, reward, terminated

    @action(listed, ("hold", "N", "S", "E", "W", "NS", "EW"), control_transition)
    def control(self, agt, option: str):
        phase = self.get(Percept("phase", Any))
        qn = self.get(Percept("queue_n", Any))
        qs = self.get(Percept("queue_s", Any))
        qe = self.get(Percept("queue_e", Any))
        qw = self.get(Percept("queue_w", Any))
        t = self.get(Percept("time", Any))
        yellow = self.get(Percept("yellow", Any))
        elapsed_green_bucket = self.get(Percept("elapsed_green_bucket", Any))
        pressure_bucket = self.get(Percept("pressure_bucket", Any))
        assert isinstance(phase, Percept)
        assert isinstance(qn, Percept) and isinstance(qs, Percept)
        assert isinstance(qe, Percept) and isinstance(qw, Percept)
        assert isinstance(t, Percept)
        assert isinstance(yellow, Percept)
        assert isinstance(elapsed_green_bucket, Percept)
        assert isinstance(pressure_bucket, Percept)

        curr_state = {
            "phase": phase.args,
            "queue_n": qn.args,
            "queue_s": qs.args,
            "queue_e": qe.args,
            "queue_w": qw.args,
            "time": t.args,
            "yellow": yellow.args,
            "last_switch_time": self.last_switch_time,
        }
        # Enforce a minimum green time to avoid flicker
        desired_option = option
        if desired_option != "hold":
            t_now = int(t.args)
            if t_now - self.last_switch_time < self.min_green:
                desired_option = "hold"
        if int(yellow.args) > 0:
            desired_option = "hold"

        new_state, _, _ = self.control_transition(curr_state, desired_option)

        # Update last switch time if phase changed
        if new_state["phase"] != phase.args:
            self.last_switch_time = int(new_state["time"])

        self.print(
            f"{agt} | green={phase.args} | action={option} | t={t.args} | "
            f"queues N={qn.args}, S={qs.args}, E={qe.args}, W={qw.args}"
        )
        self.change(phase, new_state["phase"])  
        self.change(qn, int(new_state["queue_n"]))  
        self.change(qs, int(new_state["queue_s"]))  
        self.change(qe, int(new_state["queue_e"]))  
        self.change(qw, int(new_state["queue_w"]))  
        self.change(t, int(new_state["time"]))        
        self.change(yellow, int(new_state["yellow"]))  
        self.change(elapsed_green_bucket, int(new_state["elapsed_green_bucket"]))
        self.change(pressure_bucket, int(new_state["pressure_bucket"]))
        self.print(
            f"{agt} | next t={new_state['time']} | next green={new_state['phase']} | "
            f"queues N={new_state['queue_n']}, S={new_state['queue_s']}, "
            f"E={new_state['queue_e']}, W={new_state['queue_w']}"
        )

        self._update_metrics(
            new_state["phase"],
            int(qn.args),
            int(qs.args),
            int(qe.args),
            int(qw.args),
            int(new_state["queue_n"]),
            int(new_state["queue_s"]),
            int(new_state["queue_e"]),
            int(new_state["queue_w"]),
            int(new_state["time"]),
        )

        # Stop the system when horizon is reached to avoid infinite execution
        if int(new_state["time"]) >= self.horizon:
            Admin().stop_all_agents()

    def _served_counts(self, phase: str, qn: int, qs: int, qe: int, qw: int):
        return served_counts(phase, qn, qs, qe, qw, self.service_rate)

    def _phase_allows(self, phase: str, direction: str) -> bool:
        return phase_allows(phase, direction)

    def _poisson(self, lam: float) -> int:
        l = pow(2.718281828, -lam)
        k = 0
        p = 1.0
        while p > l:
            k += 1
            p *= self.arr_rng.random()
        return max(0, k - 1)

    def set_arrival_seed(self, seed: int):
        self.arr_rng = random.Random(seed)

    def _update_metrics(
        self,
        phase: str,
        prev_n: int,
        prev_s: int,
        prev_e: int,
        prev_w: int,
        next_n: int,
        next_s: int,
        next_e: int,
        next_w: int,
        t: int,
    ):
        served_n, served_s, served_e, served_w = self._served_counts(
            phase, prev_n, prev_s, prev_e, prev_w
        )
        self.metrics["throughput"]["N"] += served_n
        self.metrics["throughput"]["S"] += served_s
        self.metrics["throughput"]["E"] += served_e
        self.metrics["throughput"]["W"] += served_w

        queue_sum = next_n + next_s + next_e + next_w
        self.metrics["queue_sum_total"] += queue_sum
        self.metrics["step_count"] += 1
        self.metrics["steps"].append(
            {
                "time": t,
                "phase": phase,
                "queue_sum": queue_sum,
                "q_n": next_n,
                "q_s": next_s,
                "q_e": next_e,
                "q_w": next_w,
                "served_n": served_n,
                "served_s": served_s,
                "served_e": served_e,
                "served_w": served_w,
            }
        )

    def save_metrics(self, filepath: str):
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "time",
                    "phase",
                    "queue_sum",
                    "q_n",
                    "q_s",
                    "q_e",
                    "q_w",
                    "served_n",
                    "served_s",
                    "served_e",
                    "served_w",
                ],
            )
            writer.writeheader()
            for row in self.metrics["steps"]:
                writer.writerow(row)


class TrafficLight(Agent):
    def __init__(self, my_name: str, direction: str, controller: bool = False):
        super().__init__(my_name)
        self.direction = direction
        # only one controller runs the policy to avoid conflicting actions
        self.auto_action = controller

    @pl(gain, Belief("status", Any))
    def on_neighbor_status(self, src, status):
        self.print(f"Received neighbor status: {status}")

    @pl(
        gain,
        Goal("broadcast"),
        [
            Belief("phase", Any),
            Belief("queue_n", Any),
            Belief("queue_s", Any),
            Belief("queue_e", Any),
            Belief("queue_w", Any),
        ],
    )
    def broadcast(self, src, phase, qn, qs, qe, qw):
        # beliefs come from perceived environment (source is env name)
        try:
            env_name = getattr(phase, "source", None)
            if env_name is None:
                env_name = next(iter(self._environments.keys()), "unknown")

            def _val(x):
                return x.args if hasattr(x, "args") else x

            payload = (env_name, _val(phase), _val(qn), _val(qs), _val(qe), _val(qw))
            self.send(broadcast, tell, Belief("status", payload), "TrafficNet")
            self.wait(1)
            self.add(Goal("broadcast"))
        except Exception as e:
            self.print(f"Broadcast error: {e}")


if __name__ == "__main__":
    env = Intersection("I1")

    model = EnvModel(env)
    model.learn(qlearning, num_episodes=1500, max_steps=50, epsilon=1.0, final_epsilon=0.1)

    # 4 semaphores (one controller + three observers)
    tl_n = TrafficLight("TL_N", "N", controller=True)
    tl_s = TrafficLight("TL_S", "S")
    tl_e = TrafficLight("TL_E", "E")
    tl_w = TrafficLight("TL_W", "W")

    tl_n.add_policy(model)
    for ag in (tl_n, tl_s, tl_e, tl_w):
        ag.add(Goal("broadcast"))

    net = Channel("TrafficNet")
    Admin().connect_to([tl_n, tl_s, tl_e, tl_w], [env, net])
    Admin().start_system()


