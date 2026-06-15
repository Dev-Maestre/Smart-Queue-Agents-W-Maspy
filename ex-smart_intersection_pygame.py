# -*- coding: latin-1 -*-
from __future__ import annotations

import argparse
import csv
import random
import threading
import time
from pathlib import Path
from typing import Literal

from maspy import Admin, Channel, Goal
from maspy.learning import EnvModel, qlearning
from maspy.environment import Percept

# Reutiliza as classes do exemplo original (nao executa o main ao importar)
from ex_smart_intersection import Intersection, TrafficLight, DIRECTIONS, PHASES
from config import SEED, TRAIN_EPISODES, HORIZON, VISUAL_DELAY, CYCLE_SPEED, PAIRED_ARRIVALS, ARRIVAL_SEED, RENDER_PYGAME
from run_utils import get_run_dir, write_run_readme

# Carregado sob demanda (replay / janela); headless nao exige pygame instalado.
pygame = None


def _ensure_pygame():
    global pygame
    if pygame is not None:
        return
    try:
        import pygame as pg
    except ImportError as e:
        raise SystemExit(
            "Pygame nao esta instalado. Ative o venv e execute: pip install pygame"
        ) from e
    pygame = pg

QUEUE_KEYS = {"N": "queue_n", "S": "queue_s", "E": "queue_e", "W": "queue_w"}
CAR_SPEED = 1.4
# Desloca os carros para a "faixa da direita" em cada sentido, afastando do eixo dos semaforos.
CAR_LANE_OFFSET = 8
REPLAY_STEP_SECONDS = 0.15
DEFAULT_SERVICE_RATE = 2
METRICS_I1_CSV = "metrics_I1.csv"
METRICS_I2_CSV = "metrics_I2.csv"

RenderEnvChoice = Literal["both", "1", "2"]


class _Vehicle:
		def __init__(self, direction: str, speed: float = CAR_SPEED):
				self.direction = direction  # "NS" ou "EW"
				self.progress = 0.0         # 0..1 atraves do cruzamento
				self.speed = speed          # unidade/segundo (fra��o do percurso por segundo)


class _VisualIntersection:
		def __init__(self, env: Intersection | None, service_rate: int = DEFAULT_SERVICE_RATE):
				self.env = env
				self.service_rate = service_rate
				self.prev_state: dict | None = None
				self.crossing: dict[str, list[_Vehicle]] = {d: [] for d in DIRECTIONS}
				self.rewards: list[float] = []
				self.total_reward: float = 0.0
				self.avg_reward: float = 0.0

		def _spawn_crossing(self, curr_state: dict, yellow_active: bool):
				if yellow_active:
						return
				for d in DIRECTIONS:
						if f"served_{d.lower()}" in curr_state:
								served = int(curr_state.get(f"served_{d.lower()}", 0))
						else:
								service_rate = getattr(self.env, "service_rate", self.service_rate)
								phase = curr_state.get("phase")
								if not (phase == d or (phase == "NS" and d in {"N", "S"}) or (phase == "EW" and d in {"E", "W"})):
										continue
								served = min(service_rate, int(self.prev_state[QUEUE_KEYS[d]]))
						for _ in range(int(served)):
								self.crossing[d].append(_Vehicle(d))

		def update(self, state: dict, dt: float):
				# Detecta tick do ambiente pelo aumento de tempo
				if self.prev_state is None:
						self.prev_state = state.copy()
				elif state["time"] != self.prev_state["time"]:
						# Recompensa estimada do passo anterior -> atual
						self._append_reward(self.prev_state, state)

						# Veiculos atravessando conforme fase ativa (estimado pelo estado anterior)
						self._spawn_crossing(state, int(state.get("yellow", 0)) > 0)

						self.prev_state = state.copy()

				# Avan�a ve�culos que est�o cruzando
				for d in DIRECTIONS:
						for v in self.crossing[d][:]:
								v.progress += v.speed * dt
								if v.progress >= 1.0:
										self.crossing[d].remove(v)

		def _append_reward(self, prev_s: dict, curr_s: dict):
				try:
						# Inferir acao: se fase mudou -> troca, senao mantem
						action = "switch" if curr_s["phase"] != prev_s["phase"] else "hold"
						# Throughput: usa servidos do replay quando disponivel
						if "served_n" in curr_s:
								throughput = int(curr_s.get("served_n", 0)) + int(curr_s.get("served_s", 0)) + int(curr_s.get("served_e", 0)) + int(curr_s.get("served_w", 0))
						else:
								next_phase = curr_s["phase"]
								service_rate = getattr(self.env, "service_rate", self.service_rate)
								throughput = 0
								if next_phase in {"NS", "N"}:
										throughput += min(service_rate, int(prev_s[QUEUE_KEYS["N"]]))
								if next_phase in {"NS", "S"}:
										throughput += min(service_rate, int(prev_s[QUEUE_KEYS["S"]]))
								if next_phase in {"EW", "E"}:
										throughput += min(service_rate, int(prev_s[QUEUE_KEYS["E"]]))
								if next_phase in {"EW", "W"}:
										throughput += min(service_rate, int(prev_s[QUEUE_KEYS["W"]]))
						# Penalidade de fila usa as filas atuais (apos transicao)
						queue_sum = sum(int(curr_s[QUEUE_KEYS[d]]) for d in DIRECTIONS)
						queue_penalty = getattr(self.env, "queue_penalty_w", 0.1) * queue_sum
						switch_penalty = getattr(self.env, "switch_penalty", 0.5) if action == "switch" else 0.0
						r = float(throughput - queue_penalty - switch_penalty)
				except Exception:
						r = 0.0
				self.rewards.append(r)
				if len(self.rewards) > 200:
						self.rewards.pop(0)
				self.total_reward += r
				self.avg_reward = sum(self.rewards) / len(self.rewards) if self.rewards else 0.0


def read_intersection_state(env: Intersection) -> dict:
		# Usa snapshot seguro do ambiente
		info = env.get_info
		state = {
				"phase": None,
				"queue_n": 0,
				"queue_s": 0,
				"queue_e": 0,
				"queue_w": 0,
				"time": 0,
				"yellow": 0,
				"target_phase": None,
		}
		for prc in info["percepts"]:
				if not isinstance(prc, Percept):
						continue
				if prc.name == "phase":
						state["phase"] = prc.args  # "NS" ou "EW"
				elif prc.name == "queue_n":
						state["queue_n"] = int(prc.args)
				elif prc.name == "queue_s":
						state["queue_s"] = int(prc.args)
				elif prc.name == "queue_e":
						state["queue_e"] = int(prc.args)
				elif prc.name == "queue_w":
						state["queue_w"] = int(prc.args)
				elif prc.name == "time":
						state["time"] = int(prc.args)
				elif prc.name == "yellow":
						state["yellow"] = int(prc.args)
				elif prc.name == "target_phase":
						state["target_phase"] = prc.args
		return state


def _colors(phase: str | None):
		return {
				"bg": (25, 25, 30),
				"road": (60, 60, 70),
				"line": (100, 100, 120),
				"text": (230, 230, 230),
				"queue": (80, 180, 220),
				"car": (240, 200, 80),
				"light_green": (80, 200, 120),
				"light_red": (150, 70, 70),
				"phase": phase,
		}


def _draw_roads(surface, x, y, w, h, colors):
		pygame.draw.rect(surface, colors["bg"], (x, y, w, h))
		road_w = int(w * 0.25)
		road_h = int(h * 0.25)
		pygame.draw.rect(surface, colors["road"], (x + (w - road_w) // 2, y, road_w, h))
		pygame.draw.rect(surface, colors["road"], (x, y + (h - road_h) // 2, w, road_h))
		pygame.draw.rect(surface, colors["line"], (x, y, w, h), width=1)


def _draw_lights(surface, x, y, w, h, colors, yellow_active: bool):
		light_size = 16
		phase = colors["phase"]
		amber = (220, 170, 30)
		def light_color(d):
				if yellow_active:
						return amber
				if phase in {"NS", "N", "S"}:
						return colors["light_green"] if d in ("N", "S") and phase in {"NS", d} else colors["light_red"]
				if phase in {"EW", "E", "W"}:
						return colors["light_green"] if d in ("E", "W") and phase in {"EW", d} else colors["light_red"]
				return colors["light_red"]
		pygame.draw.circle(surface, light_color("N"), (x + w // 2, y + 18), light_size)
		pygame.draw.circle(surface, light_color("S"), (x + w // 2, y + h - 18), light_size)
		pygame.draw.circle(surface, light_color("W"), (x + 18, y + h // 2), light_size)
		pygame.draw.circle(surface, light_color("E"), (x + w - 18, y + h // 2), light_size)


def _draw_queues(surface, x, y, w, h, state, colors):
		bar = 10
		gap = 4
		qn = min(int(state.get("queue_n", 0)), 5)
		qs = min(int(state.get("queue_s", 0)), 5)
		qe = min(int(state.get("queue_e", 0)), 5)
		qw = min(int(state.get("queue_w", 0)), 5)

		for i in range(qn):
				bx = x + w // 2 - bar // 2
				by = y + 40 + i * (bar + gap)
				pygame.draw.rect(surface, colors["queue"], (bx, by, bar, bar), border_radius=3)
		for i in range(qs):
				bx = x + w // 2 - bar // 2
				by = y + h - 40 - i * (bar + gap)
				pygame.draw.rect(surface, colors["queue"], (bx, by, bar, bar), border_radius=3)
		for i in range(qw):
				bx = x + 40 + i * (bar + gap)
				by = y + h // 2 - bar // 2
				pygame.draw.rect(surface, colors["queue"], (bx, by, bar, bar), border_radius=3)
		for i in range(qe):
				bx = x + w - 40 - i * (bar + gap)
				by = y + h // 2 - bar // 2
				pygame.draw.rect(surface, colors["queue"], (bx, by, bar, bar), border_radius=3)


def _draw_vehicles(surface, x, y, w, h, vis, colors, yellow_active: bool):
		if vis is None:
				return
		if yellow_active:
				return
		center_x = x + w // 2
		center_y = y + h // 2
		car_w, car_h = 12, 18
		path_len = min(w, h) * 0.6
		off = CAR_LANE_OFFSET

		for v in vis.crossing["N"]:
				offset = (v.progress - 0.5) * path_len
				cy = int(center_y + offset)
				rect = pygame.Rect(center_x - car_w // 2 + off, cy - car_h // 2, car_w, car_h)
				pygame.draw.rect(surface, colors["car"], rect, border_radius=3)
		for v in vis.crossing["S"]:
				offset = (0.5 - v.progress) * path_len
				cy = int(center_y + offset)
				rect = pygame.Rect(center_x - car_w // 2 - off, cy - car_h // 2, car_w, car_h)
				pygame.draw.rect(surface, colors["car"], rect, border_radius=3)
		for v in vis.crossing["E"]:
				offset = (0.5 - v.progress) * path_len
				cx = int(center_x + offset)
				rect = pygame.Rect(cx - car_h // 2, center_y - car_w // 2 + off, car_h, car_w)
				pygame.draw.rect(surface, colors["car"], rect, border_radius=3)
		for v in vis.crossing["W"]:
				offset = (v.progress - 0.5) * path_len
				cx = int(center_x + offset)
				rect = pygame.Rect(cx - car_h // 2, center_y - car_w // 2 - off, car_h, car_w)
				pygame.draw.rect(surface, colors["car"], rect, border_radius=3)


def _draw_reward_overlay(surface, x, y, h, state, vis, colors):
		font = pygame.font.SysFont(None, 18)
		qs = [state.get(QUEUE_KEYS[d], 0) for d in DIRECTIONS]
		text = font.render(
				f"phase={state.get('phase')}  tp={state.get('target_phase')}  N={qs[0]} S={qs[1]} E={qs[2]} W={qs[3]}  t={state.get('time')}  y={state.get('yellow')}",
				True,
				colors["text"],
		)
		surface.blit(text, (x + 8, y + h - 24))
		if vis is None:
				return
		font2 = pygame.font.SysFont(None, 18)
		t2 = font2.render(
				f"reward(avg={vis.avg_reward:0.2f}  total={int(vis.total_reward)})",
				True,
				colors["text"],
		)
		surface.blit(t2, (x + 8, y + h - 44))
		if not vis.rewards:
				return
		base_y = y + h - 60
		bar_w = 2
		max_h = 30
		mx = max(1.0, max(abs(v) for v in vis.rewards))
		start_x = x + 8
		for i, val in enumerate(vis.rewards[-160:]):
				hh = int((abs(val) / mx) * max_h)
				color = (90, 200, 120) if val >= 0 else (200, 90, 90)
				rect = pygame.Rect(start_x + i * (bar_w + 1), base_y - (hh if val >= 0 else 0), bar_w, hh)
				pygame.draw.rect(surface, color, rect)


def draw_intersection(surface, x, y, w, h, state: dict, vis: _VisualIntersection | None = None):
		colors = _colors(state.get("phase"))
		yellow_active = int(state.get("yellow", 0)) > 0
		_draw_roads(surface, x, y, w, h, colors)
		_draw_lights(surface, x, y, w, h, colors, yellow_active)
		_draw_queues(surface, x, y, w, h, state, colors)
		_draw_vehicles(surface, x, y, w, h, vis, colors, yellow_active)
		_draw_reward_overlay(surface, x, y, h, state, vis, colors)


def run_pygame(i1: Intersection, i2: Intersection, render_env: RenderEnvChoice = "both"):
    _ensure_pygame()
    pygame.init()
    dual = render_env == "both"
    W, H = (900, 420) if dual else (480, 420)
    screen = pygame.display.set_mode((W, H))
    cap = "MASPY Smart Intersection - Pygame"
    if render_env == "1":
        cap += " (I1)"
    elif render_env == "2":
        cap += " (I2)"
    pygame.display.set_caption(cap)
    clock = pygame.time.Clock()

    vis1 = _VisualIntersection(i1, service_rate=getattr(i1, "service_rate", DEFAULT_SERVICE_RATE))
    vis2 = _VisualIntersection(i2, service_rate=getattr(i2, "service_rate", DEFAULT_SERVICE_RATE))

    info_font = pygame.font.SysFont(None, 18)

    system_started = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = clock.get_time() / 1000.0  # segundos desde o ultimo frame
        screen.fill((15, 15, 20))

        # Le estados atuais e desenha lado a lado
        s1 = read_intersection_state(i1)
        s2 = read_intersection_state(i2)

        # Atualiza animacoes com base nas mudancas de estado (servico/tempo)
        vis1.update(s1, dt)
        vis2.update(s2, dt)

        if dual:
            draw_intersection(screen, 60, 60, 360, 300, s1, vis1)
            draw_intersection(screen, 480, 60, 360, 300, s2, vis2)
        elif render_env == "1":
            draw_intersection(screen, 60, 60, 360, 300, s1, vis1)
        else:
            draw_intersection(screen, 60, 60, 360, 300, s2, vis2)

        # Overlay de informacao do treinamento
        info = info_font.render(f"trained episodes: {TRAIN_EPISODES}", True, (230, 230, 230))
        screen.blit(info, (12, 8))

        # Se o treinamento ainda esta em progresso, mostrar status
        if not TRAINING_STATUS.get("done"):
            status = TRAINING_STATUS.get("msg", "training...")
            msg = info_font.render(status, True, (230, 230, 230))
            screen.blit(msg, (12, 28))
        elif not system_started:
            system_started = True
            sys_thread = threading.Thread(target=lambda: Admin().start_system(), daemon=True)
            sys_thread.start()

        pygame.display.flip()
        clock.tick(30)

        # Encerra automaticamente ao atingir o horizonte
        t1_ok = int(s1.get("time", 0)) >= HORIZON
        t2_ok = int(s2.get("time", 0)) >= HORIZON
        if render_env == "both":
            horizon_done = t1_ok and t2_ok
        elif render_env == "1":
            horizon_done = t1_ok
        else:
            horizon_done = t2_ok
        if horizon_done:
            running = False

    Admin().stop_all_agents()
    run_dir = get_run_dir("pygame")
    write_run_readme(
        run_dir,
        {
            "type": "pygame",
            "train_episodes": TRAIN_EPISODES,
            "horizon": HORIZON,
            "paired_arrivals": PAIRED_ARRIVALS,
            "arrival_seed": ARRIVAL_SEED,
            "render": True,
        },
    )
    i1.save_metrics(str(run_dir / METRICS_I1_CSV))
    i2.save_metrics(str(run_dir / METRICS_I2_CSV))
    pygame.quit()


def _parse_csv_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "episode": int(r.get("episode", 0) or 0),
                    "time": int(r.get("time", 0)),
                    "phase": r.get("phase", "NS"),
                    "queue_n": int(r.get("q_n", r.get("queue_n", 0))),
                    "queue_s": int(r.get("q_s", r.get("queue_s", 0))),
                    "queue_e": int(r.get("q_e", r.get("queue_e", 0))),
                    "queue_w": int(r.get("q_w", r.get("queue_w", 0))),
                    "yellow": int(r.get("yellow", 0)),
                    "served_n": int(r.get("served_n", 0)),
                    "served_s": int(r.get("served_s", 0)),
                    "served_e": int(r.get("served_e", 0)),
                    "served_w": int(r.get("served_w", 0)),
                    "target_phase": r.get("target_phase") or None,
                }
            )
    return rows


class _ReplaySource:
    def __init__(self, rows: list[dict], step_seconds: float = REPLAY_STEP_SECONDS):
        self.rows = rows
        self.step_seconds = step_seconds
        self.idx = 0
        self.acc = 0.0
        self.pause_remaining = 0.0
        self.last_episode = rows[0].get("episode", 0) if rows else 0

    def current(self) -> dict:
        if not self.rows:
            return {
                "time": 0,
                "phase": "NS",
                "queue_n": 0,
                "queue_s": 0,
                "queue_e": 0,
                "queue_w": 0,
                "yellow": 0,
                "target_phase": None,
            }
        return self.rows[self.idx]

    def advance(self, dt: float):
        if not self.rows:
            return
        if self.pause_remaining > 0:
            self.pause_remaining = max(0.0, self.pause_remaining - dt)
            return
        self.acc += dt
        while self.acc >= self.step_seconds and self.idx < len(self.rows) - 1:
            self.acc -= self.step_seconds
            self.idx += 1
            curr_ep = self.rows[self.idx].get("episode", 0)
            if curr_ep != self.last_episode:
                self.last_episode = curr_ep
                self.pause_remaining = self.step_seconds * 4
                break

    def done(self) -> bool:
        return self.idx >= len(self.rows) - 1


def run_replay(
    csv_i1: Path,
    csv_i2: Path,
    step_seconds: float = REPLAY_STEP_SECONDS,
    render_env: RenderEnvChoice = "both",
):
    _ensure_pygame()
    pygame.init()
    dual = render_env == "both"
    W, H = (900, 420) if dual else (480, 420)
    screen = pygame.display.set_mode((W, H))
    cap = "MASPY Smart Intersection - Replay"
    if render_env == "1":
        cap += " (I1)"
    elif render_env == "2":
        cap += " (I2)"
    pygame.display.set_caption(cap)
    clock = pygame.time.Clock()
    info_font = pygame.font.SysFont(None, 18)

    rows1 = _parse_csv_rows(csv_i1) if render_env in ("both", "1") else []
    rows2 = _parse_csv_rows(csv_i2) if render_env in ("both", "2") else []
    src1 = _ReplaySource(rows1, step_seconds=step_seconds)
    src2 = _ReplaySource(rows2, step_seconds=step_seconds)
    vis1 = _VisualIntersection(None, service_rate=DEFAULT_SERVICE_RATE)
    vis2 = _VisualIntersection(None, service_rate=DEFAULT_SERVICE_RATE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = clock.get_time() / 1000.0
        screen.fill((15, 15, 20))

        if render_env in ("both", "1"):
            src1.advance(dt)
        if render_env in ("both", "2"):
            src2.advance(dt)
        s1 = src1.current()
        s2 = src2.current()

        if render_env in ("both", "1"):
            vis1.update(s1, dt)
        if render_env in ("both", "2"):
            vis2.update(s2, dt)

        if dual:
            draw_intersection(screen, 60, 60, 360, 300, s1, vis1)
            draw_intersection(screen, 480, 60, 360, 300, s2, vis2)
        elif render_env == "1":
            draw_intersection(screen, 60, 60, 360, 300, s1, vis1)
        else:
            draw_intersection(screen, 60, 60, 360, 300, s2, vis2)

        ep1 = s1.get("episode", 0)
        ep2 = s2.get("episode", 0)
        if dual:
            ep_txt = f"replay (CSV)  ep={max(ep1, ep2)}"
        elif render_env == "1":
            ep_txt = f"replay (CSV) I1  ep={ep1}"
        else:
            ep_txt = f"replay (CSV) I2  ep={ep2}"
        info = info_font.render(ep_txt, True, (230, 230, 230))
        screen.blit(info, (12, 8))

        pygame.display.flip()
        clock.tick(30)

        if render_env == "both":
            replay_done = src1.done() and src2.done()
        elif render_env == "1":
            replay_done = src1.done()
        else:
            replay_done = src2.done()
        if replay_done:
            running = False

    pygame.quit()


def _build_envs():
    i1 = Intersection("I1")
    i2 = Intersection("I2")
    i1.horizon = HORIZON
    i2.horizon = HORIZON
    if PAIRED_ARRIVALS:
        i1.set_arrival_seed(ARRIVAL_SEED)
        i2.set_arrival_seed(ARRIVAL_SEED)
    else:
        i1.set_arrival_seed(ARRIVAL_SEED)
        i2.set_arrival_seed(ARRIVAL_SEED + 1)
    i1._states["time"] = list(range(HORIZON + 1))
    i2._states["time"] = list(range(HORIZON + 1))
    return i1, i2


def _build_agents():
    a1 = TrafficLight("TL1_N", "N", controller=True)
    a2 = TrafficLight("TL1_S", "S")
    a3 = TrafficLight("TL1_E", "E")
    a4 = TrafficLight("TL1_W", "W")
    for ag in (a1, a2, a3, a4):
        ag.add(Goal("broadcast"))

    b1 = TrafficLight("TL2_N", "N", controller=True)
    b2 = TrafficLight("TL2_S", "S")
    b3 = TrafficLight("TL2_E", "E")
    b4 = TrafficLight("TL2_W", "W")
    for ag in (b1, b2, b3, b4):
        ag.add(Goal("broadcast"))

    return a1, a2, a3, a4, b1, b2, b3, b4


def _train_models(m1: EnvModel, m2: EnvModel, a1: TrafficLight, b1: TrafficLight):
    TRAINING_STATUS["msg"] = "training model I1..."
    m1.learn(qlearning, num_episodes=TRAIN_EPISODES, max_steps=HORIZON, epsilon=1.0, final_epsilon=0.1)
    TRAINING_STATUS["msg"] = "training model I2..."
    m2.learn(qlearning, num_episodes=TRAIN_EPISODES, max_steps=HORIZON, epsilon=1.0, final_epsilon=0.1)
    a1.add_policy(m1)
    b1.add_policy(m2)
    TRAINING_STATUS["done"] = True
    TRAINING_STATUS["msg"] = "training done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", type=str, default="")
    parser.add_argument("--replay-i1", type=str, default="")
    parser.add_argument("--replay-i2", type=str, default="")
    parser.add_argument("--replay-step", type=float, default=REPLAY_STEP_SECONDS)
    parser.add_argument(
        "--render-env",
        type=str,
        choices=("both", "1", "2"),
        default=None,
        help=(
            "Visualiza I1, I2 ou ambos. Em replay, se omitido: apenas I1 (uma janela). "
            "Na execucao live com Pygame, se omitido: I1 e I2 lado a lado."
        ),
    )
    args = parser.parse_args()
    replay_mode = bool(args.replay_dir or args.replay_i1 or args.replay_i2)
    if args.render_env is not None:
        render_env: RenderEnvChoice = args.render_env  # type: ignore[assignment]
    elif replay_mode:
        render_env = "1"
    else:
        render_env = "both"

    if replay_mode:
        base = Path(args.replay_dir) if args.replay_dir else None
        need_i1 = render_env in ("both", "1")
        need_i2 = render_env in ("both", "2")
        if need_i1 and not args.replay_i1 and not args.replay_dir:
            raise SystemExit("Replay I1: informe --replay-dir ou --replay-i1.")
        if need_i2 and not args.replay_i2 and not args.replay_dir:
            raise SystemExit("Replay I2: informe --replay-dir ou --replay-i2.")
        csv_i1 = Path(args.replay_i1) if args.replay_i1 else (base / METRICS_I1_CSV if base else Path())
        csv_i2 = Path(args.replay_i2) if args.replay_i2 else (base / METRICS_I2_CSV if base else Path())
        if need_i1 and not csv_i1.is_file():
            raise SystemExit(f"CSV nao encontrado: {csv_i1}")
        if need_i2 and not csv_i2.is_file():
            raise SystemExit(f"CSV nao encontrado: {csv_i2}")
        run_replay(csv_i1, csv_i2, step_seconds=max(0.02, args.replay_step), render_env=render_env)
        raise SystemExit(0)

    random.seed(SEED)
    # Instancia ambientes e ajusta horizonte manualmente (compatível com EnvironmentMultiton)
    i1, i2 = _build_envs()

    # Treina modelos (igual ao exemplo base)
    m1 = EnvModel(i1)
    m2 = EnvModel(i2)

    # 4 semaforos por intersecao (1 controlador + 3 observadores)
    a1, a2, a3, a4, b1, b2, b3, b4 = _build_agents()

    # Canal de comunicacao (opcional, como no exemplo original)
    net = Channel("TrafficNet")
    Admin().connect_to([a1, a2, a3, a4, b1, b2, b3, b4], [i1, i2, net])

    # Tornar a execucao observavel: pequena pausa por ciclo
    a1.delay = VISUAL_DELAY
    b1.delay = VISUAL_DELAY
    Admin().sys_settings(cycle_speed=CYCLE_SPEED)

    # Status compartilhado de treinamento
    TRAINING_STATUS = {"done": False, "msg": "training..."}

    if RENDER_PYGAME:
        tr_thread = threading.Thread(target=lambda: _train_models(m1, m2, a1, b1), daemon=True)
        tr_thread.start()
        # Pygame loop (starts system after training)
        run_pygame(i1, i2, render_env=render_env)
    else:
        # Execucao headless: gera CSVs sem renderizar
        _train_models(m1, m2, a1, b1)
        Admin().start_system()
        run_dir = get_run_dir("pygame")
        write_run_readme(
            run_dir,
            {
                "type": "pygame_headless",
                "train_episodes": TRAIN_EPISODES,
                "horizon": HORIZON,
                "paired_arrivals": PAIRED_ARRIVALS,
                "arrival_seed": ARRIVAL_SEED,
                "render": False,
            },
        )
        i1.save_metrics(str(run_dir / METRICS_I1_CSV))
        i2.save_metrics(str(run_dir / METRICS_I2_CSV))


