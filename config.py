import os

SEED = 123
# Perfil padrao para TCC em PC pessoal (leve, mas ainda comparavel)
# 3 seeds ja permite variabilidade minima sem ficar pesado.
SEEDS = [SEED + i for i in range(3)]

# Treinamento / avaliacao
TRAIN_EPISODES = 5
TRAIN_EPISODES_LONG = 220
# Passos por episodio na avaliacao / series temporais (metrics_*.csv, graficos alinhados).
HORIZON = 25

 # Visualizacao (nao afeta os experimentos)
VISUAL_DELAY = 1.5
CYCLE_SPEED = 0.5

# Ambiente (dinamica e limites)
QUEUE_CAP = 5
SERVICE_RATE = 2
ARR_MIN = 0
ARR_MAX = 2
MIN_GREEN = 3

# Recompensa / shaping
SWITCH_PENALTY = 0.5
QUEUE_PENALTY_W = 0.1
PRIORITY_BONUS = 0.5
PRIORITY_PENALTY = 0.2

# Reward unificado (MASPY e baseline)
REWARD_THROUGHPUT_W = 1.0
REWARD_QUEUE_LINEAR_W = 0.20
REWARD_QUEUE_QUADRATIC_W = 0.03
REWARD_MAX_QUEUE_W = 0.15

# SARSA
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

# Schedulers unificados de exploracao e taxa de aprendizado
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_K = 4.0

ALPHA_START = 0.20
ALPHA_END = 0.03
ALPHA_DECAY_K = 3.0

# Comparacao pareada (mesma sequencia de chegadas)
PAIRED_ARRIVALS = True
ARRIVAL_SEED = 999

# TensorBoard
# Desligado por padrao para reduzir overhead em execucoes de comparacao.
ENABLE_TENSORBOARD = False
TB_LOG_DIR = "runs"

# Pygame (renderizacao)
# Render desligado por padrao para manter benchmark justo e reprodutivel.
RENDER_PYGAME = False

# Replay (quantos episodios gravar nos CSVs)
REPLAY_EPISODES = 1

# Convergencia (deteccao por janela movel em recompensa por episodio)
CONV_WINDOW = 20
CONV_PATIENCE = 6
CONV_STD_TOL = 1.0
CONV_SLOPE_TOL = 0.25

# Adaptacao a choque de demanda
ADAPT_SHOCK_STEP = 12
ADAPT_SHOCK_FACTOR = 2.0
ADAPT_SHOCK_DIRECTION = "N"
ADAPT_EVAL_STEPS = 30
ADAPT_RECOVERY_WINDOW = 5
ADAPT_RECOVERY_RATIO = 0.9

# Ablation curto de gamma
ABLATION_GAMMAS = [0.9, 0.96, 0.98]
ABLATION_EPISODES = 200
ABLATION_MIN_THROUGHPUT_DROP = 0.0

# Saidas (runs)
RUNS_BASE = "runs_output"


def _env_int(name: str, default: int) -> int:
	raw = os.getenv(name)
	if raw is None or raw == "":
		return default
	return int(raw)


def _env_float(name: str, default: float) -> float:
	raw = os.getenv(name)
	if raw is None or raw == "":
		return default
	return float(raw)


def _env_bool(name: str, default: bool) -> bool:
	raw = os.getenv(name)
	if raw is None:
		return default
	return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_list(name: str, default: list[float]) -> list[float]:
	raw = os.getenv(name)
	if raw is None or raw.strip() == "":
		return default
	return [float(p.strip()) for p in raw.split(",") if p.strip()]


def _env_int_list(name: str, default: list[int]) -> list[int]:
	raw = os.getenv(name)
	if raw is None or raw.strip() == "":
		return default
	return [int(p.strip()) for p in raw.split(",") if p.strip()]


# Modo leve para execucoes em maquinas com menos recursos.
# Ative com: RUN_LIGHT=1 (ex.: via run_all.sh --light)
RUN_LIGHT = _env_bool("RUN_LIGHT", False)

if RUN_LIGHT:
	SEEDS = _env_int_list("LIGHT_SEEDS", [SEED])
	TRAIN_EPISODES_LONG = _env_int("LIGHT_TRAIN_EPISODES_LONG", 120)
	HORIZON = _env_int("LIGHT_HORIZON", 8)
	REPLAY_EPISODES = _env_int("LIGHT_REPLAY_EPISODES", 1)
	ENABLE_TENSORBOARD = _env_bool("LIGHT_ENABLE_TENSORBOARD", False)
	RENDER_PYGAME = _env_bool("LIGHT_RENDER_PYGAME", False)

	CONV_WINDOW = _env_int("LIGHT_CONV_WINDOW", 10)
	CONV_PATIENCE = _env_int("LIGHT_CONV_PATIENCE", 3)

	ADAPT_EVAL_STEPS = _env_int("LIGHT_ADAPT_EVAL_STEPS", 20)
	ADAPT_RECOVERY_WINDOW = _env_int("LIGHT_ADAPT_RECOVERY_WINDOW", 4)

	ABLATION_EPISODES = _env_int("LIGHT_ABLATION_EPISODES", 80)
	ABLATION_GAMMAS = _env_float_list("LIGHT_ABLATION_GAMMAS", [0.9, 0.96, 0.98])
	ABLATION_MIN_THROUGHPUT_DROP = _env_float("LIGHT_ABLATION_MIN_THROUGHPUT_DROP", ABLATION_MIN_THROUGHPUT_DROP)
