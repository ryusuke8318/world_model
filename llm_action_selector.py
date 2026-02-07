# -*- coding: utf-8 -*-
"""
llm_action_selector.py
======================
OpenRouter API 経由で LLM（Gemma / Mistral Small 等）を呼び出し、
simple_tag_v3 環境のエージェント行動（離散 0-4）を直接選択するモジュール。

既存の MATWMAgent.select_action() を置き換えるドロップイン設計。

使い方
------
    from llm_action_selector import LLMActionSelector, LLMConfig

    cfg = LLMConfig(
        api_key="sk-or-v1-xxxx",                       # OpenRouter APIキー
        model="google/gemma-3-4b-it:free",              # モデル指定
    )
    selector = LLMActionSelector("adversary", "adversary_0", cfg)
    action   = selector.select_action(observation)      # int 0-4

必要パッケージ
--------------
    pip install requests numpy
"""

from __future__ import annotations

import json
import re
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logger = logging.getLogger(__name__)


# ====================================================================
# 1. 設定
# ====================================================================

@dataclass
class LLMConfig:
    """LLM 行動選択の設定"""

    # --- OpenRouter 接続 ---
    api_key: str = ""                                        # 環境変数から渡しても可
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "google/gemma-3-4b-it:free"                 # ★デフォルト: Gemma 3 4B (無料枠)
    temperature: float = 0.1                                 # 低めで安定行動
    max_tokens: int = 8                                      # 数字1桁で十分
    timeout: float = 15.0                                    # 秒

    # --- フォールバック ---
    fallback_to_rule: bool = True          # LLM 失敗時 → ルールベース
    max_retries: int = 2

    # --- キャッシュ ---
    use_cache: bool = True
    cache_precision: int = 1               # 丸め小数桁
    cache_max_size: int = 5000

    # --- ログ ---
    verbose: bool = False                  # 詳細ログ

    # --- 推奨モデル一覧（参考） ---
    # "google/gemma-3-4b-it:free"                            … 無料・軽量
    # "google/gemma-3-12b-it:free"                           … 無料・中量
    # "mistralai/mistral-small-3.1-24b-instruct:free"        … 無料・高品質
    # "google/gemma-3-27b-it:free"                           … 無料・大型


# ====================================================================
# 2. 観測パーサー
# ====================================================================

# --- simple_tag_v3 の観測ベクトル構造 ---
_ADV_MAP = {                     # adversary: 16 次元
    "self_vel":        (0,  2),  # 速度 (vx, vy)
    "self_pos":        (2,  4),  # 位置 (x, y)
    "landmark0_rel":   (4,  6),  # 障害物0 相対位置
    "landmark1_rel":   (6,  8),  # 障害物1 相対位置
    "other0_rel":      (8,  10), # 他エージェント0 相対位置
    "other1_rel":      (10, 12), # 他エージェント1 相対位置
    "other2_rel":      (12, 14), # 他エージェント2 相対位置
    "prey_rel":        (14, 16), # 逃走者 相対位置
}

_PREY_MAP = {                    # agent_0: 14 次元 (→16 にパディング)
    "self_vel":        (0,  2),
    "self_pos":        (2,  4),
    "landmark0_rel":   (4,  6),
    "landmark1_rel":   (6,  8),
    "adv0_rel":        (8,  10), # adversary_0
    "adv1_rel":        (10, 12), # adversary_1
    "adv2_rel":        (12, 14), # adversary_2
}


def _parse_obs(obs: np.ndarray, role: str) -> Dict[str, Any]:
    """観測ベクトル → 構造化辞書 + 戦略分析"""
    m = _ADV_MAP if role == "adversary" else _PREY_MAP
    p: Dict[str, Any] = {}
    for k, (s, e) in m.items():
        if e <= len(obs):
            p[k] = (round(float(obs[s]), 3), round(float(obs[s + 1]), 3))

    # ----------------------------------------------------------
    # adversary 用: 追跡戦略の分析
    # ----------------------------------------------------------
    if role == "adversary" and "prey_rel" in p:
        dx, dy = p["prey_rel"]
        p["prey_dist"] = round(float(np.hypot(dx, dy)), 3)

        # チームメイトの位置と距離を収集
        teammates = {}
        for k in ("other0_rel", "other1_rel", "other2_rel"):
            if k in p:
                teammates[k] = {
                    "rel": p[k],
                    "dist_to_me": round(float(np.hypot(*p[k])), 3),
                }
        p["teammates"] = teammates

        # 各チームメイトの prey への推定距離
        # (自分→prey) と (自分→teammate) から三角不等式で概算
        if "self_pos" in p and "prey_rel" in p:
            for tk, tv in teammates.items():
                # teammate の prey への相対位置 ≈ prey_rel - teammate_rel
                est_dx = dx - tv["rel"][0]
                est_dy = dy - tv["rel"][1]
                tv["est_prey_dist"] = round(float(np.hypot(est_dx, est_dy)), 3)

        # 挟み撃ち判定: チームメイトが prey の反対側にいるか
        encirclement = []
        for tk, tv in teammates.items():
            # チームメイトの方向と prey の方向が逆 → 挟み撃ち可能
            t_dx, t_dy = tv["rel"]
            # prey方向との内積（負なら反対側）
            dot = dx * t_dx + dy * t_dy
            if dot < 0 and tv.get("est_prey_dist", 999) < 1.0:
                encirclement.append(tk)
        p["encirclement_allies"] = encirclement

        # prey が近いか遠いかの状況判定
        if p["prey_dist"] < 0.2:
            p["situation"] = "VERY_CLOSE"
        elif p["prey_dist"] < 0.5:
            p["situation"] = "CLOSE"
        elif p["prey_dist"] < 1.0:
            p["situation"] = "MEDIUM"
        else:
            p["situation"] = "FAR"

        # 障害物が prey との間にあるか
        obstacles_blocking = []
        for lk in ("landmark0_rel", "landmark1_rel"):
            if lk in p:
                lx, ly = p[lk]
                l_dist = np.hypot(lx, ly)
                # 障害物が自分と prey の間にある判定
                # (障害物が prey より近く、かつ同じ方向)
                l_dot = lx * dx + ly * dy
                if 0 < l_dot and l_dist < p["prey_dist"] * 0.8:
                    obstacles_blocking.append(lk)
        p["obstacles_blocking"] = obstacles_blocking

    # ----------------------------------------------------------
    # prey 用: 逃走戦略の分析
    # ----------------------------------------------------------
    if role == "prey":
        adv_info = {}
        for k in ("adv0_rel", "adv1_rel", "adv2_rel"):
            if k in p:
                adv_dx, adv_dy = p[k]
                dist = float(np.hypot(adv_dx, adv_dy))
                angle = float(np.degrees(np.arctan2(adv_dy, adv_dx)))
                adv_info[k] = {"rel": p[k], "dist": round(dist, 3), "angle": round(angle, 1)}
        p["adv_info"] = adv_info

        if adv_info:
            # 最も近い追手
            closest_k = min(adv_info, key=lambda k: adv_info[k]["dist"])
            p["closest"] = closest_k
            p["closest_dist"] = adv_info[closest_k]["dist"]

            # 危険度: 近い追手の数
            danger_count = sum(1 for v in adv_info.values() if v["dist"] < 0.5)
            p["danger_count"] = danger_count

            if danger_count >= 2:
                p["situation"] = "SURROUNDED"
            elif p["closest_dist"] < 0.2:
                p["situation"] = "CRITICAL"
            elif p["closest_dist"] < 0.5:
                p["situation"] = "DANGER"
            elif p["closest_dist"] < 1.0:
                p["situation"] = "CAUTIOUS"
            else:
                p["situation"] = "SAFE"

            # 逃走経路分析: 4方向のうちどこが最も安全か
            escape_scores = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}  # left, right, down, up
            for v in adv_info.values():
                adv_dx, adv_dy = v["rel"]
                w = 1.0 / max(v["dist"], 0.05)  # 近いほど重み大
                # 各方向への逃走に対する追手の影響
                escape_scores[1] += w * max(0, -adv_dx)  # left(x-)  : 追手がx-方向→危険
                escape_scores[2] += w * max(0,  adv_dx)  # right(x+) : 追手がx+方向→危険
                escape_scores[3] += w * max(0, -adv_dy)  # down(y-)  : 追手がy-方向→危険
                escape_scores[4] += w * max(0,  adv_dy)  # up(y+)    : 追手がy+方向→危険
            # スコアが低い方向 = 安全
            p["escape_scores"] = escape_scores
            best_escape = min(escape_scores, key=escape_scores.get)
            p["best_escape"] = best_escape

            # 追手の間のギャップ分析
            angles = sorted([v["angle"] for v in adv_info.values()])
            if len(angles) >= 2:
                gaps = []
                for i in range(len(angles)):
                    next_i = (i + 1) % len(angles)
                    gap = angles[next_i] - angles[i]
                    if gap < 0:
                        gap += 360
                    mid_angle = angles[i] + gap / 2
                    gaps.append({"gap_deg": round(gap, 1), "mid_angle": round(mid_angle, 1)})
                p["gaps"] = sorted(gaps, key=lambda g: g["gap_deg"], reverse=True)

    return p


# ====================================================================
# 3. プロンプト生成
# ====================================================================

def _system_prompt(role: str, name: str) -> str:
    if role == "adversary":
        return (
            f"You are {name}, one of 3 CHASERS in a 2D tag game.\n"
            "\n"
            "GOAL: Cooperate with your 2 teammates to catch the prey.\n"
            "\n"
            "ACTIONS: 0=stay 1=left(x-) 2=right(x+) 3=down(y-) 4=up(y+)\n"
            "\n"
            "STRATEGY PRIORITIES (in order):\n"
            "1. ENCIRCLE: If a teammate is on the opposite side of the prey,\n"
            "   move to cut off the prey's escape route (flank from the side),\n"
            "   NOT directly toward the prey.\n"
            "2. INTERCEPT: If the prey is moving (has velocity), predict where\n"
            "   it will be and move to intercept, not just chase.\n"
            "3. SPREAD: If all chasers are on the same side of the prey,\n"
            "   one should go around to the other side to create a pincer.\n"
            "4. AVOID OBSTACLES: Do not move toward an obstacle between you\n"
            "   and the prey. Go around it.\n"
            "5. DIRECT CHASE: If none of the above apply, move toward the\n"
            "   prey along the axis with the larger offset.\n"
            "\n"
            "COORDINATE RULES:\n"
            "  dx>0 means target is to the RIGHT  -> action 2\n"
            "  dx<0 means target is to the LEFT   -> action 1\n"
            "  dy>0 means target is ABOVE          -> action 4\n"
            "  dy<0 means target is BELOW          -> action 3\n"
            "\n"
            "Reply with ONLY one digit 0-4. No explanation."
        )
    return (
        f"You are {name}, the sole EVADER in a 2D tag game.\n"
        "\n"
        "GOAL: Survive as long as possible against 3 coordinated chasers.\n"
        "\n"
        "ACTIONS: 0=stay 1=left(x-) 2=right(x+) 3=down(y-) 4=up(y+)\n"
        "\n"
        "STRATEGY PRIORITIES (in order):\n"
        "1. ESCAPE ENCIRCLEMENT: If surrounded (2+ chasers close),\n"
        "   find the widest gap between chasers and dash through it.\n"
        "   The analysis will show the best gap angle.\n"
        "2. AVOID CLOSEST THREAT: Move in the OPPOSITE direction\n"
        "   of the closest chaser.\n"
        "3. USE OBSTACLES: If an obstacle is between you and a chaser,\n"
        "   keep it there as a shield. Move to keep obstacles between you\n"
        "   and the threats.\n"
        "4. OPEN SPACE: Move toward the direction with fewest chasers.\n"
        "   The escape score analysis shows which direction is safest.\n"
        "5. PREDICT: If chasers are closing in from one direction,\n"
        "   change direction early before they arrive.\n"
        "\n"
        "COORDINATE RULES (OPPOSITE of chaser direction):\n"
        "  chaser dx>0 (right of me) -> action 1 (go LEFT)\n"
        "  chaser dx<0 (left of me)  -> action 2 (go RIGHT)\n"
        "  chaser dy>0 (above me)    -> action 3 (go DOWN)\n"
        "  chaser dy<0 (below me)    -> action 4 (go UP)\n"
        "\n"
        "Reply with ONLY one digit 0-4. No explanation."
    )


def _user_prompt(parsed: Dict[str, Any], role: str) -> str:
    lines = []

    # --- 自分の状態 ---
    if "self_pos" in parsed:
        lines.append(f"MY POS=({parsed['self_pos'][0]}, {parsed['self_pos'][1]})")
    if "self_vel" in parsed:
        vx, vy = parsed["self_vel"]
        if abs(vx) > 0.01 or abs(vy) > 0.01:
            lines.append(f"MY VEL=({vx}, {vy})")

    # ==============================================================
    # ADVERSARY（追跡者）
    # ==============================================================
    if role == "adversary":
        # ターゲット情報
        if "prey_rel" in parsed:
            dx, dy = parsed["prey_rel"]
            lines.append(f"PREY: rel=({dx}, {dy}) dist={parsed.get('prey_dist','?')}")

        # 状況判定
        sit = parsed.get("situation", "?")
        lines.append(f"SITUATION: {sit}")

        # チームメイトの位置と prey への推定距離
        teammates = parsed.get("teammates", {})
        if teammates:
            lines.append("TEAMMATES:")
            for tk, tv in teammates.items():
                est = tv.get("est_prey_dist", "?")
                lines.append(f"  {tk}: rel=({tv['rel'][0]}, {tv['rel'][1]}) "
                             f"prey_dist~{est}")

        # 挟み撃ち情報
        enc = parsed.get("encirclement_allies", [])
        if enc:
            lines.append(f"ENCIRCLEMENT: {', '.join(enc)} on opposite side of prey!")
            lines.append("-> Consider flanking (move perpendicular to prey) "
                         "to tighten the trap.")
        else:
            lines.append("NO ENCIRCLEMENT: Consider spreading out.")
            # チームメイトが全員同じ方向にいる場合のヒント
            if "prey_rel" in parsed and teammates:
                prey_dx, prey_dy = parsed["prey_rel"]
                same_side = all(
                    (prey_dx * tv["rel"][0] + prey_dy * tv["rel"][1]) > 0
                    for tv in teammates.values()
                )
                if same_side:
                    lines.append("-> All chasers on same side! "
                                 "You should go AROUND to the opposite side.")

        # 障害物警告
        blocking = parsed.get("obstacles_blocking", [])
        if blocking:
            lines.append(f"OBSTACLE WARNING: {', '.join(blocking)} blocking path to prey!")
            lines.append("-> Go around the obstacle, not through it.")

        # 障害物
        for k in ("landmark0_rel", "landmark1_rel"):
            if k in parsed and k not in blocking:
                lines.append(f"Obstacle {k}: ({parsed[k][0]}, {parsed[k][1]})")

        lines.append("Action?")

    # ==============================================================
    # PREY（逃走者）
    # ==============================================================
    else:
        # 状況判定
        sit = parsed.get("situation", "?")
        lines.append(f"SITUATION: {sit}")

        # 追跡者情報
        adv_info = parsed.get("adv_info", {})
        if adv_info:
            lines.append("CHASERS:")
            for k, v in adv_info.items():
                closest_mark = " <<< CLOSEST" if k == parsed.get("closest") else ""
                lines.append(f"  {k}: rel=({v['rel'][0]}, {v['rel'][1]}) "
                             f"dist={v['dist']} angle={v['angle']}deg{closest_mark}")

        # 危険度
        dc = parsed.get("danger_count", 0)
        if dc >= 2:
            lines.append(f"DANGER: {dc} chasers within close range!")

        # 逃走経路分析
        escape = parsed.get("escape_scores")
        if escape:
            best = parsed.get("best_escape", "?")
            action_names = {1: "LEFT", 2: "RIGHT", 3: "DOWN", 4: "UP"}
            # スコアが低い = 安全
            scored = sorted(escape.items(), key=lambda x: x[1])
            lines.append("ESCAPE ANALYSIS (lower=safer):")
            for act_id, score in scored:
                marker = " <<< SAFEST" if act_id == best else ""
                lines.append(f"  {act_id}={action_names[act_id]}: "
                             f"danger={score:.2f}{marker}")

        # ギャップ分析（包囲されている場合）
        gaps = parsed.get("gaps", [])
        if gaps and dc >= 2:
            best_gap = gaps[0]
            lines.append(f"WIDEST GAP: {best_gap['gap_deg']}deg "
                         f"at angle {best_gap['mid_angle']}deg")
            # ギャップ角度から行動を提案
            mid = best_gap["mid_angle"]
            if -45 <= mid < 45:
                lines.append("-> Gap is to the RIGHT (action 2)")
            elif 45 <= mid < 135:
                lines.append("-> Gap is ABOVE (action 4)")
            elif -135 <= mid < -45:
                lines.append("-> Gap is BELOW (action 3)")
            else:
                lines.append("-> Gap is to the LEFT (action 1)")

        # 障害物（盾として使える可能性）
        for k in ("landmark0_rel", "landmark1_rel"):
            if k in parsed:
                lx, ly = parsed[k]
                ldist = np.hypot(lx, ly)
                # 障害物が追手との間にあるかチェック
                shield = False
                if parsed.get("closest") and parsed["closest"] in adv_info:
                    c_rel = adv_info[parsed["closest"]]["rel"]
                    dot = lx * c_rel[0] + ly * c_rel[1]
                    if dot > 0 and ldist < adv_info[parsed["closest"]]["dist"]:
                        shield = True
                shield_txt = " [SHIELD - keep this between you and chaser!]" if shield else ""
                lines.append(f"Obstacle {k}: ({lx}, {ly}) dist={ldist:.2f}{shield_txt}")

        lines.append("Action?")

    return "\n".join(lines)


# ====================================================================
# 4. ルールベースフォールバック（LLM失敗時）
# ====================================================================

def _rule_action(parsed: Dict[str, Any], role: str) -> int:
    """戦略的ルールベース（LLMフォールバック用）"""

    # ==============================================================
    # ADVERSARY: 挟み撃ち優先 → 迂回 → 直進
    # ==============================================================
    if role == "adversary":
        if "prey_rel" not in parsed:
            return 0
        dx, dy = parsed["prey_rel"]
        prey_dist = parsed.get("prey_dist", 999)

        # 挟み撃ち成立中 → 横から詰める（垂直方向に移動）
        enc = parsed.get("encirclement_allies", [])
        if enc and prey_dist < 1.0:
            # prey方向に対して垂直に動く
            if abs(dx) >= abs(dy):
                # preyがx軸方向 → y軸方向（垂直）で詰める
                return 4 if dy > 0 else 3
            else:
                return 2 if dx > 0 else 1

        # 障害物が邪魔 → 迂回
        blocking = parsed.get("obstacles_blocking", [])
        if blocking:
            # prey方向の垂直方向に迂回
            if abs(dx) >= abs(dy):
                return 4 if dy >= 0 else 3
            else:
                return 2 if dx >= 0 else 1

        # 通常: 最大軸方向へ直進
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return 0
        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1
        else:
            return 4 if dy > 0 else 3

    # ==============================================================
    # PREY: ギャップ突破 → 逃走スコア → 最近追手の逆方向
    # ==============================================================
    else:
        situation = parsed.get("situation", "SAFE")

        # 包囲時: ギャップを狙う
        if situation == "SURROUNDED":
            gaps = parsed.get("gaps", [])
            if gaps:
                mid = gaps[0]["mid_angle"]
                if -45 <= mid < 45:
                    return 2   # right
                elif 45 <= mid < 135:
                    return 4   # up
                elif -135 <= mid < -45:
                    return 3   # down
                else:
                    return 1   # left

        # 逃走スコアがあれば最安全方向へ
        best = parsed.get("best_escape")
        if best is not None:
            return best

        # フォールバック: 最も近い追手の逆方向
        closest = parsed.get("closest")
        if closest is None or closest not in parsed:
            return 0
        dx, dy = parsed[closest]
        dx, dy = -dx, -dy  # 逆方向

        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return 0
        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1
        else:
            return 4 if dy > 0 else 3


# ====================================================================
# 5. LLMActionSelector 本体
# ====================================================================

class LLMActionSelector:
    """
    OpenRouter 経由で LLM を呼び出し、行動を選択する。
    MATWMAgent.select_action() と同じインターフェース。
    """

    def __init__(self, role: str, agent_name: str, config: LLMConfig | None = None):
        if not _HAS_REQUESTS:
            raise ImportError("requests が必要です: pip install requests")

        self.role = role
        self.agent_name = agent_name
        self.cfg = config or LLMConfig()
        self._sys_prompt = _system_prompt(role, agent_name)
        self._cache: OrderedDict[str, int] = OrderedDict()

        self.stats = {
            "calls": 0, "cache_hits": 0,
            "api_ok": 0, "api_fail": 0, "fallback": 0,
            "total_latency": 0.0,
        }

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """観測ベクトルから行動 (0-4) を返す。"""
        self.stats["calls"] += 1
        parsed = _parse_obs(obs, self.role)

        # キャッシュ
        if self.cfg.use_cache:
            key = self._cache_key(obs)
            if key in self._cache:
                self.stats["cache_hits"] += 1
                self._cache.move_to_end(key)
                return self._cache[key]

        # LLM 呼び出し
        user_msg = _user_prompt(parsed, self.role)
        action = self._call_llm(user_msg)

        if action is None:
            self.stats["fallback"] += 1
            action = _rule_action(parsed, self.role)
            if self.cfg.verbose:
                logger.warning("[%s] LLM failed -> rule fallback -> %d", self.agent_name, action)

        # キャッシュ書き込み
        if self.cfg.use_cache:
            self._cache[key] = action
            if len(self._cache) > self.cfg.cache_max_size:
                self._cache.popitem(last=False)

        return action

    def print_stats(self):
        s = self.stats
        hit_rate = s["cache_hits"] / max(s["calls"], 1) * 100
        avg_lat  = s["total_latency"] / max(s["api_ok"] + s["api_fail"], 1) * 1000
        print(f"[{self.agent_name}] calls={s['calls']}  "
              f"cache_hit={hit_rate:.0f}%  "
              f"api_ok={s['api_ok']}  api_fail={s['api_fail']}  "
              f"fallback={s['fallback']}  avg_latency={avg_lat:.0f}ms")

    # --- 内部 ---

    def _cache_key(self, obs: np.ndarray) -> str:
        return np.round(obs, self.cfg.cache_precision).tobytes()

    def _call_llm(self, user_msg: str) -> Optional[int]:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "messages": [
                {"role": "system", "content": self._sys_prompt},
                {"role": "user",   "content": user_msg},
            ],
        }

        for attempt in range(1, self.cfg.max_retries + 1):
            t0 = time.time()
            try:
                resp = requests.post(
                    self.cfg.base_url, headers=headers,
                    json=payload, timeout=self.cfg.timeout,
                )
                elapsed = time.time() - t0
                self.stats["total_latency"] += elapsed

                if resp.status_code != 200:
                    self.stats["api_fail"] += 1
                    if self.cfg.verbose:
                        logger.warning("[%s] HTTP %d: %s",
                                       self.agent_name, resp.status_code, resp.text[:200])
                    continue

                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                action = self._parse_action(text)

                if action is not None:
                    self.stats["api_ok"] += 1
                    if self.cfg.verbose:
                        logger.info("[%s] LLM -> '%s' -> action=%d (%.0fms)",
                                    self.agent_name, text, action, elapsed * 1000)
                    return action

                self.stats["api_fail"] += 1
                if self.cfg.verbose:
                    logger.warning("[%s] parse failed: '%s'", self.agent_name, text)

            except Exception as e:
                elapsed = time.time() - t0
                self.stats["total_latency"] += elapsed
                self.stats["api_fail"] += 1
                if self.cfg.verbose:
                    logger.warning("[%s] API error (attempt %d): %s",
                                   self.agent_name, attempt, e)
        return None

    @staticmethod
    def _parse_action(text: str) -> Optional[int]:
        text = text.strip()
        if text in ("0", "1", "2", "3", "4"):
            return int(text)
        m = re.match(r"(\d)", text)
        if m and 0 <= int(m.group(1)) <= 4:
            return int(m.group(1))
        m = re.search(r"\b([0-4])\b", text)
        if m:
            return int(m.group(1))
        return None


# ====================================================================
# 6. MATWMAgent 統合ヘルパー
# ====================================================================

def patch_agent_with_llm(agent, role: str, llm_config: LLMConfig) -> None:
    """
    既存の MATWMAgent の select_action を LLM に差し替える。

    使い方:
        patch_agent_with_llm(agents["adversary_0"], "adversary", cfg)
    """
    selector = LLMActionSelector(role, agent.agent_name, llm_config)

    def _llm_select(obs, deterministic=False):
        return selector.select_action(obs, deterministic)

    agent.select_action = _llm_select
    agent._llm_selector = selector


def create_llm_selectors(
    agent_names: list[str],
    llm_config: LLMConfig,
) -> Dict[str, LLMActionSelector]:
    """全エージェント分の LLMActionSelector を一括生成"""
    selectors = {}
    for name in agent_names:
        role = "prey" if name.startswith("agent") else "adversary"
        selectors[name] = LLMActionSelector(role, name, llm_config)
    return selectors


# ====================================================================
# 7. 動作確認用
# ====================================================================

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("LLM Action Selector - Unit Test")
    print("=" * 60)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("\n!  OPENROUTER_API_KEY not set.")
        print("   export OPENROUTER_API_KEY='sk-or-v1-xxxx'")
        print("   Testing rule-based fallback only.\n")

    # ルールベーステスト
    print("--- Rule-based fallback test ---")
    dummy_adv = np.array([
        0.1, 0.0, 0.0, 0.0, 0.5, 0.3, -0.3, 0.2,
        0.2, 0.1, -0.1, 0.3, 0.0, -0.2, 0.4, -0.2,
    ], dtype=np.float32)

    parsed = _parse_obs(dummy_adv, "adversary")
    print(f"  Adversary: prey_rel={parsed['prey_rel']}, dist={parsed['prey_dist']}")
    print(f"  Rule action = {_rule_action(parsed, 'adversary')}  (expect 2=right)")

    dummy_prey = np.array([
        0.0, 0.0, 0.0, 0.0, 0.5, 0.3, -0.3, 0.2,
        -0.3, 0.1, 0.1, 0.5, 0.0, -0.6, 0.0, 0.0,
    ], dtype=np.float32)

    parsed_p = _parse_obs(dummy_prey, "prey")
    print(f"  Prey: closest={parsed_p.get('closest')}, dist={parsed_p.get('closest_dist')}")
    print(f"  Rule action = {_rule_action(parsed_p, 'prey')}  (expect 2=right)")

    # LLM テスト
    if api_key:
        print("\n--- LLM API test ---")
        cfg = LLMConfig(api_key=api_key, verbose=True)
        sel = LLMActionSelector("adversary", "adversary_0", cfg)
        action = sel.select_action(dummy_adv)
        print(f"  adversary_0 LLM action = {action}")
        sel.print_stats()

    print("\nDone.")
