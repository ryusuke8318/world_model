# -*- coding: utf-8 -*-
"""
curiosity_reward.py
===================
非中央集権型 MARL における好奇心駆動の内発的報酬モジュール。

== 研究のモチベーション ==

1. 非中央集権型MARLにおける好奇心の必要性
   ─────────────────────────────────────
   非中央集権型（decentralized）MARL では各エージェントは自分の局所的観測のみ
   に基づいて行動を決定する。これにより以下の根本的な課題が生じる:

   (a) 部分観測性と信用割当の困難
       他エージェントの行動・意図が直接観測できないため、環境報酬の変動が
       「自分の行動の結果」なのか「他エージェントの行動の結果」なのか
       区別できない（Multi-Agent Credit Assignment Problem）。
       好奇心は「自分の World Model の予測が外れた」ことを検出することで、
       未知の他エージェント行動パターンの存在を間接的に示唆し、
       信用割当を補助する。

   (b) 探索の局所化（Exploration Locality）
       中央集権型では全体の状態空間を見渡して探索を誘導できるが、
       非中央集権型では各エージェントが局所的な情報だけで探索を決定する。
       これにより「全員が同じ局所最適に陥る」相関探索問題が発生する。
       好奇心は各エージェントに個別の探索動機を与え、
       エージェント間の探索行動を多様化させる。

   (c) 環境報酬の希薄性（Reward Sparsity）
       simple_tag のような協調追跡タスクでは、追手チームが獲物を捕まえる
       まで有意な報酬が得られない。好奇心報酬は環境の構造（状態遷移の
       予測可能性）に基づく密な報酬シグナルを提供し、報酬の希薄性を緩和する。

2. World Model ベースの好奇心の優位性
   ─────────────────────────────────
   MATWM はすでに環境ダイナミクスの World Model を訓練している。
   この World Model の「予測誤差」は追加コストなしで好奇心シグナルとして
   利用できる。具体的には:

   (a) 状態予測誤差（Dynamics Curiosity）
       World Model が予測した次状態 z_pred と実際の次状態 z_actual の乖離。
       →「予想外の状態遷移」を経験した = 探索的に価値が高い

   (b) 報酬予測誤差（Reward Curiosity）
       World Model が予測した報酬と実際の報酬の乖離。
       →「予想外の報酬」を経験した = 環境の未学習領域にいる

   (c) チームメイト予測誤差（Social Curiosity）★ 本研究の核心
       TeammatePredictor が予測した他エージェントの行動と実際の行動の乖離。
       →「予想外の協調/競争パターン」を経験した = 社会的探索の動機

       Social Curiosity は非中央集権型 MARL に固有の好奇心形態であり、
       単一エージェント RL の好奇心（RND, ICM 等）にはない概念である。
       他エージェントの行動は環境ダイナミクスの一部だが、学習可能な
       非定常要素であるため、社会的好奇心は「環境の非定常部分への適応」
       を促進する。

3. LLM による意味的好奇心（Semantic Curiosity）
   ────────────────────────────────────────────
   World Model の予測誤差は「量的な新規性」を捉えるが、
   「質的な意味のある新規性」は区別できない。例えば:
   - 数値ノイズによる微小な予測誤差 vs 戦略的に重要な新パターン
   - 偶然の逸脱 vs 再現性のある協調戦術

   LLM はトラジェクトリの文脈を理解し、「戦略的に意味のある探索」と
   「無意味なランダム変動」を区別できる。これを Semantic Curiosity と定義し、
   計算的好奇心（World Model ベース）を補完する。

4. 好奇心の減衰と過学習防止
   ───────────────────────
   好奇心報酬が一定のままだと、エージェントは新規性を追い求め続け
   タスク報酬を無視する「好奇心中毒」に陥る。本実装では:
   - 訪問カウントベースの減衰: 同じ状態領域を繰り返し訪問すると好奇心低下
   - World Model の学習進捗に連動した減衰: WM が正確になるにつれ好奇心低下
   - LLM による適応的重み調整: 探索フェーズ/活用フェーズの判断を委ねる


== 実装の構造 ==

CuriosityReward（計算型好奇心: World Model ベース、高速、毎ステップ）
    ├── DynamicsCuriosity    … 状態予測誤差
    ├── RewardCuriosity      … 報酬予測誤差
    └── SocialCuriosity      … チームメイト予測誤差 ★

LLMCuriosityEvaluator（意味的好奇心: LLM ベース、低頻度、エピソード単位）
    └── トラジェクトリの戦略的新規性を評価

CuriosityManager（統合管理）
    └── 計算型 + 意味的好奇心を統合し、最終的な内発的報酬を生成
"""

from __future__ import annotations

import json
import os
import re
import time
import datetime
import logging
import math
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
class CuriosityConfig:
    """好奇心報酬の設定"""

    # --- 計算型好奇心の重み ---
    dynamics_curiosity_weight: float = 1.0    # 状態予測誤差
    reward_curiosity_weight: float = 0.5      # 報酬予測誤差
    social_curiosity_weight: float = 2.0      # チームメイト予測誤差 ★重め
    # Social Curiosity を重くする理由:
    #   非中央集権型では他エージェントの行動が最大の不確実性源。
    #   この不確実性を積極的に探索することが協調学習の加速に直結する。

    # --- 好奇心の正規化 ---
    curiosity_normalize: bool = True           # 実行平均で正規化
    curiosity_ema_decay: float = 0.99          # 指数移動平均の減衰率

    # --- 好奇心の減衰 ---
    curiosity_decay_method: str = "adaptive"   # "fixed", "count", "adaptive"
    # fixed:    固定の減衰スケジュール (initial_weight → 0)
    # count:    訪問カウントベースの減衰
    # adaptive: World Model の学習進捗に連動
    curiosity_initial_weight: float = 1.0      # 初期の好奇心重み
    curiosity_min_weight: float = 0.1          # 最小好奇心重み
    curiosity_decay_steps: int = 10000         # fixed: この歩数で min_weight に到達

    # --- 状態空間の離散化（訪問カウント用） ---
    state_bin_resolution: float = 0.2          # 状態の離散化解像度

    # --- LLM 意味的好奇心 ---
    use_llm_curiosity: bool = True
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    llm_model: str = "google/gemma-3-4b-it:free"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_eval_every_n_episodes: int = 1        # N エピソードごとにLLM評価
    semantic_curiosity_weight: float = 0.5     # LLM 好奇心の重み

    # --- ログ ---
    verbose: bool = False
    log_dir: str = "llm_logs"


# ====================================================================
# 2. 観測パーサー
# ====================================================================

_ADV_MAP = {
    "self_vel":      (0,  2),
    "self_pos":      (2,  4),
    "landmark0_rel": (4,  6),
    "landmark1_rel": (6,  8),
    "other0_rel":    (8,  10),
    "other1_rel":    (10, 12),
    "other2_rel":    (12, 14),
    "prey_rel":      (14, 16),
}

_PREY_MAP = {
    "self_vel":      (0,  2),
    "self_pos":      (2,  4),
    "landmark0_rel": (4,  6),
    "landmark1_rel": (6,  8),
    "adv0_rel":      (8,  10),
    "adv1_rel":      (10, 12),
    "adv2_rel":      (12, 14),
}

ACTION_NAMES = {0: "stay", 1: "left", 2: "right", 3: "down", 4: "up"}


def _parse_obs(obs: np.ndarray, role: str) -> Dict[str, Any]:
    """観測ベクトル → 構造化辞書"""
    m = _ADV_MAP if role == "adversary" else _PREY_MAP
    p: Dict[str, Any] = {}
    for k, (s, e) in m.items():
        if e <= len(obs):
            p[k] = (round(float(obs[s]), 3), round(float(obs[s + 1]), 3))

    if role == "adversary" and "prey_rel" in p:
        dx, dy = p["prey_rel"]
        p["prey_dist"] = round(float(np.hypot(dx, dy)), 3)
        # チームメイトの prey 推定距離
        for k in ("other0_rel", "other1_rel", "other2_rel"):
            if k in p:
                tx, ty = p[k]
                est_dx = dx - tx
                est_dy = dy - ty
                p[f"{k}_prey_dist"] = round(float(np.hypot(est_dx, est_dy)), 3)

    if role == "prey":
        for k in ("adv0_rel", "adv1_rel", "adv2_rel"):
            if k in p:
                dx, dy = p[k]
                p[f"{k}_dist"] = round(float(np.hypot(dx, dy)), 3)
        dists = [p.get(f"{k}_dist", 999) for k in ("adv0_rel", "adv1_rel", "adv2_rel")]
        p["closest_adv_dist"] = round(min(dists), 3)

    return p


# ====================================================================
# 3. 計算型好奇心（World Model ベース）
# ====================================================================

class CuriosityReward:
    """
    World Model の予測誤差を利用した計算型好奇心報酬。

    3つの好奇心成分:
      1. Dynamics Curiosity: ||z_pred - z_actual||
      2. Reward Curiosity:   |r_pred - r_actual|
      3. Social Curiosity:   CE(teammate_pred, teammate_actual) ★

    World Model は既に訓練済み（または訓練中）であるため、
    追加のネットワークは不要。World Model の学習が進むにつれて
    予測精度が上がり、好奇心報酬は自然に減少する。
    """

    def __init__(self, config: CuriosityConfig, agent_name: str, agent_idx: int):
        self.config = config
        self.agent_name = agent_name
        self.agent_idx = agent_idx

        # 正規化用の実行統計
        self._dynamics_ema = RunningEMA(config.curiosity_ema_decay)
        self._reward_ema = RunningEMA(config.curiosity_ema_decay)
        self._social_ema = RunningEMA(config.curiosity_ema_decay)

        # 訪問カウント（状態空間の離散化）
        self._visit_counts: Dict[str, int] = defaultdict(int)

        # 好奇心重みの現在値
        self._current_weight = config.curiosity_initial_weight
        self._global_step = 0

        # 統計
        self.stats = {
            "dynamics_mean": 0.0,
            "reward_mean": 0.0,
            "social_mean": 0.0,
            "total_mean": 0.0,
            "weight": config.curiosity_initial_weight,
            "compute_count": 0,
        }

    def compute(
        self,
        world_model: nn.Module,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        other_actions: Dict[str, int],
        device: torch.device,
    ) -> Dict[str, float]:
        """
        1ステップの好奇心報酬を計算。

        Args:
            world_model: 訓練中の World Model
            obs: 現在の観測（パディング済み）
            action: 自エージェントの行動
            reward: 環境報酬
            next_obs: 次の観測（パディング済み）
            other_actions: 他エージェントの行動 {agent_name: action}
            device: torch device

        Returns:
            {
                "dynamics": float,    # 状態予測誤差
                "reward_cur": float,  # 報酬予測誤差
                "social": float,      # チームメイト予測誤差
                "total": float,       # 重み付き合計
                "weight": float,      # 現在の好奇心重み
            }
        """
        self._global_step += 1
        self.stats["compute_count"] += 1

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(device)

            # === (1) Dynamics Curiosity ===
            z, _ = world_model.encode(obs_t)
            z_actual, _ = world_model.encode(next_obs_t)

            scaled_action = action + self.agent_idx * 5  # action_dim=5
            action_t = torch.LongTensor([scaled_action]).to(device)
            z_pred, _ = world_model.predict_next(
                z.unsqueeze(1), action_t.unsqueeze(1)
            )
            z_pred = z_pred.squeeze(1)

            # 予測誤差: カテゴリカル分布間の距離
            dynamics_error = float(
                F.mse_loss(z_pred.reshape(-1), z_actual.reshape(-1)).item()
            )

            # === (2) Reward Curiosity ===
            reward_logits = world_model.predict_reward(z_pred)
            # two-hot decode で予測報酬を取得
            from matwm_implementation import symexp, two_hot_decode
            reward_dist = F.softmax(reward_logits, dim=-1)
            reward_pred = float(symexp(two_hot_decode(reward_dist)).item())
            reward_error = abs(reward_pred - reward)

            # === (3) Social Curiosity ★ ===
            social_error = 0.0
            social_count = 0
            teammate_logits = world_model.predict_teammates(z, self.agent_idx)
            
            # other_actions は {agent_name(str): action} 形式
            # predict_teammates は {agent_idx(int): logits} を返す
            # agent_name → agent_idx の変換マッピング
            name_to_idx = {
                'adversary_0': 0, 'adversary_1': 1,
                'adversary_2': 2, 'agent_0': 3,
            }
            other_actions_by_idx = {}
            for aname, aact in other_actions.items():
                aidx = name_to_idx.get(aname)
                if aidx is not None:
                    other_actions_by_idx[aidx] = aact
            
            for other_idx, logits in teammate_logits.items():
                actual_action = other_actions_by_idx.get(other_idx)
                if actual_action is not None:
                    target = torch.LongTensor([actual_action]).to(device)
                    # CrossEntropy で予測の外れ具合を測定
                    logits_2d = logits.reshape(-1, logits.shape[-1])[:1]
                    ce = F.cross_entropy(logits_2d, target)
                    social_error += float(ce.item())
                    social_count += 1

            if social_count > 0:
                social_error /= social_count

        # --- 正規化 ---
        if self.config.curiosity_normalize:
            dynamics_norm = self._dynamics_ema.normalize(dynamics_error)
            reward_norm = self._reward_ema.normalize(reward_error)
            social_norm = self._social_ema.normalize(social_error)
        else:
            dynamics_norm = dynamics_error
            reward_norm = reward_error
            social_norm = social_error

        # --- 訪問カウントベースの減衰 ---
        visit_bonus = 1.0
        if self.config.curiosity_decay_method == "count":
            state_key = self._discretize_state(obs)
            self._visit_counts[state_key] += 1
            count = self._visit_counts[state_key]
            visit_bonus = 1.0 / math.sqrt(count)

        # --- 好奇心重みの更新 ---
        self._update_weight(dynamics_error)

        # --- 重み付き合計 ---
        total = (
            self.config.dynamics_curiosity_weight * dynamics_norm +
            self.config.reward_curiosity_weight * reward_norm +
            self.config.social_curiosity_weight * social_norm
        ) * self._current_weight * visit_bonus

        # 統計更新
        self.stats["dynamics_mean"] = self._dynamics_ema.mean
        self.stats["reward_mean"] = self._reward_ema.mean
        self.stats["social_mean"] = self._social_ema.mean
        self.stats["total_mean"] = (
            self.stats["total_mean"] * 0.99 + total * 0.01
        )
        self.stats["weight"] = self._current_weight

        return {
            "dynamics": dynamics_norm,
            "reward_cur": reward_norm,
            "social": social_norm,
            "total": total,
            "weight": self._current_weight,
        }

    def _update_weight(self, dynamics_error: float):
        """好奇心重みを更新"""
        cfg = self.config
        if cfg.curiosity_decay_method == "fixed":
            # 線形減衰
            progress = min(1.0, self._global_step / max(1, cfg.curiosity_decay_steps))
            self._current_weight = (
                cfg.curiosity_initial_weight * (1 - progress) +
                cfg.curiosity_min_weight * progress
            )
        elif cfg.curiosity_decay_method == "adaptive":
            # World Model の予測精度に連動:
            # dynamics_error が小さい = WM が正確 → 好奇心を下げる
            # dynamics_error が大きい = WM が不正確 → 好奇心を維持
            if self._dynamics_ema.count > 10:
                # 正規化された誤差が 1.0 付近 = 平均的
                # 1.0 より大きい = まだ学習が必要 → 好奇心維持
                # 1.0 より小さい = 学習済み → 好奇心低下
                ratio = self._dynamics_ema.normalize(dynamics_error)
                target = cfg.curiosity_initial_weight * min(1.0, max(0.0, ratio))
                target = max(target, cfg.curiosity_min_weight)
                # 緩やかに追従
                self._current_weight = 0.995 * self._current_weight + 0.005 * target
        # "count" は compute() 内で visit_bonus として処理済み

    def _discretize_state(self, obs: np.ndarray) -> str:
        """状態を離散化してカウントキーを生成"""
        binned = np.round(obs / self.config.state_bin_resolution).astype(int)
        return binned.tobytes()


class RunningEMA:
    """指数移動平均による正規化"""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def normalize(self, x: float) -> float:
        self.count += 1
        self.mean = self.decay * self.mean + (1 - self.decay) * x
        self.var = self.decay * self.var + (1 - self.decay) * (x - self.mean) ** 2
        std = max(math.sqrt(self.var), 1e-8)
        return (x - self.mean) / std + 1.0  # 1.0 中心に正規化


# ====================================================================
# 4. LLM 意味的好奇心
# ====================================================================

class LLMCuriosityEvaluator:
    """
    LLM を用いた意味的好奇心の評価。

    計算型好奇心（World Model ベース）は「量的な予測誤差」を捉えるが、
    以下を区別できない:
      - ノイズによる予測誤差 vs 戦略的に重要な新パターン
      - 偶然の行動逸脱 vs 再現可能な協調戦術の発見

    LLM はトラジェクトリの文脈を理解し、以下を評価する:
      1. 戦略的新規性: 新しい協調パターンや競争戦術の発見
      2. 探索の有効性: 未踏の状態空間を有意義に探索しているか
      3. 学習段階の判断: まだ探索すべきか、活用に移行すべきか
    """

    def __init__(self, config: CuriosityConfig, agent_name: str, role: str):
        self.config = config
        self.agent_name = agent_name
        self.role = role
        self._sys_prompt = self._build_system_prompt()
        self._episode_history: List[Dict] = []
        self._llm_logger = _LLMLogger(config.log_dir, f"curiosity_{agent_name}")

        self.stats = {
            "calls": 0, "api_ok": 0, "api_fail": 0,
            "fallback": 0, "total_latency": 0.0,
        }

    def _build_system_prompt(self) -> str:
        role_desc = "CHASER (adversary)" if self.role == "adversary" else "EVADER (prey)"
        return (
            f"You are a CURIOSITY EVALUATOR for {self.agent_name}, a {role_desc} "
            f"in a 2D multi-agent tag game (3 chasers vs 1 prey).\n"
            "\n"
            "Your role is to evaluate whether the agent's recent trajectory contains\n"
            "STRATEGICALLY INTERESTING discoveries that deserve exploration bonuses.\n"
            "\n"
            "This is NOT about whether the agent performed well (that's the task reward).\n"
            "This is about whether the agent DISCOVERED something new and potentially useful.\n"
            "\n"
            "TYPES OF INTERESTING DISCOVERIES:\n"
            "\n"
            "1. SOCIAL NOVELTY (most important in decentralized MARL):\n"
            "   - Observed unexpected behavior from other agents\n"
            "   - Discovered a new coordination pattern (chasers flanking)\n"
            "   - Found a counter-strategy to opponents' behavior\n"
            "   - Noticed a teammate adapting their strategy\n"
            "\n"
            "2. SPATIAL NOVELTY:\n"
            "   - Explored new regions of the environment\n"
            "   - Found useful positions (e.g., near obstacles for cover)\n"
            "   - Discovered spatial configurations that lead to success\n"
            "\n"
            "3. STRATEGIC NOVELTY:\n"
            "   - Tried a new action sequence (not just random)\n"
            "   - Found a timing-based tactic (e.g., waiting then dashing)\n"
            "   - Developed a new approach to the task\n"
            "\n"
            "ANTI-PATTERNS (not interesting, should NOT get curiosity bonus):\n"
            "   - Random or chaotic behavior\n"
            "   - Repeated oscillation\n"
            "   - Doing nothing (staying still)\n"
            "   - Same strategy as always (no novelty)\n"
            "\n"
            "OUTPUT FORMAT (strict JSON):\n"
            "{\n"
            '  "novelty_score": float (0.0 to 1.0),\n'
            '  "social_novelty": float (0.0 to 1.0),\n'
            '  "spatial_novelty": float (0.0 to 1.0),\n'
            '  "strategic_novelty": float (0.0 to 1.0),\n'
            '  "exploration_phase": "explore" | "exploit" | "transition",\n'
            '  "reasoning": "brief explanation of what was novel or not"\n'
            "}\n"
            "\n"
            "Reply ONLY with the JSON."
        )

    def evaluate_episode(
        self,
        steps: List[Dict[str, Any]],
        curiosity_stats: Dict[str, float],
        episode: int,
    ) -> Dict[str, Any]:
        """
        エピソードのトラジェクトリを LLM で意味的に評価。

        Args:
            steps: エピソードの全ステップデータ
            curiosity_stats: 計算型好奇心の統計
            episode: エピソード番号

        Returns:
            LLM の評価結果辞書
        """
        summary = self._build_summary(steps, curiosity_stats, episode)
        user_msg = (
            f"Episode {episode} trajectory for {self.agent_name}:\n\n"
            f"{summary}\n\n"
            "Evaluate the strategic novelty and decide if this agent should\n"
            "continue exploring or shift to exploitation."
        )

        self.stats["calls"] += 1
        result = self._call_llm(user_msg, episode)

        if result is None:
            self.stats["fallback"] += 1
            result = self._fallback_evaluation(steps, curiosity_stats)

        return result

    def _build_summary(
        self,
        steps: List[Dict],
        curiosity_stats: Dict[str, float],
        episode: int,
    ) -> str:
        lines = []
        n = len(steps)
        lines.append(f"Agent: {self.agent_name} (role: {self.role})")
        lines.append(f"Episode: {episode}, Steps: {n}")

        # 計算型好奇心の統計
        lines.append(f"\nComputational Curiosity Statistics:")
        lines.append(f"  Dynamics curiosity (mean): {curiosity_stats.get('dynamics_mean', 0):.4f}")
        lines.append(f"  Social curiosity (mean):   {curiosity_stats.get('social_mean', 0):.4f}")
        lines.append(f"  Reward curiosity (mean):   {curiosity_stats.get('reward_mean', 0):.4f}")
        lines.append(f"  Current curiosity weight:  {curiosity_stats.get('weight', 0):.4f}")

        if not steps:
            return "\n".join(lines)

        # 位置の推移
        positions = [s.get("parsed", {}).get("self_pos", None) for s in steps]
        positions = [p for p in positions if p is not None]
        if positions:
            lines.append(f"\nMovement: {positions[0]} -> {positions[-1]}")
            # 移動距離
            total_dist = sum(
                np.hypot(positions[i+1][0] - positions[i][0],
                         positions[i+1][1] - positions[i][1])
                for i in range(len(positions) - 1)
            )
            lines.append(f"  Total distance traveled: {total_dist:.2f}")

        # 行動分布
        actions = [s.get("action", 0) for s in steps]
        action_counts = {i: actions.count(i) for i in range(5)}
        lines.append(f"\nAction distribution: " +
                     ", ".join(f"{ACTION_NAMES[k]}={v}" for k, v in action_counts.items()))

        # 行動の連続性（同じ行動の連続 = 単調、変化 = 探索的）
        changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        if n > 1:
            lines.append(f"  Action changes: {changes}/{n-1} ({changes/(n-1)*100:.0f}%)")
        else:
            lines.append(f"  Action changes: 0/0 (single step episode)")

        # ロール固有の情報
        if self.role == "adversary":
            dists = [s.get("parsed", {}).get("prey_dist", None) for s in steps]
            dists = [d for d in dists if d is not None]
            if dists:
                lines.append(f"\nPrey tracking:")
                lines.append(f"  Distance: start={dists[0]:.2f}, end={dists[-1]:.2f}, "
                             f"min={min(dists):.2f}")
                # チームメイトとの相対位置の変化
                for k in ("other0_rel", "other1_rel", "other2_rel"):
                    teammate_dists = [s.get("parsed", {}).get(f"{k}_prey_dist", None) for s in steps]
                    teammate_dists = [d for d in teammate_dists if d is not None]
                    if teammate_dists:
                        lines.append(f"  {k} prey_dist: {teammate_dists[0]:.2f} -> {teammate_dists[-1]:.2f}")
        else:
            dists = [s.get("parsed", {}).get("closest_adv_dist", None) for s in steps]
            dists = [d for d in dists if d is not None]
            if dists:
                lines.append(f"\nEvasion:")
                lines.append(f"  Closest adversary: start={dists[0]:.2f}, end={dists[-1]:.2f}, "
                             f"min={min(dists):.2f}")

        # キーステップのサンプル
        sample_idx = list(range(0, n, max(1, n // 5)))[:5]
        lines.append(f"\nSample steps:")
        for i in sample_idx:
            s = steps[i]
            act = ACTION_NAMES.get(s.get("action", 0), "?")
            env_r = s.get("env_reward", 0)
            cur_r = s.get("curiosity_total", 0)
            lines.append(f"  t={i}: action={act}, env_r={env_r:.2f}, curiosity={cur_r:.3f}")

        return "\n".join(lines)

    def _call_llm(self, user_msg: str, episode: int) -> Optional[Dict]:
        """LLM API 呼び出し"""
        if not _HAS_REQUESTS or not self.config.llm_api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.llm_model,
            "max_tokens": self.config.llm_max_tokens,
            "temperature": self.config.llm_temperature,
            "messages": [
                {"role": "system", "content": self._sys_prompt},
                {"role": "user",   "content": user_msg},
            ],
        }

        for attempt in range(1, self.config.llm_max_retries + 1):
            t0 = time.time()
            raw = ""
            try:
                resp = requests.post(
                    self.config.llm_base_url, headers=headers,
                    json=payload, timeout=self.config.llm_timeout,
                )
                elapsed = time.time() - t0
                self.stats["total_latency"] += elapsed

                if resp.status_code != 200:
                    self.stats["api_fail"] += 1
                    raw = resp.text[:500]
                    self._llm_logger.log(
                        self._sys_prompt, user_msg, raw, None,
                        elapsed * 1000, False, episode,
                    )
                    continue

                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
                parsed = self._parse_response(raw)

                if parsed is not None:
                    self.stats["api_ok"] += 1
                    self._llm_logger.log(
                        self._sys_prompt, user_msg, raw, parsed,
                        elapsed * 1000, True, episode,
                    )
                    return parsed

                self.stats["api_fail"] += 1
                self._llm_logger.log(
                    self._sys_prompt, user_msg, raw, None,
                    elapsed * 1000, False, episode,
                )

            except Exception as e:
                elapsed = time.time() - t0
                self.stats["total_latency"] += elapsed
                self.stats["api_fail"] += 1
                self._llm_logger.log(
                    self._sys_prompt, user_msg, str(e), None,
                    elapsed * 1000, False, episode,
                )

        return None

    @staticmethod
    def _parse_response(text: str) -> Optional[Dict]:
        """LLM 応答をパース"""
        text = text.strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
        else:
            m = re.search(r"(\{.*\})", text, re.DOTALL)
            if m:
                text = m.group(1)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        # 必須フィールドの検証
        required = ["novelty_score"]
        for field_name in required:
            if field_name not in data:
                return None

        # 数値のクリップ
        for key in ["novelty_score", "social_novelty", "spatial_novelty", "strategic_novelty"]:
            if key in data:
                data[key] = float(np.clip(data[key], 0.0, 1.0))

        return data

    @staticmethod
    def _fallback_evaluation(steps: List[Dict], curiosity_stats: Dict) -> Dict:
        """LLM 失敗時のフォールバック"""
        # 計算型好奇心の統計からヒューリスティックに判断
        social_mean = curiosity_stats.get("social_mean", 0)
        dynamics_mean = curiosity_stats.get("dynamics_mean", 0)

        # action の多様性を新規性の指標にする
        actions = [s.get("action", 0) for s in steps]
        unique_ratio = len(set(actions)) / 5.0 if actions else 0

        # 社会的好奇心が高い = 他エージェントの行動が予測外
        social_novelty = min(1.0, social_mean / max(dynamics_mean, 0.01))

        novelty = 0.3 * unique_ratio + 0.5 * min(1.0, social_mean) + 0.2 * min(1.0, dynamics_mean)

        return {
            "novelty_score": round(novelty, 3),
            "social_novelty": round(social_novelty, 3),
            "spatial_novelty": round(unique_ratio, 3),
            "strategic_novelty": round(novelty, 3),
            "exploration_phase": "explore" if novelty > 0.5 else "exploit",
            "reasoning": "Heuristic fallback based on computational curiosity stats",
        }

    def print_stats(self):
        s = self.stats
        avg_lat = s["total_latency"] / max(s["api_ok"] + s["api_fail"], 1) * 1000
        print(f"[{self.agent_name}] LLM curiosity: calls={s['calls']}  "
              f"ok={s['api_ok']}  fail={s['api_fail']}  fallback={s['fallback']}  "
              f"avg_latency={avg_lat:.0f}ms")
        print(f"  Log: {self._llm_logger.log_path}")


# ====================================================================
# 5. 好奇心統合マネージャ
# ====================================================================

class CuriosityManager:
    """
    計算型好奇心 + 意味的好奇心を統合管理するクラス。

    各エージェントに1つずつ作成し、毎ステップの内発的報酬計算と
    エピソード単位の LLM 評価を統括する。
    """

    def __init__(
        self,
        config: CuriosityConfig,
        agent_name: str,
        agent_idx: int,
        role: str,
    ):
        self.config = config
        self.agent_name = agent_name
        self.agent_idx = agent_idx
        self.role = role

        # 計算型好奇心
        self.curiosity = CuriosityReward(config, agent_name, agent_idx)

        # LLM 意味的好奇心
        self.llm_evaluator = None
        if config.use_llm_curiosity and config.llm_api_key:
            self.llm_evaluator = LLMCuriosityEvaluator(config, agent_name, role)

        # エピソードバッファ
        self._episode_steps: List[Dict] = []
        self._episode = 0

        # LLM の最新評価結果
        self._llm_result: Optional[Dict] = None
        self._semantic_bonus = 0.0  # LLM からの追加好奇心ボーナス

    def reset_episode(self, episode: int = 0):
        """エピソード開始時にリセット"""
        self._episode_steps = []
        self._episode = episode

    def compute_intrinsic_reward(
        self,
        world_model: nn.Module,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        other_actions: Dict[str, int],
        device: torch.device,
    ) -> float:
        """
        1ステップの内発的報酬を計算。

        Returns:
            intrinsic_reward: float（環境報酬に加算する内発的報酬）
        """
        # 計算型好奇心
        cur = self.curiosity.compute(
            world_model, obs, action, reward, next_obs, other_actions, device,
        )

        # エピソードバッファに記録
        parsed = _parse_obs(obs, self.role)
        self._episode_steps.append({
            "obs": obs.copy(),
            "action": action,
            "env_reward": reward,
            "next_obs": next_obs.copy(),
            "other_actions": other_actions,
            "parsed": parsed,
            "curiosity_dynamics": cur["dynamics"],
            "curiosity_reward": cur["reward_cur"],
            "curiosity_social": cur["social"],
            "curiosity_total": cur["total"],
        })

        # 意味的好奇心のボーナスを加算（LLM 評価結果が利用可能な場合）
        total = cur["total"] + self._semantic_bonus

        return total

    def end_episode(self) -> Optional[Dict]:
        """
        エピソード終了時の処理。
        条件を満たせば LLM 評価を実行。

        Returns:
            LLM 評価結果（実行された場合）
        """
        result = None
        if (self.llm_evaluator is not None and
                self._episode % self.config.llm_eval_every_n_episodes == 0):
            result = self.llm_evaluator.evaluate_episode(
                self._episode_steps,
                self.curiosity.stats,
                self._episode,
            )
            self._llm_result = result

            if result:
                # 意味的好奇心ボーナスを次エピソードに反映
                novelty = result.get("novelty_score", 0.5)
                social = result.get("social_novelty", 0.5)
                # Social novelty を重視
                self._semantic_bonus = (
                    self.config.semantic_curiosity_weight *
                    (0.4 * novelty + 0.6 * social)
                )

                # 探索/活用フェーズの判断に基づいて好奇心重みを調整
                phase = result.get("exploration_phase", "explore")
                if phase == "exploit":
                    self.curiosity._current_weight = max(
                        self.config.curiosity_min_weight,
                        self.curiosity._current_weight * 0.8,
                    )
                elif phase == "explore":
                    self.curiosity._current_weight = min(
                        self.config.curiosity_initial_weight,
                        self.curiosity._current_weight * 1.1,
                    )

        return result

    def get_episode_summary(self) -> Dict[str, float]:
        """エピソードの好奇心統計サマリー"""
        if not self._episode_steps:
            return {}

        dynamics_vals = [s["curiosity_dynamics"] for s in self._episode_steps]
        social_vals = [s["curiosity_social"] for s in self._episode_steps]
        total_vals = [s["curiosity_total"] for s in self._episode_steps]

        summary = {
            "curiosity_dynamics_mean": np.mean(dynamics_vals),
            "curiosity_social_mean": np.mean(social_vals),
            "curiosity_total_mean": np.mean(total_vals),
            "curiosity_total_sum": sum(total_vals),
            "curiosity_weight": self.curiosity._current_weight,
            "semantic_bonus": self._semantic_bonus,
        }
        if self._llm_result:
            summary["llm_novelty_score"] = self._llm_result.get("novelty_score", 0)
            summary["llm_social_novelty"] = self._llm_result.get("social_novelty", 0)
            summary["llm_phase"] = self._llm_result.get("exploration_phase", "?")

        return summary

    def print_stats(self):
        s = self.curiosity.stats
        print(f"[{self.agent_name}] Curiosity: "
              f"dynamics={s['dynamics_mean']:.4f}  "
              f"social={s['social_mean']:.4f}  "
              f"reward={s['reward_mean']:.4f}  "
              f"weight={s['weight']:.4f}  "
              f"computed={s['compute_count']}")
        if self.llm_evaluator:
            self.llm_evaluator.print_stats()


# ====================================================================
# 6. LLM ログ
# ====================================================================

class _LLMLogger:
    """LLM 呼び出しのログ保存"""

    def __init__(self, log_dir: str, prefix: str):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{prefix}_{ts}.jsonl")
        self._count = 0

    def log(
        self,
        system_prompt: str,
        user_prompt: str,
        raw_response: str,
        parsed: Optional[Dict],
        latency_ms: float,
        success: bool,
        episode: int,
    ):
        self._count += 1
        entry = {
            "call_id": self._count,
            "timestamp": datetime.datetime.now().isoformat(),
            "episode": episode,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": raw_response,
            "parsed_result": parsed,
            "latency_ms": round(latency_ms, 1),
            "success": success,
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def log_path(self) -> str:
        return self.log_file


# ====================================================================
# 7. ヘルパー
# ====================================================================

def create_curiosity_managers(
    agent_names: List[str],
    config: CuriosityConfig,
) -> Dict[str, CuriosityManager]:
    """全エージェント分の CuriosityManager を一括生成"""
    managers = {}
    for idx, name in enumerate(agent_names):
        role = "prey" if name.startswith("agent") else "adversary"
        managers[name] = CuriosityManager(config, name, idx, role)
    return managers


# ====================================================================
# 8. 動作確認
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Curiosity Reward Module - Unit Test")
    print("=" * 60)

    cfg = CuriosityConfig(verbose=True)
    print(f"\nConfig: social_weight={cfg.social_curiosity_weight}, "
          f"decay={cfg.curiosity_decay_method}")

    # RunningEMA テスト
    ema = RunningEMA(0.99)
    vals = [1.0, 0.5, 2.0, 0.3, 1.5]
    for v in vals:
        n = ema.normalize(v)
        print(f"  EMA: input={v:.2f}, normalized={n:.3f}, mean={ema.mean:.3f}")

    # パーステスト
    test_json = '{"novelty_score": 0.7, "social_novelty": 0.8, "spatial_novelty": 0.3, "strategic_novelty": 0.5, "exploration_phase": "explore", "reasoning": "test"}'
    parsed = LLMCuriosityEvaluator._parse_response(test_json)
    print(f"\nParse test: {parsed}")

    print("\nDone.")
