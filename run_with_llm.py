# -*- coding: utf-8 -*-
"""
run_with_llm.py
===============
既存の MATWM 訓練ループに LLM 行動選択を統合する実行スクリプト。

3つのモードを提供:
  A) LLM-only   : 全エージェントが LLM で行動（World Model訓練なし）
  B) Hybrid      : 環境での行動は LLM、World Model + Critic は通常通り訓練
  C) LLM-evaluate: 訓練済みMATWMと LLM を比較評価

使い方
------
    # 方法1: 環境変数でキーを設定して実行
    export OPENROUTER_API_KEY="sk-or-v1-xxxx"
    python run_with_llm.py

    # 方法2: コード内で直接キーを設定（下記 ★ 部分を編集）
"""

import os
import sys
import time
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm

# ------------------------------------------------------------------
# ★ ここを編集してください
# ------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_API_KEY_HERE")

# 使用モデル（OpenRouter のモデル名）
# 無料枠の推奨モデル:
#   "google/gemma-3-4b-it:free"                         … 軽量・高速
#   "google/gemma-3-12b-it:free"                        … 中量
#   "mistralai/mistral-small-3.1-24b-instruct:free"     … 高品質
MODEL_NAME = "google/gemma-3-4b-it:free"

# 実行モード: "llm_only" / "hybrid" / "compare"
RUN_MODE = "llm_only"

# エピソード数（LLM呼び出し回数を制御）
NUM_EPISODES = 5         # LLMモードの評価エピソード数
MAX_CYCLES   = 25        # 1エピソードの最大ステップ数
# ------------------------------------------------------------------


def make_env(max_cycles=25, seed=None):
    """simple_tag環境を作成"""
    from pettingzoo.mpe import simple_tag_v3
    env = simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=None,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


# ==================================================================
# モードA: LLM-only（全エージェントがLLMで行動）
# ==================================================================

def run_llm_only():
    """
    World Model訓練なし。全エージェントの行動をLLMで決定し、
    環境上でのパフォーマンスを評価する。
    """
    from llm_action_selector import LLMConfig, create_llm_selectors

    print("=" * 60)
    print("Mode A: LLM-Only Evaluation")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Episodes: {NUM_EPISODES}")
    print("=" * 60)

    cfg = LLMConfig(
        api_key=OPENROUTER_API_KEY,
        model=MODEL_NAME,
        verbose=True,
    )

    env = make_env(max_cycles=MAX_CYCLES, seed=42)
    obs, _ = env.reset(seed=42)
    agent_names = list(obs.keys())
    env.close()

    # LLMセレクターを作成
    selectors = create_llm_selectors(agent_names, cfg)

    # 評価ループ
    env = make_env(max_cycles=MAX_CYCLES)
    all_rewards = {name: [] for name in agent_names}

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        episode_reward = {name: 0.0 for name in agent_names}

        for step in range(MAX_CYCLES):
            actions = {}
            for name in agent_names:
                actions[name] = selectors[name].select_action(obs[name])

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            for name in agent_names:
                episode_reward[name] += rewards[name]

            done = all(terms[n] or truncs[n] for n in agent_names)
            obs = next_obs
            if done:
                break

        # エピソード結果
        print(f"\nEpisode {ep+1}/{NUM_EPISODES}:")
        for name in agent_names:
            all_rewards[name].append(episode_reward[name])
            print(f"  {name}: reward={episode_reward[name]:.2f}")

    env.close()

    # 統計
    print("\n" + "=" * 60)
    print("LLM-Only Results")
    print("=" * 60)
    for name in agent_names:
        r = all_rewards[name]
        print(f"  {name}: mean={np.mean(r):.2f}  std={np.std(r):.2f}")

    print("\nLLM Call Statistics:")
    for name, sel in selectors.items():
        sel.print_stats()

    return all_rewards


# ==================================================================
# モードB: Hybrid（環境行動=LLM、WM+Critic=通常訓練）
# ==================================================================

def run_hybrid():
    """
    環境でのデータ収集はLLMで行動を決定し、
    World Model と Critic は通常通り訓練する。
    Actor Network の訓練は行わない（LLMが方策を担う）。

    これにより:
    - World Model は LLM の行動データから環境ダイナミクスを学習
    - Teammate Predictor は LLM の行動パターンを学習
    - 将来的に Actor を LLM の蒸留対象にすることも可能
    """
    import torch
    from llm_action_selector import LLMConfig, patch_agent_with_llm

    print("=" * 60)
    print("Mode B: Hybrid Training (LLM actions + WM training)")
    print(f"  Model : {MODEL_NAME}")
    print(f"  Episodes: {NUM_EPISODES}")
    print("=" * 60)

    # --- MATWM セットアップ ---
    from matwm_implementation import MATWMConfig, pad_observation
    from matwm_agent import MATWMAgent
    from matwm_utils import initialize_matwm_weights, init_weights

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matwm_config = MATWMConfig(
        total_steps=NUM_EPISODES * MAX_CYCLES,
        warmup_steps=0,       # LLM使用のためwarmup不要
    )

    env = make_env(max_cycles=MAX_CYCLES, seed=42)
    obs, _ = env.reset(seed=42)
    agent_names = list(obs.keys())
    env.close()

    # 共有 World Model
    shared_wm, shared_opt = MATWMAgent.create_shared_world_model(matwm_config, device)

    # エージェント作成
    agents = {}
    for idx, name in enumerate(agent_names):
        agents[name] = MATWMAgent(matwm_config, name, idx, device, shared_world_model=shared_wm)

    # 重み初期化
    initialize_matwm_weights(shared_wm,
                             list(agents.values())[0].actor,
                             list(agents.values())[0].critic)
    for agent in agents.values():
        agent.actor.apply(init_weights)
        agent.critic.apply(init_weights)

    # --- LLM で select_action を差し替え ---
    llm_cfg = LLMConfig(
        api_key=OPENROUTER_API_KEY,
        model=MODEL_NAME,
        verbose=False,
    )
    for name, agent in agents.items():
        role = "prey" if name.startswith("agent") else "adversary"
        patch_agent_with_llm(agent, role, llm_cfg)

    # --- 訓練ループ ---
    env = make_env(max_cycles=MAX_CYCLES)
    episode_rewards = {name: [] for name in agent_names}
    training_metrics = defaultdict(list)
    global_step = 0
    min_data_steps = matwm_config.wm_batch_length + 10  # WM訓練に必要な最小データ量

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        ep_reward = {name: 0.0 for name in agent_names}

        for step in range(MAX_CYCLES):
            # LLM で行動選択（patch済み）
            actions = {}
            for name, agent in agents.items():
                actions[name] = agent.select_action(obs[name])

            next_obs, rewards, terms, truncs, _ = env.step(actions)

            # 経験を保存
            for name, agent in agents.items():
                other_acts = {k: v for k, v in actions.items() if k != name}
                agent.store_experience(
                    obs[name], actions[name], rewards[name],
                    next_obs[name],
                    terms[name] or truncs[name],
                    other_acts,
                )
                ep_reward[name] += rewards[name]

            obs = next_obs
            global_step += 1

            # World Model 訓練（十分なデータが溜まったら）
            if global_step > min_data_steps:
                wm_metrics = MATWMAgent.train_world_model_shared(
                    agents, matwm_config, device, shared_opt
                )
                if wm_metrics:
                    for k, v in wm_metrics.items():
                        training_metrics[f"shared_{k}"].append(v)

            if all(terms[n] or truncs[n] for n in agent_names):
                break

        # エピソード結果
        print(f"\nEp {ep+1}/{NUM_EPISODES}:", end="")
        for name in agent_names:
            episode_rewards[name].append(ep_reward[name])
            print(f"  {name}={ep_reward[name]:.2f}", end="")
        print()

        # WM損失
        wm_key = "shared_wm_total_loss"
        if wm_key in training_metrics and training_metrics[wm_key]:
            recent = training_metrics[wm_key][-5:]
            print(f"  WM loss (last 5 avg): {np.mean(recent):.4f}")

    env.close()

    # 統計
    print("\n" + "=" * 60)
    print("Hybrid Training Results")
    print("=" * 60)
    for name in agent_names:
        r = episode_rewards[name]
        print(f"  {name}: mean={np.mean(r):.2f}  std={np.std(r):.2f}")

    print("\nLLM Call Statistics:")
    for name, agent in agents.items():
        if hasattr(agent, "_llm_selector"):
            agent._llm_selector.print_stats()

    if "shared_wm_total_loss" in training_metrics:
        losses = training_metrics["shared_wm_total_loss"]
        print(f"\nWorld Model: {len(losses)} updates, "
              f"final loss={losses[-1]:.4f}" if losses else "no updates")

    return agents, episode_rewards, training_metrics


# ==================================================================
# モードC: 比較評価（MATWM Actor vs LLM vs Random）
# ==================================================================

def run_compare(checkpoint_path: str = None):
    """
    同一環境で 3つの方策を比較:
      1. ランダム行動
      2. LLM 行動
      3. MATWM Actor（チェックポイントがある場合）
    """
    from llm_action_selector import LLMConfig, create_llm_selectors

    print("=" * 60)
    print("Mode C: Compare Random vs LLM vs MATWM")
    print("=" * 60)

    agent_names_cache = None

    def _eval_policy(policy_fn, label, num_ep=NUM_EPISODES):
        nonlocal agent_names_cache
        env = make_env(max_cycles=MAX_CYCLES)
        rewards_all = None

        for ep in range(num_ep):
            obs, _ = env.reset(seed=ep)
            if agent_names_cache is None:
                agent_names_cache = list(obs.keys())
            if rewards_all is None:
                rewards_all = {n: [] for n in agent_names_cache}
            ep_r = {n: 0.0 for n in agent_names_cache}

            for step in range(MAX_CYCLES):
                actions = policy_fn(obs, agent_names_cache, env)
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                for n in agent_names_cache:
                    ep_r[n] += rewards[n]
                obs = next_obs
                if all(terms[n] or truncs[n] for n in agent_names_cache):
                    break

            for n in agent_names_cache:
                rewards_all[n].append(ep_r[n])

        env.close()

        print(f"\n  [{label}]")
        for n in agent_names_cache:
            r = rewards_all[n]
            print(f"    {n}: mean={np.mean(r):.2f}  std={np.std(r):.2f}")
        return rewards_all

    # 1. ランダム
    def random_policy(obs, names, env):
        return {n: env.action_space(n).sample() for n in names}

    print("\n--- Random Policy ---")
    random_results = _eval_policy(random_policy, "Random")

    # 2. LLM
    llm_cfg = LLMConfig(api_key=OPENROUTER_API_KEY, model=MODEL_NAME)
    selectors = None

    def llm_policy(obs, names, env):
        nonlocal selectors
        if selectors is None:
            selectors = create_llm_selectors(names, llm_cfg)
        return {n: selectors[n].select_action(obs[n]) for n in names}

    print("\n--- LLM Policy ---")
    llm_results = _eval_policy(llm_policy, f"LLM ({MODEL_NAME})")

    if selectors:
        print("\n  LLM Stats:")
        for n, s in selectors.items():
            s.print_stats()

    # 3. MATWM（チェックポイントがあれば）
    if checkpoint_path and os.path.exists(checkpoint_path):
        import torch
        from matwm_implementation import MATWMConfig
        from matwm_agent import MATWMAgent

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = MATWMConfig()
        shared_wm, _ = MATWMAgent.create_shared_world_model(cfg, device)
        agents_matwm = {}
        for idx, name in enumerate(agent_names_cache):
            agents_matwm[name] = MATWMAgent(cfg, name, idx, device, shared_world_model=shared_wm)
            agent_path = os.path.join(os.path.dirname(checkpoint_path), f"{name}.pt")
            if os.path.exists(agent_path):
                agents_matwm[name].load(agent_path)

        def matwm_policy(obs, names, env):
            return {n: agents_matwm[n].select_action(obs[n], deterministic=True) for n in names}

        print("\n--- MATWM Actor Policy ---")
        matwm_results = _eval_policy(matwm_policy, "MATWM Actor")

    return


# ==================================================================
# メイン
# ==================================================================

if __name__ == "__main__":
    # API キー確認
    if OPENROUTER_API_KEY == "YOUR_API_KEY_HERE" or not OPENROUTER_API_KEY:
        print("!" * 60)
        print("  OPENROUTER_API_KEY が未設定です。")
        print()
        print("  方法1: 環境変数で設定")
        print("    export OPENROUTER_API_KEY='sk-or-v1-xxxx'")
        print()
        print("  方法2: このファイル冒頭の OPENROUTER_API_KEY を直接編集")
        print()
        print("  OpenRouter の無料APIキーは https://openrouter.ai で取得できます。")
        print("!" * 60)
        sys.exit(1)

    print(f"\n  API Key: {OPENROUTER_API_KEY[:12]}...{OPENROUTER_API_KEY[-4:]}")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Mode:    {RUN_MODE}\n")

    if RUN_MODE == "llm_only":
        run_llm_only()
    elif RUN_MODE == "hybrid":
        run_hybrid()
    elif RUN_MODE == "compare":
        run_compare()
    else:
        print(f"Unknown mode: {RUN_MODE}")
        print("Choose: llm_only / hybrid / compare")
