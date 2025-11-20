import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from pikazoo import pikazoo_v0
from pikazoo.wrappers import SimplifyAction
import random


class SelfPlayGymWrapper(gym.Env):
    """自我对战的 Gymnasium 环境包装器"""

    def __init__(self, model_path=None):
        super().__init__()

        # 创建基础环境
        self.base_env = pikazoo_v0.env(winning_score=15, render_mode=None)
        self.base_env = SimplifyAction(self.base_env)

        # 设置空间
        self.action_space = self.base_env.action_space("player_1")
        self.observation_space = self.base_env.observation_space("player_1")

        # 自我对战相关属性
        self.current_player = "player_1"
        self.opponent_model = None
        self.episode_count = 0
        self.last_obs = None

        # 加载对手模型
        if model_path and os.path.exists(model_path + ".zip"):
            try:
                self.opponent_model = PPO.load(model_path)
                print(f"成功加载对手模型: {model_path}")
            except Exception as e:
                print(f"加载对手模型失败: {e}")

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        obs, infos = self.base_env.reset()

        # 随机选择训练的玩家（增加训练的多样性）
        self.current_player = random.choice(["player_1", "player_2"])
        self.episode_count += 1
        self.last_obs = obs

        return obs[self.current_player], infos[self.current_player]

    def step(self, action):
        if not self.base_env.agents:
            # 游戏已结束
            return np.zeros(self.observation_space.shape), 0, True, False, {}

        actions = {}

        # 当前训练的智能体动作
        actions[self.current_player] = action

        # 对手智能体动作
        other_player = "player_2" if self.current_player == "player_1" else "player_1"

        if self.opponent_model is not None and other_player in self.last_obs:
            try:
                # 使用训练好的模型作为对手
                opponent_obs = self.last_obs[other_player]
                opponent_action, _ = self.opponent_model.predict(
                    opponent_obs,
                    deterministic=False
                )
                actions[other_player] = opponent_action
            except Exception as e:
                # 如果模型预测失败，使用随机动作
                actions[other_player] = self.base_env.action_space(other_player).sample()
        else:
            # 使用随机策略作为对手
            actions[other_player] = self.base_env.action_space(other_player).sample()

        # 执行动作
        obs, rewards, terms, truncs, infos = self.base_env.step(actions)
        self.last_obs = obs

        # 检查游戏状态
        if not self.base_env.agents:
            terminated = True
            current_obs = np.zeros(self.observation_space.shape)
            current_reward = rewards.get(self.current_player, 0)
            current_info = {}
        else:
            terminated = terms.get(self.current_player, False)
            current_obs = obs.get(self.current_player, np.zeros(self.observation_space.shape))
            current_reward = rewards.get(self.current_player, 0)
            current_info = infos.get(self.current_player, {})

        truncated = truncs.get(self.current_player, False)

        return current_obs, current_reward, terminated, truncated, current_info

    def close(self):
        self.base_env.close()

    def update_opponent_model(self, model_path):
        """更新对手模型"""
        if os.path.exists(model_path + ".zip"):
            try:
                self.opponent_model = PPO.load(model_path)
                print(f"对手模型已更新: {model_path}")
            except Exception as e:
                print(f"更新对手模型失败: {e}")


def train_selfplay():
    """自我对战训练"""
    print("开始自我对战训练...")

    # 创建检查点目录
    os.makedirs("selfplay_checkpoints", exist_ok=True)

    # 创建初始环境（没有对手模型）
    env = SelfPlayGymWrapper()

    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    # 创建初始模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./pika_selfplay_tb/"
    )

    # 训练迭代
    total_iterations = 20
    timesteps_per_iteration = 100000  # 每次迭代的训练步数

    try:
        for iteration in range(total_iterations):
            print(f"\n=== 自我对战迭代 {iteration + 1}/{total_iterations} ===")

            # 如果有之前的模型，更新对手
            if iteration > 0:
                prev_model_path = f"selfplay_checkpoints/pikazoo_selfplay_iter_{iteration - 1}"
                env.update_opponent_model(prev_model_path)
                print(f"使用迭代 {iteration} 的模型作为对手")
            else:
                print("使用随机策略作为对手")

            # 训练当前模型
            print(f"开始训练 {timesteps_per_iteration} 步...")
            model.learn(
                total_timesteps=timesteps_per_iteration,
                reset_num_timesteps=False,
                tb_log_name=f"selfplay_iter_{iteration}",
                progress_bar=True
            )

            # 保存当前迭代的模型
            model_path = f"selfplay_checkpoints/pikazoo_selfplay_iter_{iteration}"
            model.save(model_path)
            print(f"模型已保存: {model_path}")

            # 每隔几次迭代进行一次快速测试
            if (iteration + 1) % 3 == 0:
                print(f"进行快速测试...")
                quick_test(model, iteration + 1)

        # 保存最终模型
        final_model_path = "pikazoo_selfplay_final"
        model.save(final_model_path)
        print(f"\n🎉 自我对战训练完成！最终模型保存为: {final_model_path}")

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        interrupted_path = f"selfplay_checkpoints/pikazoo_selfplay_interrupted_{iteration}"
        model.save(interrupted_path)
        print(f"当前模型已保存: {interrupted_path}")

    finally:
        env.close()

    return model


def quick_test(model, iteration):
    """快速测试模型表现"""
    test_env = SelfPlayGymWrapper()

    wins = 0
    total_games = 5

    for game in range(total_games):
        obs, info = test_env.reset()
        total_reward = 0

        for step in range(1000):  # 最大步数限制
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            if terminated or truncated:
                if total_reward > 0:
                    wins += 1
                break

    win_rate = wins / total_games
    print(f"快速测试结果 (迭代 {iteration}): {wins}/{total_games} 胜利 (胜率: {win_rate:.1%})")

    test_env.close()


def test_selfplay_models():
    """测试自我对战训练的模型"""
    print("测试自我对战模型...")

    # 尝试加载模型
    model_paths = [
        "pikazoo_selfplay_final",
        "selfplay_checkpoints/pikazoo_selfplay_iter_9",
        "selfplay_checkpoints/pikazoo_selfplay_iter_8"
    ]

    model = None
    for path in model_paths:
        try:
            model = PPO.load(path)
            print(f"成功加载模型: {path}")
            break
        except:
            continue

    if model is None:
        print("未找到训练好的模型！请先运行训练。")
        return

    # 创建测试环境（带可视化）
    test_env = pikazoo_v0.env(winning_score=5, render_mode="human")
    test_env = SimplifyAction(test_env)

    print("开始AI对战测试（AI vs 随机对手）...")

    wins = {"ai": 0, "random": 0}

    for game in range(3):
        print(f"\n=== 第 {game + 1} 场测试 ===")
        obs, info = test_env.reset()

        while test_env.agents:
            actions = {}

            # AI 控制 player_1
            action, _ = model.predict(obs["player_1"], deterministic=True)
            actions["player_1"] = action

            # 随机策略控制 player_2
            actions["player_2"] = test_env.action_space("player_2").sample()

            obs, rewards, terms, truncs, infos = test_env.step(actions)

            if any(terms.values()):
                if rewards.get("player_1", 0) > 0:
                    wins["ai"] += 1
                    print("🤖 AI 获胜！")
                else:
                    wins["random"] += 1
                    print("🎲 随机对手获胜！")
                break

    print(f"\n=== 最终测试结果 ===")
    print(f"AI 获胜: {wins['ai']} 场")
    print(f"随机对手获胜: {wins['random']} 场")

    test_env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_selfplay_models()
    else:
        model = train_selfplay()

        # 训练完成后询问是否测试
        test_choice = input("\n自我对战训练完成！是否立即测试模型？(y/n): ")
        if test_choice.lower() == 'y':
            test_selfplay_models()