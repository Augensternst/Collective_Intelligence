import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from pikazoo import pikazoo_v0
from pikazoo.wrappers import SimplifyAction, RewardByBallPosition


class PikaZooGymWrapper(gym.Env):
    """将 PettingZoo 环境包装为标准的 Gymnasium 环境"""

    def __init__(self, player_side="player_1"):
        super().__init__()

        # 创建基础环境
        self.env = pikazoo_v0.env(
            winning_score=15,
            render_mode=None
        )

        # 使用包装器
        self.env = SimplifyAction(self.env)
        self.env = RewardByBallPosition(
            self.env,
            additional_reward=[0.1, 0.05, -0.05, -0.1, -0.1, -0.05, 0.05, 0.1]
        )

        self.player_side = player_side
        self.other_side = "player_2" if player_side == "player_1" else "player_1"

        # 设置动作和观察空间
        self.action_space = self.env.action_space(self.player_side)
        self.observation_space = self.env.observation_space(self.player_side)

        # 初始化状态
        self.agents = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        observations, infos = self.env.reset()
        self.agents = self.env.agents.copy()

        return observations[self.player_side], infos[self.player_side]

    def step(self, action):
        if not self.agents:
            # 如果游戏已结束，返回默认值
            return np.zeros(self.observation_space.shape), 0, True, False, {}

        # 构造动作字典
        actions = {self.player_side: action}

        # 为对手选择动作（随机或者简单策略）
        if self.other_side in self.agents:
            # 简单的追球策略
            obs = self.last_obs if hasattr(self, 'last_obs') else None
            if obs is not None and self.other_side in obs:
                other_obs = obs[self.other_side]
                ball_x, ball_y = other_obs[26], other_obs[27]  # 球的位置
                player_x = other_obs[0]  # 对手位置

                # 简单策略：追球
                if abs(ball_x - player_x) > 50:
                    if ball_x < player_x:
                        actions[self.other_side] = 4  # 向左（BACK for player_2）
                    else:
                        actions[self.other_side] = 3  # 向右（FRONT for player_2）
                elif ball_y > 200:  # 球接近地面
                    actions[self.other_side] = 2  # 跳跃
                else:
                    actions[self.other_side] = 0  # 无动作
            else:
                actions[self.other_side] = self.env.action_space(self.other_side).sample()

        # 执行步骤
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # 更新智能体列表
        self.agents = self.env.agents.copy()
        self.last_obs = observations

        # 返回当前玩家的信息
        obs = observations.get(self.player_side, np.zeros(self.observation_space.shape))
        reward = rewards.get(self.player_side, 0)
        terminated = terminations.get(self.player_side, True)
        truncated = truncations.get(self.player_side, False)
        info = infos.get(self.player_side, {})

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        self.env.close()


class TrainingCallback(BaseCallback):
    """训练回调，用于记录进度"""

    def __init__(self, eval_freq=10000, save_freq=50000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"训练步数: {self.n_calls}")

        if self.n_calls % self.save_freq == 0:
            self.model.save(f"checkpoints/pikazoo_ppo_{self.n_calls}")
            print(f"模型已保存: checkpoints/pikazoo_ppo_{self.n_calls}")

        return True


def train_single_agent():
    """训练单个智能体"""
    print("开始训练皮卡丘排球AI...")

    # 创建检查点目录
    os.makedirs("checkpoints", exist_ok=True)

    # 创建环境
    env = PikaZooGymWrapper("player_1")

    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./pika_tensorboard/"
    )

    # 设置回调
    callback = TrainingCallback(eval_freq=10000, save_freq=50000)

    try:
        # 开始训练
        model.learn(
            total_timesteps=100000,  # 100万步
            callback=callback,
            tb_log_name="pikazoo_ppo",
            progress_bar=True
        )

        # 保存最终模型
        model.save("pikazoo_ppo_final")
        print("训练完成！模型已保存为 'pikazoo_ppo_final'")

    except KeyboardInterrupt:
        print("训练被用户中断")
        model.save("pikazoo_ppo_interrupted")
        print("模型已保存为 'pikazoo_ppo_interrupted'")

    finally:
        env.close()

    return model


def test_trained_model():
    """测试训练好的模型"""
    try:
        model = PPO.load("pikazoo_ppo_final")
        print("成功加载训练模型")
    except:
        try:
            model = PPO.load("pikazoo_ppo_interrupted")
            print("加载中断保存的模型")
        except:
            print("未找到训练模型")
            return

    # 创建测试环境（带渲染）
    env = PikaZooGymWrapper("player_1")
    env.env.render_mode = "human"  # 启用渲染

    print("开始测试训练好的模型...")

    # 进行几场测试游戏
    for game in range(3):
        print(f"\n=== 第 {game + 1} 场测试 ===")
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                print(f"游戏结束！总奖励: {total_reward:.2f}, 步数: {steps}")
                if reward > 0:
                    print("AI 获胜！")
                elif reward < 0:
                    print("AI 失败...")
                else:
                    print("平局")
                break

            if steps > 5000:  # 防止无限循环
                print("游戏超时")
                break

    env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_trained_model()
    else:
        model = train_single_agent()

        # 训练完成后询问是否测试
        test_choice = input("\n训练完成！是否立即测试模型？(y/n): ")
        if test_choice.lower() == 'y':
            test_trained_model()