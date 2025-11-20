from stable_baselines3 import PPO
from pikazoo import pikazoo_v0
from pikazoo.wrappers import SimplifyAction
import time


def test_trained_vs_random():
    """测试训练好的模型对战随机智能体"""
    # 加载训练好的模型
    try:
        model = PPO.load("pikazoo_ppo_final")
        print("成功加载训练模型")
    except:
        print("未找到训练模型，请先运行训练脚本")
        return

    # 创建测试环境
    env = pikazoo_v0.env(
        winning_score=5,
        render_mode="human"
    )
    env = SimplifyAction(env)

    wins = {"player_1": 0, "player_2": 0}

    # 进行多场对战
    for game in range(5):
        print(f"\n=== 第 {game + 1} 场比赛 ===")
        obs, infos = env.reset()

        while env.agents:
            actions = {}

            # player_1 使用训练模型
            action, _ = model.predict(obs["player_1"], deterministic=True)
            actions["player_1"] = action

            # player_2 使用随机策略
            actions["player_2"] = env.action_space("player_2").sample()

            obs, rewards, terms, truncs, infos = env.step(actions)

            if any(terms.values()):
                if rewards["player_1"] > 0:
                    wins["player_1"] += 1
                    print("AI (player_1) 获胜！")
                else:
                    wins["player_2"] += 1
                    print("随机智能体 (player_2) 获胜！")
                break

            time.sleep(0.1)  # 稍微放慢速度以便观察

    print(f"\n=== 最终结果 ===")
    print(f"AI 获胜: {wins['player_1']} 场")
    print(f"随机智能体 获胜: {wins['player_2']} 场")

    env.close()


def test_selfplay_models():
    """测试自我对战模型"""
    try:
        model1 = PPO.load("pikazoo_selfplay_final_1")
        model2 = PPO.load("pikazoo_selfplay_final_1")  # 使用不同迭代的模型
        print("成功加载自我对战模型")
    except:
        print("未找到自我对战模型")
        return

    env = pikazoo_v0.env(winning_score=5, render_mode="human")
    env = SimplifyAction(env)

    obs, infos = env.reset()
    print("两个训练模型对战开始！")

    while env.agents:
        actions = {}

        # 两个不同的训练模型对战
        action1, _ = model1.predict(obs["player_1"], deterministic=True)
        action2, _ = model2.predict(obs["player_2"], deterministic=True)

        actions["player_1"] = action1
        actions["player_2"] = action2

        obs, rewards, terms, truncs, infos = env.step(actions)

        if any(terms.values()):
            winner = "最终模型" if rewards["player_1"] > 0 else "中期模型"
            print(f"获胜者: {winner}")
            break

        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 训练模型 vs 随机智能体")
    print("2. 自我对战模型测试")

    choice = input("输入选择 (1 或 2): ")

    if choice == "1":
        test_trained_vs_random()
    elif choice == "2":
        test_selfplay_models()
    else:
        print("无效选择")