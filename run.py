from pikazoo import pikazoo_v0

# 创建游戏环境
env = pikazoo_v0.env(
    winning_score=5,
    render_mode="human"  # 显示游戏界面
)

# 游戏循环
observations, infos = env.reset()
print("皮卡丘排球游戏开始！按 Ctrl+C 退出")

try:
    while env.agents:
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_space(agent).sample()

        observations, rewards, terminations, truncations, infos = env.step(actions)

        if any(rewards.values()):
            print(f"得分变化: {rewards}")
            print(f"当前比分: {infos[list(infos.keys())[0]]['score']}")

        if any(terminations.values()):
            print("游戏结束！")
            break

except KeyboardInterrupt:
    print("游戏被用户终止")
finally:
    env.close()