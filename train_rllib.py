import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pikazoo import pikazoo_v0
from pikazoo.wrappers import SimplifyAction


def env_creator(args):
    """环境创建函数"""
    env = pikazoo_v0.env(
        winning_score=15,
        render_mode=None
    )
    env = SimplifyAction(env)
    return env


def train_multiagent():
    """多智能体训练"""
    # 初始化 Ray
    ray.init(ignore_reinit_error=True)

    # 注册环境
    tune.register_env("pikazoo", lambda config: ParallelPettingZooEnv(env_creator(config)))

    # 获取环境信息
    test_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    # 配置PPO
    config = (
        PPOConfig()
        .environment("pikazoo")
        .framework("torch")
        .rollouts(num_rollout_workers=4)
        .training(
            learning_rate=3e-4,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies={
                "player_1": (None, obs_space, act_space, {}),
                "player_2": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
        )
    )

    # 开始训练
    print("开始多智能体训练...")
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 100},
        checkpoint_freq=10,
        local_dir="./ray_results",
        name="pikazoo_multiagent"
    )

    ray.shutdown()
    return results


if __name__ == "__main__":
    results = train_multiagent()