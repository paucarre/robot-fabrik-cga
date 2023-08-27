import torch
from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from torch.utils.tensorboard import SummaryWriter
from linguamechanica.training_context import EpisodeState, TrainingState


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(agent, training_state, eval_episodes=10):
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chain = urdf_robot.extract_open_chains(0.3)[-1]
    eval_env = Environment(
        open_chain, training_state.weights, training_state.max_steps_done
    )

    avg_acc_reward = 0.0
    initial_rewards = torch.zeros(eval_episodes)
    final_rewards = torch.zeros(eval_episodes)
    for idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        eval_reward = None
        reward = None
        while not done:
            action, log_prob, _, _ = agent.choose_action(state, training_state)
            state, reward, done = eval_env.step(action)
            if eval_reward is None:
                initial_rewards[idx] = reward
                eval_reward = reward
            else:
                eval_reward += reward
        final_rewards[idx] = reward
        avg_acc_reward += eval_reward
    avg_acc_reward /= eval_episodes

    print(f"Evaluation over {eval_episodes} episodes: {avg_acc_reward.item():.3f}")
    return avg_acc_reward, initial_rewards, final_rewards


def summary_evaluatation(
    summary, initial_rewards, final_rewards, avg_acc_reward, training_state
):
    # summary.add_scalar("Avg Initial Reward Eval", initial_rewards.mean(), t)
    # summary.add_scalar("Avg Final Reward Eval", final_rewards.mean(), t)
    summary.add_scalar(
        "Avg Improved Reward Eval",
        (final_rewards / (initial_rewards + 1e-10)).mean(),
        training_state.t,
    )
    summary.add_scalar("Accumulated Reward Eval", avg_acc_reward, training_state.t)


def summary_done(summary, training_state, episode):
    # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
    print(
        f"Total T: {training_state.t+1} Episode Num: {episode.num+1} Episode T: {episode.timesteps} Reward: {episode.reward.item():.3f}"
    )
    summary.add_scalar("Reward/train", episode.reward, training_state.t)


def train():
    summary = SummaryWriter()
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chain = urdf_robot.extract_open_chains(0.3)[-1]
    # TODO: place all these constants as arguments

    training_state = TrainingState()
    env = Environment(open_chain, training_state.weights, training_state.max_steps_done)
    agent = IKAgent(
        open_chain=open_chain,
        summary=summary,
        lr_actor=training_state.lr_actor,
        lr_critic=training_state.lr_critic,
        state_dims=(env.observation_space.shape),
        action_dims=env.action_dims,
        gamma=training_state.gamma,
        policy_freq=training_state.policy_freq,
        tau=training_state.tau,
        max_action=training_state.max_action,
        min_variance=training_state.min_variance,
        max_variance=training_state.max_variance,
        replay_buffer_max_size=training_state.replay_buffer_max_size,
    )
    episode = EpisodeState()
    state, done = env.reset(), False
    initial_reward = None
    # agent.load(f"checkpoint_46999")

    for training_state.t in range(training_state.t, int(training_state.max_timesteps)):
        episode.timesteps += 1
        # if training_state.use_actor_for_data_generation():
        action, log_prob, mu_v, var_v = agent.choose_action(state, training_state)
        summary.add_scalar(
            "Data Generation Action Mean",
            action.mean(),
            training_state.t,
        )
        summary.add_scalar(
            "Data Generation Action Std",
            action.std(),
            training_state.t,
        )
        summary.add_scalar(
            "Data Generation Mean",
            mu_v.mean(),
            training_state.t,
        )
        summary.add_scalar(
            "Data Generation Variance",
            var_v.std(),
            training_state.t,
        )
        # else:
        # action = env.sample_noisy_jacobian_action()
        # action = env.sample_random_action()

        # Perform action
        next_state, reward, done = env.step(action)
        agent.store_transition(
            state=state.detach().cpu(),
            action=action[0, :].detach().cpu(),
            reward=reward.detach().cpu(),
            next_state=next_state.detach().cpu(),
            done=torch.Tensor([done]).detach().cpu(),
        )

        state = next_state
        episode.reward += reward
        if initial_reward is None:
            initial_reward = reward

        if training_state.can_train_buffer():
            agent.train_buffer(training_state)
        if done and training_state.can_train_buffer():
            summary_done(summary, training_state, episode)
        if training_state.can_eval_policy():
            avg_acc_reward, initial_rewards, final_rewards = eval_policy(
                agent, training_state, 2
            )
            summary_evaluatation(
                summary, initial_rewards, final_rewards, avg_acc_reward, training_state
            )

        if training_state.can_save():
            agent.save(training_state)

        if done:
            final_reward = reward
            summary.add_scalar(
                "Reward Improvement Training",
                final_reward / (initial_reward + 1e-10),
                training_state.t,
            )
            state = env.reset()
            initial_reward = None
            done = False
            episode.create_new()

    summary.close()


if __name__ == "__main__":
    train()