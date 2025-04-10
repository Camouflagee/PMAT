from mat.runner.shared.base_runner import Runner

class RRMRunner(Runner):
    def __init__(self, config):
        super(RRMRunner, self).__init__(config)
        
    def run(self):
        self.warmup()   
        
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                    
                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)
                
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                
                # insert data into buffer
                self.insert(data)
                
            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()
                
            # log information
            if episode % self.log_interval == 0:
                self.log_train(train_infos, total_num_steps)
                
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
