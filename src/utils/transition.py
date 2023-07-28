import torch

class TransitionFactory:
    def __init__(self, agent, device=torch.device('cpu'), dtype=torch.float32) -> None:
        self.agent = agent
        self.device = device
        self.dtype = dtype
        self.create = self.first_transition

    def _set_function(self, data, attr):
        if isinstance(data, torch.Tensor):
            setattr(self, f"{attr}_function", 
                    lambda x: x.to(device=self.device, dtype=self.dtype))
        else:
            setattr(self, f"{attr}_function", 
                    lambda x: torch.tensor(x, device=self.device, dtype=self.dtype))

    def set_state_function(self, state):
        if isinstance(state, dict):
            if all([isinstance(value, torch.Tensor) for value in state.values()]):
                self.state_function = lambda state: \
                    {key: value.to(device=self.device, dtype=self.dtype) for key, value in state.items()}
            else:
                self.state_function = lambda state: \
                    {key: torch.tensor(value, device=self.device, dtype=self.dtype) for key, value in state.items()}
        elif isinstance(state, torch.Tensor):
            self.state_function = lambda state: state.to(device=self.device, dtype=self.dtype)
        else:
            self.state_function = lambda state: torch.tensor(state, device=self.device, dtype=self.dtype)

    def first_transition(self, state, action, reward, logprob, done):
        state_value = self.agent.critic(state)

        self.state_shape = {key: value.shape for key, value in state.items()} if isinstance(state, dict) else state.shape
        self.action_shape = action.shape
        self.reward_shape = reward.shape
        self.logprob_shape = logprob.shape
        self.done_shape = done.shape
        self.state_value_shape = state_value.shape

        self.set_state_function(state)
        self._set_function(action, 'action')
        self._set_function(reward, 'reward')
        self._set_function(logprob, 'logprob')
        self._set_function(done, 'done')
        self._set_function(state_value, 'state_value')

        self.create = self.create_transition
        return self.create(state, action, reward, logprob, done)

    def create_transition(self, state, action, reward, logprob, done):
        state_value = self.agent.critic(state)

        state = self.state_function(state)
        action = self.action_function(action)
        reward = self.reward_function(reward)
        logprob = self.logprob_function(logprob)
        done = self.done_function(done)
        state_value = self.state_value_function(state_value)

        return Transition(
            state=state,
            action=action,
            reward=reward,
            logprob=logprob,
            done=done,
            state_value=state_value
        )

class Transition:
    def __init__(self, *, state, action, reward, logprob, done, state_value) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.logprob = logprob
        self.done = done
        self.state_value = state_value

    def __repr__(self) -> str:
        return "Transition(state={}, action={}, reward={}, logprob={}, done={}, state_value={})".format(
            self.state, self.action, self.reward, self.logprob, self.done, self.state_value
        )

    def __str__(self) -> str:
        return self.__repr__()
