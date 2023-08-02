import torch

class TransitionFactory:
    """A factory for creating and managing transitions in reinforcement learning.

    This class is used to convert raw state, action, reward, log-probability, 
    and done flag into a Transition object. It also manages the device and data type for 
    the conversion of these items into PyTorch tensors, if they aren't already.

    Attributes:
        agent (PPOAgent): The agent that generates the transitions.
        device (torch.device): The device to which tensors will be transferred.
        dtype (torch.dtype): The data type for the tensors.
        create (function): The function for creating transitions.
    """
    def __init__(self, agent, device=torch.device('cpu'), dtype=torch.float32) -> None:
        """Initializes the TransitionFactory with a specified agent, device, and dtype.

        Args:
            agent (PPOAgent): The agent that generates the transitions.
            device (torch.device): The device to which tensors will be transferred.
            dtype (torch.dtype): The data type for the tensors.
        """
        self.agent = agent
        self.device = device
        self.dtype = dtype
        self.create = self.first_transition

    def _set_function(self, data, attr):
        """Sets the function for processing a specific attribute of a transition.

        Args:
            data (torch.Tensor or other types): The data to be processed.
            attr (str): The attribute to be processed.
        """
        if isinstance(data, torch.Tensor):
            setattr(self, f"{attr}_function", 
                    lambda x: x.to(device=self.device, dtype=self.dtype))
        else:
            setattr(self, f"{attr}_function", 
                    lambda x: torch.tensor(x, device=self.device, dtype=self.dtype))

    def set_state_function(self, state):
        """Sets the function for processing the state of a transition.

        Args:
            state (torch.Tensor or dict): The state to be processed.
        """
        if isinstance(state, dict):
            if all([isinstance(value, torch.Tensor) for value in state.values()]):
                self.state_function = lambda state: \
                    {key: value.to(device=self.device, dtype=self.dtype) for key, value in state.items()}
            else:
                self.state_function = lambda state: \
                    {key: torch.from_numpy(value).to(device=self.device, dtype=self.dtype) for key, value in state.items()}
        elif isinstance(state, torch.Tensor):
            self.state_function = lambda state: state.to(device=self.device, dtype=self.dtype)
        else:
            self.state_function = lambda state: torch.tensor(state, device=self.device, dtype=self.dtype)

    def first_transition(self, state, action, reward, logprob, done):
        """Creates the first transition and sets up the processing functions.

        Args:
            state (torch.Tensor or dict): The initial state.
            action (torch.Tensor): The initial action.
            reward (float): The initial reward.
            logprob (float): The log-probability of the initial action.
            done (bool): Whether the episode is done.

        Returns:
            Transition: The first transition.
        """
        self.state_shape = {key: value.shape for key, value in state.items()} if isinstance(state, dict) else state.shape

        self.set_state_function(state)

        state = self.state_function(state)
        state_value = self.agent.critic(state)

        self.action_shape = action.shape
        self.reward_shape = reward.shape
        self.logprob_shape = logprob.shape
        self.done_shape = done.shape
        self.state_value_shape = state_value.shape

        self._set_function(action, 'action')
        self._set_function(reward, 'reward')
        self._set_function(logprob, 'logprob')
        self._set_function(done, 'done')
        self._set_function(state_value, 'state_value')

        self.create = self.create_transition

        action = self.action_function(action)
        reward = self.reward_function(reward)
        logprob = self.logprob_function(logprob)
        done = self.done_function(done)
        state_value = self.agent.critic(state)
        state_value = self.state_value_function(state_value)

        return Transition(
            state=state,
            action=action,
            reward=reward,
            logprob=logprob,
            done=done,
            state_value=state_value
        )

    def create_transition(self, state, action, reward, logprob, done):
        """Creates a transition.

        Args:
            state (torch.Tensor or dict): The state.
            action (torch.Tensor): The action.
            reward (float): The reward.
            logprob (float): The log-probability of the action.
            done (bool): Whether the episode is done.

        Returns:
            Transition: The transition.
        """
        state = self.state_function(state)
        action = self.action_function(action)
        reward = self.reward_function(reward)
        logprob = self.logprob_function(logprob)
        done = self.done_function(done)
        state_value = self.agent.critic(state)
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
    """An object that encapsulates a transition in reinforcement learning.

    A transition typically includes the state, action, reward, log-probability of the action, 
    whether the episode is done, and the estimated value of the state.

    Attributes:
        state (torch.Tensor or dict): The state.
        action (torch.Tensor): The action.
        reward (float): The reward.
        logprob (float): The log-probability of the action.
        done (bool): Whether the episode is done.
        state_value (float): The estimated value of the state.
    """
    def __init__(self, *, state, action, reward, logprob, done, state_value) -> None:
        """Initializes a transition.

        Args:
            state (torch.Tensor or dict): The state.
            action (torch.Tensor): The action.
            reward (float): The reward.
            logprob (float): The log-probability of the action.
            done (bool): Whether the episode is done.
            state_value (torch.Tensor): The state value.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.logprob = logprob
        self.done = done
        self.state_value = state_value

    def __repr__(self) -> str:
        """Generates a string representation of the Transition.

        Returns:
            str: A string representation of the Transition.
        """
        return "Transition(state={}, action={}, reward={}, logprob={}, done={}, state_value={})".format(
            self.state, self.action, self.reward, self.logprob, self.done, self.state_value
        )

    def __str__(self) -> str:
        """Generates a string representation of the Transition.

        Returns:
            str: A string representation of the Transition.
        """
        return self.__repr__()
