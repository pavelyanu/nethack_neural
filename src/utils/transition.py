import torch

class Transition:
    def __init__(self, *, state, action, reward, logprob, done, state_value=None, dtype=torch.float32) -> None:
        self.dtype = dtype
        self.state = state
        self.action = action
        self.reward = reward
        self.logprob = logprob
        self.done = done
        self.state_value = state_value
        self.feild_keys = ['state', 'action', 'reward', 'logprob', 'done', 'state_value']
        self.unsqueezable = ['action', 'reward', 'logprob', 'done', 'state_value']

    def __repr__(self) -> str:
        return "Transition(state={}, action={}, reward={}, logprob={}, done={}, state_value={})".format(
            self.state, self.action, self.reward, self.logprob, self.done, self.state_value
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_tensor(self):
        for key in self.feild_keys:
            attr = getattr(self, key)
            if isinstance(attr, dict):
                for subkey in attr.keys():
                    if not isinstance(attr[subkey], torch.Tensor):
                        attr[subkey] = torch.tensor(attr[subkey], dtype=self.dtype)
            elif not isinstance(attr, torch.Tensor):
                setattr(self, key, torch.tensor(attr, dtype=self.dtype))

    def unsqueeze(self):
        for key in self.unsqueezable:
            attr = getattr(self, key)
            if not isinstance(attr, torch.Tensor):
                exception = "Cannot unsqueeze {} because it is not a tensor".format(key)
                raise Exception(exception)
            if attr.dim() == 1:
                setattr(self, key, attr.unsqueeze(-1))

    def to(self, device):
        for key in self.feild_keys:
            attr = getattr(self, key)
            if isinstance(attr, dict):
                for subkey in attr.keys():
                    attr[subkey] = attr[subkey].to(device)
            elif isinstance(attr, torch.Tensor):
                setattr(self, key, attr.to(device))
            else:
                exception = "Cannot move {} because it is not a tensor".format(key)
                raise Exception(exception)