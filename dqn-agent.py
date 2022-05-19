import os
import queue
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class LinearFeatureExtractor(nn.Module):
    """
    Both feature extractors expect the following input:
    - Scent data (batch_size x scent_history_length x scent_size)
    - Vision data (batch_size x num_channels (4 in our case) x scent_history_length x height x width)
    - Action data (batch_size x scent_history_length x scent_size) (i think we'll one hot encode the actions as like [0, 0, 0, 1] etc.)
    """

    def __init__(self, input_size, hidden_layer_sizes):
        super(LinearFeatureExtractor, self).__init__()

        layer_list = []

        # Input layer
        layer_list.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        layer_list.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 2):
            layer_list.append(
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            )
            layer_list.append(nn.ReLU())

        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)


class CNNFeatureExtractor(nn.Module):
    """
    Applies 3D CNN to vision, basic MLPs for vision and actions, then concatenates them and outputs one feature vector
    """

    def __init__(
        self,
        scent_hidden_layer_sizes=[64, 128, 64],
        action_hidden_layer_sizes=[64, 128, 64],
        vision_history_length=5,
        action_history_length=10,
        scent_history_length=10,
        vision_shape=(15, 15, 4),
        action_shape=(4,),
        scent_shape=(3,),
    ):
        super(CNNFeatureExtractor, self).__init__()

        vision_conv1 = nn.Conv3d(
            in_channels=vision_shape[0], out_channels=2, kernel_size=3
        )
        vision_conv2 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3)

        self.vision_extractor = nn.Sequential(
            vision_conv1, nn.ReLU(), vision_conv2, nn.ReLU()
        )

        self.scent_extractor = LinearFeatureExtractor(
            scent_shape[0] * scent_history_length, scent_hidden_layer_sizes
        )
        self.action_extractor = LinearFeatureExtractor(
            action_shape[0] * action_history_length, action_hidden_layer_sizes
        )
        self.output_size = 377
        # self.final_layer = nn.Linear(377, 4)

    def forward(self, scent, vision, actions):
        scent_features = self.scent_extractor(
            torch.flatten(scent, start_dim=1)
        )  # Don't flatten batch dim
        action_features = self.action_extractor(
            torch.flatten(actions, start_dim=1)
        )  # Don't flatten batch dim
        vision = torch.transpose(vision, 1, 4)
        vision_features = self.vision_extractor(vision)
        vision_features = torch.flatten(vision_features, start_dim=1)

        # Concatenate all features
        features = torch.cat((scent_features, vision_features, action_features), dim=1)

        # Doesn't apply softmax to output layer
        return features


class BasicFeatureExtractor(nn.Module):
    """
    Flattens and concatenates all inputs then applies MLP and outputs feature vector
    """

    def __init__(
        self,
        hidden_layer_sizes,
        vision_history_length=5,
        action_history_length=5,
        scent_history_length=5,
        vision_shape=(15, 15, 4),
        action_shape=(4,),
        scent_shape=(3,),
    ):
        super(BasicFeatureExtractor, self).__init__()

        layer_list = []

        input_size = (
            (scent_shape[0] * scent_history_length)
            + (
                vision_shape[0]
                * vision_shape[1]
                * vision_shape[2]
                * vision_history_length
            )
            + (action_shape[0] * action_history_length)
        )

        # Input layer
        layer_list.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        layer_list.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            layer_list.append(
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
            )
            layer_list.append(nn.ReLU())

        # layer_list.append(nn.Linear(hidden_layer_sizes[-1], action_shape[0]))
        self.output_size = hidden_layer_sizes[-1]
        self.model = nn.Sequential(*layer_list)

    def forward(self, scent, vision, actions):
        x = torch.cat(
            (
                torch.flatten(scent, start_dim=1),
                torch.flatten(vision, start_dim=1),
                torch.flatten(actions, start_dim=1),
            ),
            dim=1,
        )

        # Doesn't apply softmax to output layer
        return self.model(x)


class QNetworkLSTM(nn.Module):
    def __init__(self, feature_extractor):
        super(QNetworkLSTM, self).__init__()
        self.feature_extractor = feature_extractor
        self.lstm = nn.LSTM(
            input_size=feature_extractor.output_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.final_layer = nn.Linear(128, 4)

    def forward(self, scent_sequence, vision_sequence, action_sequence):
        """
        Parameters:
        - scent_sequence: (batch_size x sequence_length x scent_history_length x scent_size)
        - vision_sequence: (batch_size x sequence_length x vision_history_length x num_channels (4 in our case) x height x width)
        - action_sequence: (batch_size x sequence_length x action_history_length x action_size)
        (actually I realized we don't really have a batch dimension unless we do replay buffer ~)
        """
        batch_size, sequence_length = scent_sequence.shape[0], scent_sequence.shape[1]

        scent_sequence = torch.flatten(
            scent_sequence, start_dim=0, end_dim=1
        )  # combine batch and sequence dim
        vision_sequence = torch.flatten(vision_sequence, start_dim=0, end_dim=1)
        action_sequence = torch.flatten(action_sequence, start_dim=0, end_dim=1)

        feature_sequence = self.feature_extractor(
            scent_sequence, vision_sequence, action_sequence
        )

        feature_sequence = feature_sequence.view(
            batch_size, sequence_length, -1
        )  # TODO:NOT SURE IF THIS ACTUALLY WORKS..
        outputs, (hidden_state, cell_state) = self.lstm(feature_sequence)
        outputs = F.relu(outputs)

        return self.final_layer(outputs)


class QNetworkMLP(nn.Module):
    def __init__(self, feature_extractor):
        super(QNetworkMLP, self).__init__()
        self.feature_extractor = feature_extractor
        self.final_layer = nn.Linear(feature_extractor.output_size, 4)

    def forward(self, scent, vision, action):
        """
        Parameters:
        - scent_sequence: (batch_size  x scent_history_length x scent_size)
        - vision_sequence: (batch_size  x vision_history_length x num_channels (4 in our case) x height x width)
        - action_sequence: (batch_size  x action_history_length x action_size)
        """
        features = self.feature_extractor(scent, vision, action)
        features = F.relu(features)
        return self.final_layer(features)


class History:
    def __init__(self, *, stacklen: int, device):
        self._device = device
        self._stacklen = stacklen
        self.reset()

    def get_history(self):
        return (
            torch.tensor(np.array(list(self._scent.queue)), device=self._device),
            torch.tensor(np.array(list(self._vision.queue)), device=self._device),
            torch.tensor(np.array(list(self._feature.queue)), device=self._device),
            torch.tensor(np.array(list(self._action.queue)), device=self._device),
        )

    def add_obs(self, obs):
        self._scent.get()
        self._vision.get()
        self._feature.get()

        self._scent.put(obs[0]),
        self._vision.put(obs[1])
        self._feature.put(obs[2].reshape((15, 15, 4)).astype(np.float32))

    def add_action(self, action):
        self._action.get()
        self._action.put(action)

    def reset(self):
        self._scent = queue.Queue(maxsize=self._stacklen)

        for _ in range(self._stacklen):
            self._scent.put(np.zeros((3,), dtype=np.float32))

        self._vision = queue.Queue(maxsize=self._stacklen)
        for _ in range(self._stacklen):
            self._vision.put(np.zeros((15, 15, 3), dtype=np.float32))

        self._feature = queue.Queue(maxsize=self._stacklen)
        for _ in range(self._stacklen):
            self._feature.put(np.zeros((15, 15, 4), dtype=np.float32))

        self._action = queue.Queue(maxsize=self._stacklen)
        for _ in range(self._stacklen):
            self._action.put(np.zeros((4,), dtype=np.float32))
        self._reward = queue.Queue(maxsize=self._stacklen)

    def __len__(self):
        return self._scent.qsize()


class Agent:
    """The agent class that is to be filled.
    You are allowed to add any method you
    want to this class.
    """

    def __init__(
        self,
        env_specs,
        *,
        mem_len=5,
        boltzmann_temp=10,
        gamma=0.99,
        lr=1e-05,
        target_update_interval=100
    ):
        self.env_specs = env_specs

        self.mem_len = mem_len
        self.boltzmann_temp = boltzmann_temp
        self.gamma = gamma
        self.lr = lr
        self.target_update_interval = target_update_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DEVICE is {self.device}")
        self.online_network = QNetworkMLP(
            BasicFeatureExtractor(
                [256, 256],
                vision_history_length=self.mem_len,
                action_history_length=self.mem_len,
                scent_history_length=self.mem_len,
            )
        )

        # MUST BE SAME AS ABOVE
        self.target_network = copy.deepcopy(self.online_network)

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.history = History(stacklen=self.mem_len, device=self.device)
        self.prev_state_action_val = torch.zeros(
            4, device=self.device, requires_grad=True
        )

        self.online_network.to(self.device)
        self.target_network.to(self.device)

    def load_weights(self, root_path):
        # Add root_path in front of the path of the saved network parameters
        # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
        self.online_network.load_state_dict(torch.load(os.path.join(root_path, "weights.pth")))
        self.target_network.load_state_dict(torch.load(os.path.join(root_path, "weights.pth")))

    def act(self, curr_obs, mode="eval"):
        
        with torch.no_grad():
            self.history.add_obs(curr_obs)
            self.online_network.train(mode != "eval")
            scent, _vision, feature, action = self.history.get_history()
            preds = self.online_network.forward(
                scent.unsqueeze(0), feature.unsqueeze(0), action.unsqueeze(0)
            )

            if mode == "eval":
                action =  preds.squeeze().argmax()
            else:
                probs = torch.softmax(
                    preds.squeeze().double() / self.boltzmann_temp, dim=0
                )
                action = np.random.choice(np.arange(4), p=probs.cpu())

            # 1 hot encode action (not sure if we actually need to do this but seems legit)
            action_arr = np.zeros(4, dtype=np.float32)
            action_arr[action] = 1
            if mode == "eval":
                self.history.add_action(action_arr)
            return action

    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        if timestep % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        self.online_network.train(True)
        scent, _vision, feature, action_hist = self.history.get_history()

        curr_state_action_val = self.online_network.forward(
            scent.unsqueeze(0), feature.unsqueeze(0), action_hist.unsqueeze(0)
        ).squeeze()[action]

        scent, _vision, feature, action_hist = self.history.get_history()
        with torch.no_grad():
            # Don't want to compute gradients back through next Q value
            next_state_action_vals = self.target_network.forward(
                scent.unsqueeze(0), feature.unsqueeze(0), action_hist.unsqueeze(0)
            )

            expected_state_action_val = (
                torch.max(next_state_action_vals) * self.gamma
            ) + reward

        loss = self.criterion(
            curr_state_action_val.unsqueeze(0),
            expected_state_action_val.unsqueeze(0),
        )

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Do this in update() if not doing it in act() (eval mode)
        action_arr = np.zeros(4, dtype=np.float32)
        action_arr[action] = 1
        self.history.add_action(action_arr)
        if done:
            # TBD: Perform training once here instead?
            # TBD: Should we do special training steps at the end of episodes?
            self.history.reset()

    def _save(self, path: str):
        torch.save(self.online_network.state_dict, path)

    def _reset(self):
        self.history.reset()
