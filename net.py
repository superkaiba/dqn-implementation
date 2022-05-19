from torch import nn
import torch
import torch.nn.functional as F


class LinearFeatureExtractor(nn.Module):
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


"""
Both feature extractors expect the following input:
- Scent data (batch_size x scent_history_length x scent_size)
- Vision data (batch_size x num_channels (4 in our case) x scent_history_length x height x width)
- Action data (batch_size x scent_history_length x scent_size)                  (i think we'll one hot encode the actions as like [0, 0, 0, 1] etc.)
"""


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
            in_channels=vision_shape[2], out_channels=2, kernel_size=3
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
        self.output_size = 249
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
        vision_shape=(4, 15, 15),
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
        print(layer_list)

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


if __name__ == "__main__":
    batch_size = 16
    vision_history_length = 5
    action_history_length = 10
    scent_history_length = 10
    vision_shape = (4, 15, 15)
    action_shape = (4,)
    scent_shape = (3,)
    sequence_length = 13
    test_scent_sequence = torch.rand(
        batch_size, sequence_length, scent_history_length, scent_shape[0]
    )
    test_action_sequence = torch.rand(
        batch_size, sequence_length, action_history_length, action_shape[0]
    )
    test_vision_sequence = torch.rand(
        batch_size,
        sequence_length,
        vision_shape[0],
        vision_history_length,
        vision_shape[1],
        vision_shape[2],
    )

    basic_feature_extractor = BasicFeatureExtractor([128, 128])
    lstm_q_network = QNetworkLSTM(basic_feature_extractor)
    outputs = lstm_q_network(
        test_scent_sequence, test_vision_sequence, test_action_sequence
    )
    print("LSTM outputs size:", outputs.shape)

    test_scent = torch.rand(batch_size, scent_history_length, scent_shape[0])
    test_action = torch.rand(batch_size, action_history_length, action_shape[0])
    test_vision = torch.rand(
        batch_size,
        vision_shape[0],
        vision_history_length,
        vision_shape[1],
        vision_shape[2],
    )
    mlp_q_network = QNetworkMLP(feature_extractor=basic_feature_extractor)
    y = mlp_q_network(test_scent, test_vision, test_action)
    print("MLP outputs size:", y.shape)
