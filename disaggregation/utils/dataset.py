from torch.utils.data import Dataset
import numpy as np


class DisaggregationSet(Dataset):
    """Disaggregation Set Object

   This dataset object ensures that the transformer network is
   feed correctly and also saves the true targets values for
   posterior evaluation.

    Args:
        x_matrix : Matrix containing the past steps of a univariate
                   time series. With shape (time_steps, window_of_features)

        y_matrix : Matrix containing the future steps of a univariate time series.
                   With shape (time_steps, window_of_features)

        n_time_steps: Number of timesteps used in the entry of the transformer.
    """

    def __init__(self, x_matrix, y_matrix):
        self.encoder_input = x_matrix.reshape(
            x_matrix.shape[0], x_matrix.shape[1], 1)
        self.label = y_matrix

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.label[idx]


def main():
    test_mtx = np.arange(1000).reshape(50, 20)
    test_dataset = DisaggregationSet(test_mtx, test_mtx)
    print(test_dataset[0])


if __name__ == '__main__':
    main()
