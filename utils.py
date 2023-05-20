import torch

from tqdm import tqdm


def predict(features, n_samples, delta_t, model):
    is_state = False
    prediction = list()
    model.eval()
    with torch.no_grad():
        xn = torch.tensor([0, 0], dtype=torch.float)
        xf = features
        for _ in tqdm(range(n_samples)):
            output_n, state_n = model.predict(xn, xf=xf, is_state=is_state)

            xn = torch.tensor([delta_t, output_n], dtype=torch.float)

            xf = state_n
            is_state = True

            prediction.append(output_n.item())
    return prediction