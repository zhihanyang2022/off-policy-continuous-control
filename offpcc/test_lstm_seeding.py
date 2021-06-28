import torch

input = torch.randn(32, 100, 8)
target = torch.randn(32, 100, 256)


def trial():
    torch.manual_seed(42)
    layer = torch.nn.LSTM(8, 256, batch_first=True)
    output, _ = layer(input)
    loss = torch.mean((output - target) ** 2)
    loss.backward()
    return layer, output


trial1_layer, trial1_output = trial()
trial2_layer, trial2_output = trial()

gradient_identical = True
for p1, p2 in zip(trial1_layer.parameters(), trial2_layer.parameters()):
    print(p1)
    if not torch.eq(p1, p2):
        gradient_identical = False

print('Output identical?', torch.eq(trial1_output, trial2_output))
print('Gradient identical', gradient_identical)
