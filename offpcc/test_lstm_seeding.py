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

parameter_identical = True
gradient_identical = True

for p1, p2 in zip(trial1_layer.parameters(), trial2_layer.parameters()):

    if not torch.all(torch.eq(p1, p2)):
        parameter_identical = False

    if not torch.all(torch.eq(p1.grad, p2.grad)):
        gradient_identical = False

print('Output identical?', torch.all(torch.eq(trial1_output, trial2_output)))
print('Parameter identical?', parameter_identical)
print('Gradient identical?', gradient_identical)
