import torch
import torch.nn.functional as F
import sys


##Testing

def test(model, device, test_loader, criterion):
    try:
        model.eval()
        test_loss = 0
        correct = 0
        test_losses = []
        test_acc = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        test_losses.append(test_loss)
        test_acc.append(100. * correct / len(test_loader.dataset))

    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + " " + type(e).__name__ + " " + str(e))
        sys.exit(1)
    return test_losses, test_acc