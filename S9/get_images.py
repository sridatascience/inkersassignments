import numpy as np
import sys
import torch


def get_images(model,device, test_loader,batch_size,num_imgs=2):
    try:
        model = model.to(device)
        # obtain one batch of test images
        data_iterator = iter(test_loader)
        count1 = 0
        count2 = 0

        misclass = []
        correct = []
        while ((count1 < num_imgs) | (count2 < num_imgs)):
            images, labels = data_iterator.next()
            images, labels = images.to(device), labels.to(device)
            output = model(images)
    
            _, predictions = torch.max(output, 1)
            images = images.cpu().numpy()
    
            for idx in np.arange(batch_size):
                if predictions[idx] != labels[idx]:
                    misclass.append([images[idx], predictions[idx], labels[idx]])
                    count1 = count1 + 1
                    if count1 == num_imgs:
                        break
    
            for idx in np.arange(batch_size):
                if predictions[idx] == labels[idx]:
                    correct.append([images[idx], predictions[idx], labels[idx]])
                    count2 = count2 + 1
                    if count2 == num_imgs:
                        break

    except Exception as e:
        print('get images: Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + " " + type(
            e).__name__ + " " + str(e))
        sys.exit(1)

    return misclass, correct