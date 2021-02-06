import matplotlib.pyplot as plt
import sys

##Plotting Test Accuracies and Test Losses

def plot_acc_loss(test_losses, test_acc):
	try:
  
		fig, (ax1, ax2) = plt.subplots(2, figsize=(15,10))
		fig.suptitle('Test Loss and Test Accuracy for the models', fontsize=16)
		ax1.plot(test_losses)

		ax1.set_title("Test Loss")

		ax2.plot(test_acc)

		ax2.set_title("Test Accuracy")

		fig.savefig('acc_vs_loss.jpg')
		fig
		fig.show()
	except Exception as e:
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + " " + type(e).__name__ + " " + str(e))
		sys.exit(1)