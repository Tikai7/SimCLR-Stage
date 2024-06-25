import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot_images(images, titles, fig_size=(12,7)):
        n = len(images)
        plt.figure(figsize=fig_size)
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')

    @staticmethod
    def plot_compare(img1, img2, title1, title2, fig_size=(12,7)):
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.title(title1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.title(title2)
        plt.axis('off')