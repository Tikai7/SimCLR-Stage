import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot_images(images : list , titles : list, fig_size=(12,7)):
        """
        Plot images with their titles
        Args:
        -----
            images (list): List of images to plot
            titles (list): List of titles for the images
            fig_size (tuple): Size of the figure
        """

        n = len(images)
        plt.figure(figsize=fig_size)
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(images[i], cmap="gray")
            plt.title(titles[i])
            plt.axis('off')

    @staticmethod
    def plot_compare(img1, img2, title1, title2, fig_size=(12,7)):
        """
        Plot two images side by side
        Args:
        -----
            img1 (np.array): First image
            img2 (np.array): Second image
            title1 (str): Title for the first image
            title2 (str): Title for the second image
            fig_size (tuple): Size of the figure
        """
        plt.figure(figsize=fig_size)
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap="gray")
        plt.title(title1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img2,  cmap="gray")
        plt.title(title2)
        plt.axis('off')

    @staticmethod
    def plot_best_pairs(best_pairs, original_images, augmented_images, max_images=5, figsize=(12,7), title1="Original Image",title2="Best Match"):
        """
        Plot the best pairs of images
        Args:
        -----
            best_pairs (list): List of the best pairs
            original_images (list): List of original images
            augmented_images (list): List of augmented images
            max_images (int): Number of images to plot
            figsize (tuple): Size of the figure
            title1 (str): Title for the first image
            title2 (str): Title for the second image
        """

        for i in range(max_images):
            original_image = original_images[i].permute(1, 2, 0).numpy()
            similar_index = best_pairs[i]
            augmented_image = augmented_images[similar_index].permute(1, 2, 0).numpy()
            
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.imshow(original_image, cmap="gray")
            plt.title(title1)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(augmented_image, cmap="gray")
            plt.title(title2)
            plt.axis('off')
            plt.show()