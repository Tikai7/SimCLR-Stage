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

    @staticmethod
    def get_image_from_batch(test_loader, batch_index, batch_offset):
        i = 0
        for batch_x, batch_y in test_loader:
            if i == batch_index:
                return batch_x[batch_offset]
            i+=1
            if i == batch_index:
                return batch_y[batch_offset]
            i+=1


    @staticmethod
    def plot_best_pairs(test_loader, best_pairs, batch_size, max_images=1, figsize=(12,7), title1="Image", title2="Best Match"):

        for pairs in best_pairs[:max_images]:
            image_index = pairs[0]
            image_to_match_index = pairs[1]
            image_batch, image_offset = (image_index // batch_size), (image_index % batch_size)
            image_to_match_batch, image_to_match_offset = (image_to_match_index // batch_size), (image_to_match_index % batch_size)

            image = Plotter.get_image_from_batch(test_loader, image_batch, image_offset)
            image_to_match = Plotter.get_image_from_batch(test_loader, image_to_match_batch, image_to_match_offset)

            image = image.permute(1,2,0)
            image_to_match = image_to_match.permute(1,2,0)

            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(title1)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(image_to_match)
            plt.title(title2)
            plt.axis('off')