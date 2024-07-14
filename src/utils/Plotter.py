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
    def get_image_from_batch(test_loader, batch_index, batch_offset, use_context=False):
        i = 0
        for data in test_loader:
            if use_context : 
                batch_x, batch_y, _ , _ = data
            else:
                batch_x, batch_y = data

            if i == batch_index:
                return batch_x[batch_offset]
            i+=1
            if i == batch_index:
                return batch_y[batch_offset]
            i+=1


    @staticmethod
    def plot_best_pairs(best_pairs, original_images, augmented_images, max_images=5, figsize=(12,7), title1="Original Image",title2="Best Match"):

        for i in range(max_images):
            original_image = original_images[i].permute(1, 2, 0).numpy()
            similar_index = best_pairs[i]
            augmented_image = augmented_images[similar_index].permute(1, 2, 0).numpy()
            
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title(title1)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(augmented_image)
            plt.title(title2)
            plt.axis('off')
            plt.show()