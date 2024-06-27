import cv2
import matplotlib.pyplot as plt
import pickle

class SIFTDetector:
    @staticmethod
    def computeSIFT(image):
        try:
            sift_detector = cv2.SIFT_create()
            kp, des = sift_detector.detectAndCompute(image, None)
            return kp, des
        except:
            return None, None
    @staticmethod
    def getBestMatch(image_query_des, liste_descriptor):
        all_matches = []
        bf = cv2.BFMatcher()
        for i,des in enumerate(liste_descriptor):
            matches = bf.knnMatch(image_query_des,des, k=2)
            good = [[m] for m,n in matches if m.distance < 0.75*n.distance]
            all_matches.append((i,len(good),good))

        best_match = max(all_matches, key=lambda x: x[1])
        return best_match

    @staticmethod
    def saveSIFTDescriptors(all_images_sift, path):
        for i, image_sift in enumerate(all_images_sift):
            kp, des = image_sift
            kp_tuples = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
            with open(f"{path}/kp_{i}", 'wb') as f:
                pickle.dump(kp_tuples, f)
            with open(f"{path}/des_{i}", 'wb') as f:
                pickle.dump(des, f)

          
    def loadSIFTDescriptors(path, num_sift):
        all_images_sift = []
        for i in range(num_sift):
            try : 
                with open(f"{path}/kp_{i}", 'rb') as f:
                    kp_tuples = pickle.load(f)

                kp = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=pt[1], _angle=pt[2], 
                                _response=pt[3], _octave=pt[4], _class_id=pt[5]) 
                    for pt in kp_tuples]

                with open(f"{path}/des_{i}", 'rb') as f:
                    des = pickle.load(f)

                all_images_sift.append((kp, des))
            except:
                break
        return all_images_sift
    

    @staticmethod
    def loadSIFTDescriptors(path):
        all_images_sift = []
        for i in range(1000):
            with open(f"{path}/sift_{i}.txt", "r") as f:
                kp = f.readline()
                des = f.readline()
                all_images_sift.append((kp,des))
        return all_images_sift

    @staticmethod
    def displayKeypoints(image_query, image_kp, image_best, image_best_kp, good_kp):
        image_result = cv2.drawMatchesKnn(image_query,image_kp,image_best,image_best_kp, good_kp, None, flags=2)
        plt.figure(figsize=(12,7))
        plt.imshow(image_result)
        plt.title("Keypoints of the two images")
        plt.show()