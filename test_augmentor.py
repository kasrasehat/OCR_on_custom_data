from augment import Augmentor
import cv2

augmentor = Augmentor('/home/kasra/PycharmProjects/Larkimas/Data/background')
processed_image, angle, transport, scale = augmentor.scale_rotate_background(
    '/home/kasra/kasra_files/data-shenasname/ai_files_20230606/0010202501_0.jpg')
grayscale_image_3ch = augmentor._3ch_grayscale(
    '/home/kasra/PycharmProjects/Larkimas/Data/ai_files_20230625_1/0010038159_2.jpg')
grayscale_image_1ch = augmentor._1ch_grayscale(
    '/home/kasra/PycharmProjects/Larkimas/Data/ai_files_20230625_1/0010038159_2.jpg')
cv2.imshow('Image with Background', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

