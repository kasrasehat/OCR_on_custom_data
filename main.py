from augment import Augmentor
import cv2

augmentor = Augmentor('E:/codes_py/Larkimas/Data_source/all_data/background')
processed_image = augmentor.scale_rotate_background('E:/codes_py/Larkimas/Data_source/all_data/0010115706_1.jpg')
#grayscale_image_3ch = augmentor._3ch_grayscale('/home/kasra/PycharmProjects/Larkimas/Data/ai_files_20230625_1/0010038159_2.jpg')
#grayscale_image_1ch = augmentor._1ch_grayscale('/home/kasra/PycharmProjects/Larkimas/Data/ai_files_20230625_1/0010038159_2.jpg')
cv2.imshow('Image with Background', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

