from augment import Augmentor
import cv2

augmentor = Augmentor('E:/codes_py/Larkimas/Data_source/all_data/background')
processed_image, angle, transport, scale = augmentor.scale_rotate_background('E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/ai_files_20230528/0010011692_0.jpg')
grayscale_image_3ch = augmentor._3ch_grayscale('E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/ai_files_20230528/0010011692_0.jpg')
grayscale_image_1ch = augmentor._1ch_grayscale('E:/codes_py/Larkimas/Data_source/UBUNTU 20_0/ai_files_20230528/0010011692_0.jpg')
cv2.imshow('Image with Background', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

