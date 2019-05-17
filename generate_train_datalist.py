import os

input_path = '/home/opt603/data/GOPRO_Large/train'

filesName = sorted(os.listdir(input_path))

f = open('edge_train_datalist.txt', 'a+')

for fileName in filesName:
	path_blur = os.path.join(input_path, fileName, 'blur')
	path_sharp = os.path.join(input_path, fileName, 'sharp')
	path_edge = os.path.join(input_path, fileName, 'edge')
	imagesName = sorted(os.listdir(path_blur))
	for imageName in imagesName:
		f.write(os.path.join(fileName, 'sharp', imageName))
		f.write(' ')
		f.write(os.path.join(fileName, 'blur', imageName))
		f.write(' ')
		f.write(os.path.join(fileName, 'edge', imageName))
		f.write('\n')
f.close()
