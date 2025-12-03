from PIL import Image

import pytesseract

for filename in ['test.png', 'screenshot.png', 'IMG_4788.jpeg', 'IMG_4789.jpeg']:
	print('****************')
	print('FILENAME: ', filename)
	print(pytesseract.image_to_string(filename))
