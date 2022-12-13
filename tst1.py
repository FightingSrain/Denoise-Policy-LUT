from PIL import Image

# Open the image
im = Image.open('./img_tst/test001.png')

# Use the ImageDraw module to draw lines on the image
from PIL import ImageDraw
draw = ImageDraw.Draw(im)

# Draw lines on the image to create a "liquid" effect
for i in range(0, im.size[0], 10):
    draw.line((i, 0) + im.size, fill=128)
    draw.line((0, i) + im.size, fill=128)

# Save the liquidized image
im.save('./img_tst/test001s.png')