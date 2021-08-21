import requests
import io
import cv2
from PIL import Image

addr = 'http://localhost:8080'
predict = addr + '/predict'
content_type = 'image/jpeg'
headers = {'content-type': content_type}


source = sys.argv[1]
target = sys.argv[2]

image = cv2.imread(source)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
_, img_encoded = cv2.imencode('.jpg', image)
# send http request with image and receive response
response = requests.post(predict, data=img_encoded.tostring(), headers=headers)
image = Image.open(io.BytesIO(response.content))
image.save(target)